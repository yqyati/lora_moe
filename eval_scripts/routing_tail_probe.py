"""Routing-tail probe: for a given (base model, adapter), collect per-token
LM loss and base-router rarity on a validation set. Save as JSONL.

Workflow (single GPU):
  1. Forward each prompt with labels=input_ids, hook base router → record top-K
     indices per (layer, token), record per-token loss.
  2. After all prompts, compute per-layer expert frequency f(e, ℓ) and per-token
     rarity r(t) = mean_ℓ ((1/K) * Σ_{e ∈ topK(t,ℓ)} 1 / f(e, ℓ)).
  3. Save (rarity, loss) pairs and metadata.

Bucketing + plotting is done by a separate script.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/android/yqy/work/LlamaFactory/src")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None, help="MoE-LoRA adapter dir (skip for base-only)")
    p.add_argument("--adapter_type", default="moe_lora", choices=["moe_lora", "das_lora", "none"],
                   help="moe_lora | das_lora | none (base only)")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--top_k", type=int, default=6, help="base router top-k (DeepSeek-V2-Lite default 6)")
    p.add_argument("--output", required=True, help="output JSONL with (rarity, loss) pairs")
    return p.parse_args()


def load_prompts(dataset_name: str, limit: int):
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [s["question"] for s in ds.select(range(min(limit, len(ds))))]
    if dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [s["problem"] for s in ds.select(range(min(limit, len(ds))))]
    raise ValueError(f"unknown dataset: {dataset_name}")


def locate_moe_blocks(model):
    """Return list of MoE block modules (have .gate)."""
    blocks = []
    SUPPORTED = {"OlmoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock",
                 "DeepseekV2MoE", "DeepseekV2Moe",
                 "DeepseekV3MoE", "DeepseekV3Moe",
                 "Qwen3_5MoeSparseMoeBlock"}
    base = getattr(model, "model", None) or model
    layers = getattr(base, "layers", None) or getattr(base, "model", None).layers
    for layer in layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if block is not None and type(block).__name__ in SUPPORTED:
            blocks.append(block)
    return blocks


def main():
    args = get_args()
    print(f"Loading tokenizer / base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0}
    )
    model.eval()

    if args.adapter_type == "moe_lora" and args.adapter_path:
        from llamafactory.model.model_utils.moe_lora import load_moe_lora_state
        model = load_moe_lora_state(model, args.adapter_path)
    elif args.adapter_type == "das_lora" and args.adapter_path:
        from llamafactory.model.model_utils.das_lora import load_das_lora_state
        model = load_das_lora_state(model, args.adapter_path)
    model.eval()

    moe_blocks = locate_moe_blocks(model)
    n_layers = len(moe_blocks)
    print(f"Found {n_layers} MoE blocks")
    K = args.top_k

    # ---- hook: record (layer_idx, top_k_indices_per_token) ----
    # Storage per-batch (reset each batch)
    batch_topk = {}  # layer_idx -> [n_tokens, K]

    def make_hook(li):
        def hook(module, inputs, output):
            with torch.no_grad():
                h = inputs[0]
                if h.dim() == 3:
                    h = h.reshape(-1, h.shape[-1])
                router_logits = F.linear(h.to(module.gate.weight.dtype), module.gate.weight)
                bias = getattr(module.gate, "bias", None)
                if bias is not None:
                    router_logits = router_logits + bias
                _, idx = router_logits.topk(K, dim=-1)  # [n_tokens, K]
                batch_topk[li] = idx.cpu().numpy().astype(np.int32)
        return hook

    handles = []
    for li, block in enumerate(moe_blocks):
        handles.append(block.register_forward_pre_hook(
            lambda mod, inp, li=li: make_hook(li)(mod, inp, None)
        ))

    prompts = load_prompts(args.dataset, args.limit)
    print(f"Prompts loaded: {len(prompts)}")

    # collect per-token records:
    #   loss_per_token (float), topk_per_layer (n_layers × K ints)
    all_records = []  # list of dict (rarity placeholder, loss, topk)
    # also accumulate global freq
    freq = np.zeros((n_layers, 64), dtype=np.int64)  # 64 = DeepSeek-V2-Lite N_E; resize if needed

    for pi, prompt in enumerate(prompts):
        if pi % 20 == 0:
            print(f"  prompt {pi}/{len(prompts)}", flush=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to("cuda:0")
        seq_len = ids.shape[1]
        if seq_len < 2:
            continue
        batch_topk.clear()
        with torch.no_grad():
            out = model(input_ids=ids)
            logits = out.logits  # [1, T, V]
            shift_logits = logits[:, :-1].float()
            shift_labels = ids[:, 1:]
            losses = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(1, seq_len - 1)
        # for each non-pad token position t in [0, seq_len-2]:
        #   loss = losses[0, t]
        #   topk at layer ℓ = batch_topk[ℓ][t, :]
        # Note: hook captured n_tokens = seq_len; we use positions [0, seq_len-2]
        for li in batch_topk:
            mx = batch_topk[li].max() + 1
            if mx > freq.shape[1]:
                # resize freq array
                new_freq = np.zeros((n_layers, mx), dtype=np.int64)
                new_freq[:, :freq.shape[1]] = freq
                freq = new_freq
            for t in range(seq_len - 1):
                for k in range(K):
                    freq[li, batch_topk[li][t, k]] += 1

        # record
        rec_topk = np.stack([batch_topk[li][:seq_len - 1, :] for li in range(n_layers)], axis=0)  # [L, T-1, K]
        for t in range(seq_len - 1):
            all_records.append({
                "loss": float(losses[0, t].item()),
                "topk": rec_topk[:, t, :].tolist(),  # [L][K]
            })

    for h in handles:
        h.remove()

    # ---- compute per-token rarity ----
    total_hits = freq.sum(axis=1, keepdims=True)
    freq_norm = freq / np.maximum(total_hits, 1)  # P(e | layer)
    inv_freq = 1.0 / np.maximum(freq_norm, 1e-8)  # [L, N_E]

    print(f"Total records: {len(all_records)}")
    for rec in all_records:
        topk = np.array(rec["topk"])  # [L, K]
        # per-layer rarity = mean over K of inv_freq[ℓ, e]
        per_layer = np.zeros(n_layers, dtype=np.float64)
        for li in range(n_layers):
            per_layer[li] = inv_freq[li, topk[li]].mean()
        rec["rarity"] = float(per_layer.mean())
        del rec["topk"]   # don't save bloated indices

    # save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    # also save freq for debugging
    np.save(out.with_suffix(".freq.npy"), freq)
    print(f"Saved {len(all_records)} records → {out}")
    rarities = np.array([r["rarity"] for r in all_records])
    losses_arr = np.array([r["loss"] for r in all_records])
    print(f"Rarity: min={rarities.min():.2f} median={np.median(rarities):.2f} max={rarities.max():.2f}")
    print(f"Loss:   min={losses_arr.min():.3f} median={np.median(losses_arr):.3f} max={losses_arr.max():.3f}")


if __name__ == "__main__":
    main()
