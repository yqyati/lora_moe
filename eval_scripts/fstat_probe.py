"""Minimal F-statistic probe for a single (base model + adapter) on a probe set.

Captures per-token (base_top_K_idx, lora_routing_probs) per MoE layer via hooks,
then computes F-stat per layer: how much variance in LoRA routing is explained
by grouping tokens by which base expert they route to.

Output: JSON with per-layer F-stat + aggregate.
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
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--adapter_type", default="moe_lora", choices=["moe_lora", "das_lora"])
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_prompts(dataset_name, limit):
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [s["question"] for s in ds.select(range(min(limit, len(ds))))]
    raise ValueError(dataset_name)


def locate_moe_blocks(model):
    SUPPORTED = {"OlmoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock",
                 "DeepseekV2MoE", "DeepseekV2Moe",
                 "DeepseekV3MoE", "DeepseekV3Moe",
                 "Qwen3_5MoeSparseMoeBlock"}
    base = getattr(model, "model", None) or model
    layers = getattr(base, "layers", None) or getattr(base, "model", None).layers
    blocks = []
    for layer in layers:
        block = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if block is not None and type(block).__name__ in SUPPORTED:
            blocks.append(block)
    return blocks


def main():
    args = get_args()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0}
    )
    model.eval()

    if args.adapter_type == "moe_lora":
        from llamafactory.model.model_utils.moe_lora import load_moe_lora_state
        model = load_moe_lora_state(model, args.adapter_path)
    elif args.adapter_type == "das_lora":
        from llamafactory.model.model_utils.das_lora import load_das_lora_state
        model = load_das_lora_state(model, args.adapter_path)
    model.eval()

    moe_blocks = locate_moe_blocks(model)
    n_layers = len(moe_blocks)
    print(f"Found {n_layers} MoE blocks")

    # Per-batch storage: [n_tokens, *]
    batch_base = {}   # layer_idx -> [n_tokens, n_base_experts] (post-softmax probs)
    batch_lora = {}   # layer_idx -> [n_tokens, n_lora_experts] (post-softmax probs)

    def make_base_hook(li):
        def hook(mod, inputs, output):
            with torch.no_grad():
                h = inputs[0]
                if h.dim() == 3:
                    h = h.reshape(-1, h.shape[-1])
                logits = F.linear(h.to(mod.gate.weight.dtype), mod.gate.weight)
                bias = getattr(mod.gate, "bias", None)
                if bias is not None:
                    logits = logits + bias
                probs = torch.softmax(logits.float(), dim=-1)
                batch_base[li] = probs.cpu().numpy()
        return hook

    from llamafactory.model.model_utils.moe_lora import LoRAPool

    handles = []
    for li, block in enumerate(moe_blocks):
        handles.append(block.register_forward_hook(make_base_hook(li)))

    # LoRA hook: track call order
    lora_call_idx = [0]
    def lora_hook_track(mod, inputs, output):
        with torch.no_grad():
            if len(inputs) >= 2:
                li = lora_call_idx[0] % n_layers
                p_L = inputs[1]
                if p_L.dim() == 3:
                    p_L = p_L.reshape(-1, p_L.size(-1))
                batch_lora[li] = p_L.float().cpu().numpy()
                lora_call_idx[0] += 1

    for name, m in model.named_modules():
        if isinstance(m, LoRAPool):
            handles.append(m.register_forward_hook(lora_hook_track))

    prompts = load_prompts(args.dataset, args.limit)
    print(f"Probing {len(prompts)} prompts...")

    # Accumulate per layer
    accum_base = {li: [] for li in range(n_layers)}
    accum_lora = {li: [] for li in range(n_layers)}

    for pi, prompt in enumerate(prompts):
        if pi % 20 == 0:
            print(f"  {pi}/{len(prompts)}", flush=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to("cuda:0")
        if ids.shape[1] < 2:
            continue
        batch_base.clear(); batch_lora.clear()
        lora_call_idx[0] = 0
        with torch.no_grad():
            model(input_ids=ids)
        for li in batch_base:
            if li in batch_lora:
                accum_base[li].append(batch_base[li])
                accum_lora[li].append(batch_lora[li])

    for h in handles:
        h.remove()

    # F-stat per layer
    per_layer = {}
    fstats = []
    for li in range(n_layers):
        if not accum_base[li] or not accum_lora[li]:
            continue
        base_arr = np.concatenate(accum_base[li], axis=0)  # [N, N_E]
        lora_arr = np.concatenate(accum_lora[li], axis=0)  # [N, N_L]
        n = min(len(base_arr), len(lora_arr), 2000)
        idx = np.random.permutation(min(len(base_arr), len(lora_arr)))[:n]
        base_arr = base_arr[idx]; lora_arr = lora_arr[idx]

        base_top1 = base_arr.argmax(axis=-1)  # [N]
        n_E = base_arr.shape[-1]

        grand_mean = lora_arr.mean(axis=0)  # [N_L]
        between = 0.0
        within = 0.0
        n_groups = 0
        for e in range(n_E):
            mask = base_top1 == e
            cnt = mask.sum()
            if cnt < 5: continue
            group = lora_arr[mask]
            gm = group.mean(axis=0)
            between += cnt * ((gm - grand_mean) ** 2).sum()
            within += ((group - gm) ** 2).sum()
            n_groups += 1
        if n_groups <= 1 or within < 1e-8:
            f = 0.0
        else:
            f = (between / (n_groups - 1)) / (within / (n - n_groups))
        per_layer[str(li)] = {"f_statistic": float(f), "n_tokens": int(n), "n_groups": int(n_groups)}
        fstats.append(f)

    result = {
        "per_layer": per_layer,
        "global": {
            "avg_f_statistic": float(np.mean(fstats)) if fstats else 0.0,
            "n_layers_analyzed": len(fstats),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {args.output}")
    print(f"Avg F-stat: {result['global']['avg_f_statistic']:.2f} over {result['global']['n_layers_analyzed']} layers")


if __name__ == "__main__":
    main()
