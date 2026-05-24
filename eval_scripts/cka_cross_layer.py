"""Cross-layer CKA: similarity between MoE block outputs across layers.
Hooks the output of each MoE block on a probe set, computes Linear CKA
between every layer pair, plots the resulting L×L heatmap.

A block-diagonal structure (nearby layers similar) supports the claim that
a global shared LoRA pool can exploit cross-layer reusable subspace.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/data/android/yqy/work/LlamaFactory/src")


SUPPORTED = {"OlmoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock",
             "DeepseekV2MoE", "DeepseekV2Moe",
             "DeepseekV3MoE", "DeepseekV3Moe",
             "Qwen3_5MoeSparseMoeBlock"}


def linear_cka(X, Y):
    """Linear CKA between [N, D_x] and [N, D_y] activation matrices (centered)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    XtY = X.T @ Y
    num = (XtY * XtY).sum()
    XtX = X.T @ X
    YtY = Y.T @ Y
    den = np.sqrt((XtX * XtX).sum() * (YtY * YtY).sum()) + 1e-12
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="/data/android/yqy/work/lora_moe/model/DeepSeek-V2-Lite")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_tokens_per_layer", type=int, default=20000, help="cap for CKA computation")
    ap.add_argument("--out", default="/data/android/yqy/work/lora_moe/paper/figures/cka_cross_layer.pdf")
    ap.add_argument("--save_json", default="/data/android/yqy/work/lora_moe/data/cka_cross_layer.json")
    args = ap.parse_args()

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0}
    )
    model.eval()

    base = getattr(model, "model", None) or model
    layers_container = getattr(base, "layers", None) or getattr(base, "model", None).layers
    moe_blocks = []
    for layer in layers_container:
        block = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if block is not None and type(block).__name__ in SUPPORTED:
            moe_blocks.append(block)
    L = len(moe_blocks)
    print(f"Found {L} MoE blocks")

    # Per-layer activation collector. Hook MoE block forward output.
    acts = [[] for _ in range(L)]

    def make_hook(li):
        def hook(mod, inputs, output):
            with torch.no_grad():
                out = output[0] if isinstance(output, tuple) else output
                # [B, T, D] → [B*T, D]
                if out.dim() == 3:
                    out = out.reshape(-1, out.shape[-1])
                acts[li].append(out.float().cpu().numpy())
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(moe_blocks)]

    print(f"Loading gsm8k prompts (limit={args.limit})...")
    ds = load_dataset("gsm8k", "main", split="test")
    prompts = [s["question"] for s in ds.select(range(min(args.limit, len(ds))))]

    print("Probing...")
    for pi, prompt in enumerate(prompts):
        if pi % 20 == 0:
            print(f"  {pi}/{len(prompts)}", flush=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to("cuda:0")
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            model(input_ids=ids)

    for h in handles:
        h.remove()

    # Stack per-layer activations, cap at max_tokens_per_layer
    print("Stacking activations + computing CKA...")
    Xs = []
    for li in range(L):
        if not acts[li]:
            Xs.append(None); continue
        X = np.concatenate(acts[li], axis=0)
        if X.shape[0] > args.max_tokens_per_layer:
            idx = np.random.permutation(X.shape[0])[:args.max_tokens_per_layer]
            X = X[idx]
        Xs.append(X.astype(np.float32))
        print(f"  layer {li}: shape={X.shape}")

    # CKA matrix L×L, sample to same N rows across all layers for fair comparison
    N = min(X.shape[0] for X in Xs if X is not None)
    Xs = [X[:N] if X is not None else None for X in Xs]

    cka = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            if i > j:
                cka[i, j] = cka[j, i]
            elif Xs[i] is None or Xs[j] is None:
                cka[i, j] = 0.0
            else:
                cka[i, j] = linear_cka(Xs[i], Xs[j])

    # Save matrix
    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    import json
    json.dump({"cka": cka.tolist(), "n_layers": L, "n_tokens": N}, open(args.save_json, "w"))
    print(f"Saved CKA matrix → {args.save_json}")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cka, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("Layer index $\\ell_j$")
    ax.set_ylabel("Layer index $\\ell_i$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Linear CKA", fontsize=9)
    ax.set_xticks([0, L//4, L//2, 3*L//4, L-1])
    ax.set_yticks([0, L//4, L//2, 3*L//4, L-1])
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"Saved figure → {out}")

    # Off-diagonal summary
    triu_mask = np.triu(np.ones_like(cka, dtype=bool), k=1)
    print(f"\nOff-diagonal CKA summary (upper triangle):")
    print(f"  mean={cka[triu_mask].mean():.3f}, min={cka[triu_mask].min():.3f}, max={cka[triu_mask].max():.3f}")
    # adjacent vs distant
    adj = [cka[i, i+1] for i in range(L-1)]
    distant = [cka[i, j] for i in range(L) for j in range(L) if abs(i-j) >= 10]
    print(f"  adjacent pairs (|i-j|=1): mean={np.mean(adj):.3f}, n={len(adj)}")
    if distant:
        print(f"  distant pairs (|i-j|>=10): mean={np.mean(distant):.3f}, n={len(distant)}")


if __name__ == "__main__":
    main()
