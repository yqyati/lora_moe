"""CFR Probe: Cross-layer Functional Redundancy Validation.

Question: Do functionally-similar MoE layers (as judged by base router) use
similar LoRA experts? Conversely, do functionally-different layers diverge in
LoRA usage?

If yes → AnchorLoRA's global pool spontaneously organized itself along
functional dimensions inherited from the pretrained MoE structure. This is the
empirical validation for the CFR-based motivation of the global pool design.

Method:
  1. Load trained AnchorLoRA model (base + global LoRA pool + RCP W_l).
  2. Forward a set of probe prompts; hook each MoE block's routing_projection to capture:
       - Base router logits (input[0])  → softmax → base routing distribution
       - p_L = W_l(router_logits, h)    → softmax → LoRA routing distribution
  3. Per layer, average each distribution over all tokens of all prompts.
  4. Compute L×L cosine-similarity matrices for base and LoRA signatures.
  5. Compare the two matrices: Pearson correlation + heatmaps + scatter.

Usage (single GPU, no torchrun):
    CUDA_VISIBLE_DEVICES=0 python eval_scripts/cfr_probe.py \\
        --base_model /data/android/yqy/work/lora_moe/model/DeepSeek-V2-Lite \\
        --adapter_path saves/deepseek_v2_lite/moe_lora/rcp_global_N512_math \\
        --out_dir /data/android/yqy/work/lora_moe/data/cfr_probe/N512 \\
        --limit 300
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from llamafactory.model.model_utils.moe_lora import load_moe_lora_state  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", required=True, help="MoE-LoRA save dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=300, help="number of probe prompts")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--dataset",
        default="gsm8k",
        choices=["gsm8k", "mbpp", "mixed"],
        help="gsm8k = math; mbpp = code; mixed = math + general",
    )
    p.add_argument(
        "--mode",
        default="topk",
        choices=["softmax", "topk"],
        help="softmax: avg soft routing distribution. topk: count of top-k selections (discrete usage). topk is closer to what experts are actually used.",
    )
    p.add_argument("--base_top_k", type=int, default=0, help="base router top-k; 0 = auto-detect from config")
    return p.parse_args()


def collect_probe_prompts(dataset_name: str, limit: int):
    """Return a list of plain text prompts."""
    from datasets import load_dataset

    prompts = []
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        for s in ds.select(range(min(limit, len(ds)))):
            prompts.append(s["question"])
    elif dataset_name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", split="test")
        # sanitized config has 'prompt'; full config has 'text'
        col = "prompt" if "prompt" in ds.column_names else "text"
        for s in ds.select(range(min(limit, len(ds)))):
            prompts.append(s[col])
    elif dataset_name == "mixed":
        ds = load_dataset("gsm8k", "main", split="test")
        for s in ds.select(range(min(limit, len(ds)))):
            prompts.append(s["question"])
        general = [
            "The capital of France is",
            "In computer science, an algorithm is",
            "She walked into the room and",
            "Climate change is caused by",
            "The history of ancient Rome",
        ]
        prompts.extend(general * (limit // len(general)))
    return prompts


def main():
    args = get_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[CFR] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[CFR] Loading adapter: {args.adapter_path}")
    model = load_moe_lora_state(model, args.adapter_path)
    model.eval()

    # Find MoE blocks that carry our routing_projection
    moe_blocks = []
    for name, m in model.named_modules():
        if hasattr(m, "routing_projection") and hasattr(m, "lora_pool"):
            moe_blocks.append((name, m))
    if not moe_blocks:
        raise RuntimeError("No MoE blocks with routing_projection found.")
    print(f"[CFR] Found {len(moe_blocks)} MoE layers with LoRA injection")

    # Auto-detect base top-k if not provided
    base_top_k = args.base_top_k
    if base_top_k <= 0:
        # DeepSeek-V2 / OLMoE: config.num_experts_per_tok
        base_top_k = getattr(model.config, "num_experts_per_tok", None) or 6
    print(f"[CFR] base_top_k = {base_top_k}")

    # LoRA top-k (assume same across all blocks; read from first block's lora_pool)
    lora_top_k = moe_blocks[0][1].lora_pool.top_k
    print(f"[CFR] lora_top_k  = {lora_top_k}")
    print(f"[CFR] mode = {args.mode}")

    # Storage: per-layer accumulator for routing signatures
    # mode=softmax: sum of softmax distributions over tokens
    # mode=topk:    count of top-k selections per expert over tokens
    n_layers = len(moe_blocks)
    n_base = None
    n_lora = None
    base_accum = {}
    lora_accum = {}
    token_count = {i: 0 for i in range(n_layers)}

    hooks = []
    for layer_idx, (name, m) in enumerate(moe_blocks):
        def make_hook(idx):
            def hook(mod, inputs, output):
                nonlocal n_base, n_lora
                # inputs[0] = router_logits  [B*T, N_base]
                # output     = p_L           [B*T, N_lora]
                router_logits = inputs[0]
                p_L = output

                if args.mode == "softmax":
                    base_vec = F.softmax(router_logits.float(), dim=-1).sum(dim=0).cpu()
                    lora_vec = F.softmax(p_L.float(), dim=-1).sum(dim=0).cpu()
                else:  # topk
                    # base top-k indices: [B*T, base_top_k]
                    _, base_topk_idx = router_logits.float().topk(base_top_k, dim=-1)
                    _, lora_topk_idx = p_L.float().topk(lora_top_k, dim=-1)
                    n_b = router_logits.shape[-1]
                    n_l = p_L.shape[-1]
                    base_vec = torch.zeros(n_b)
                    lora_vec = torch.zeros(n_l)
                    base_vec.scatter_add_(0, base_topk_idx.reshape(-1).cpu(), torch.ones(base_topk_idx.numel()))
                    lora_vec.scatter_add_(0, lora_topk_idx.reshape(-1).cpu(), torch.ones(lora_topk_idx.numel()))

                if n_base is None:
                    n_base = base_vec.shape[-1]
                    n_lora = lora_vec.shape[-1]

                if idx not in base_accum:
                    base_accum[idx] = base_vec
                    lora_accum[idx] = lora_vec
                else:
                    base_accum[idx] += base_vec
                    lora_accum[idx] += lora_vec
                token_count[idx] += router_logits.shape[0]
            return hook
        h = m.routing_projection.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Forward each prompt
    prompts = collect_probe_prompts(args.dataset, args.limit)
    print(f"[CFR] {len(prompts)} probe prompts loaded")

    device = next(model.parameters()).device
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            _ = model(**inputs)
            if (i + 1) % 25 == 0:
                print(f"  forward {i + 1}/{len(prompts)}")

    for h in hooks:
        h.remove()

    # Build signature matrices
    base_sigs = []
    lora_sigs = []
    for i in range(n_layers):
        n_tok = max(token_count[i], 1)
        base_sigs.append((base_accum[i] / n_tok).numpy())
        lora_sigs.append((lora_accum[i] / n_tok).numpy())
    base_sigs = np.stack(base_sigs)   # [L, N_base]
    lora_sigs = np.stack(lora_sigs)   # [L, N_lora]

    def cosine_matrix(M):
        norms = np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12
        Mn = M / norms
        return Mn @ Mn.T

    base_sim = cosine_matrix(base_sigs)
    lora_sim = cosine_matrix(lora_sigs)

    np.save(out / "base_sim.npy", base_sim)
    np.save(out / "lora_sim.npy", lora_sim)
    np.save(out / "base_sigs.npy", base_sigs)
    np.save(out / "lora_sigs.npy", lora_sigs)
    with open(out / "layer_names.json", "w") as f:
        json.dump([n for n, _ in moe_blocks], f, indent=2)

    # Correlation: upper triangle (excluding diagonal)
    L = n_layers
    triu = np.triu_indices(L, k=1)
    base_vec = base_sim[triu]
    lora_vec = lora_sim[triu]

    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(base_vec, lora_vec)
    r_spearman, p_spearman = spearmanr(base_vec, lora_vec)

    summary = {
        "n_layers": int(L),
        "n_base_experts": int(n_base),
        "n_lora_experts": int(n_lora),
        "n_prompts": int(len(prompts)),
        "total_tokens_per_layer_avg": float(np.mean(list(token_count.values()))),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
    }
    with open(out / "cfr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[CFR] Summary:")
    print(json.dumps(summary, indent=2))

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mat, title in [
            (axes[0], base_sim, f"Base routing similarity (L={L})"),
            (axes[1], lora_sim, f"LoRA routing similarity (L={L})"),
        ]:
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Layer")
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(out / "cfr_heatmaps.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(base_vec, lora_vec, alpha=0.6, s=15)
        ax.set_xlabel("Base routing similarity (layer-pair)")
        ax.set_ylabel("LoRA routing similarity (layer-pair)")
        ax.set_title(
            f"CFR scatter | Pearson r={r_pearson:.3f} (p={p_pearson:.2e})\n"
            f"Spearman r={r_spearman:.3f}"
        )
        # 45° reference
        lim = [min(base_vec.min(), lora_vec.min()), max(base_vec.max(), lora_vec.max())]
        ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
        plt.tight_layout()
        plt.savefig(out / "cfr_scatter.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[CFR] Plots saved to {out}")
    except ImportError:
        print("[CFR] matplotlib not available; skipped plots")

    print(f"[CFR] Done. All outputs in {out}")


if __name__ == "__main__":
    main()
