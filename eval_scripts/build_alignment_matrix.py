"""Build P(LoRA expert | MoE expert) alignment matrices for Figure 1 heatmap.

Reuses analyze_routing.py's hook infrastructure to collect per-layer routing data
from both an RCP checkpoint and an independent-routing baseline, then aggregates
N_E × N_L cross-tabulation matrices.

Output: .npz containing 'rcp_matrix' (N_E × N_L), 'baseline_matrix' (N_E × N_L),
plus per-layer matrices and metadata. Consumed by plot_routing_heatmap.py.

Usage:
  python eval_scripts/build_alignment_matrix.py \\
      --base_model allenai/OLMoE-1B-7B-0924 \\
      --ours_path saves/olmoe/moe_lora/v2_global_pool128_best \\
      --baseline_path saves/olmoe/moe_lora/baseline2_independent_global \\
      --dataset gsm8k --limit 200 --batch_size 4 \\
      --output analysis_results/routing_alignment/alignment_matrices_gsm8k.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

# allow running from LlamaFactory root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_routing import (
    AlignmentCollector,
    load_dataset,
    load_model_with_adapter,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--ours_path", required=True)
    p.add_argument("--baseline_path", required=True)
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", required=True,
                   help="Path to output .npz file")
    p.add_argument("--use_top_k_weighted", action="store_true",
                   help="If set, use top-k weighted soft assignment instead of top-1.")
    return p.parse_args()


def collect_per_layer_data(model, tokenizer, prompts, batch_size, label):
    """Forward all prompts and return {layer_idx: (moe_logits_NxNE, lora_probs_NxNL)}."""
    from collections import defaultdict

    print(f"\n{'='*60}\nCollecting routing data: {label}\n{'='*60}")
    collector = AlignmentCollector(model)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        collector.start_batch()
        with torch.no_grad():
            model(**inputs)
        collector.end_batch()

        if (i // batch_size) % 10 == 0:
            print(f"  processed {min(i + batch_size, len(prompts))}/{len(prompts)}")

    moe_by_layer = defaultdict(list)
    lora_by_layer = defaultdict(list)
    for layer_idx, data in collector.moe_routing_data:
        moe_by_layer[layer_idx].append(data)
    for layer_idx, data in collector.lora_routing_data:
        lora_by_layer[layer_idx].append(data)

    collector.remove_hooks()

    out = {}
    for layer in sorted(set(moe_by_layer) & set(lora_by_layer)):
        moe_logits = torch.cat(moe_by_layer[layer], dim=0)   # [N, N_E]
        lora_probs = torch.cat(lora_by_layer[layer], dim=0)  # [N, N_L]
        n = min(moe_logits.size(0), lora_probs.size(0))
        moe_probs = torch.softmax(moe_logits[:n], dim=-1)
        lora_probs = lora_probs[:n]
        out[layer] = (moe_probs.numpy(), lora_probs.numpy())
    return out


def build_cross_tab(moe_probs, lora_probs, use_top_k_weighted=False):
    """Return P(LoRA expert | MoE expert), shape [N_E, N_L]."""
    N, N_E = moe_probs.shape
    _, N_L = lora_probs.shape
    matrix = np.zeros((N_E, N_L), dtype=np.float64)

    if use_top_k_weighted:
        # Each token contributes moe_probs[t, i] * lora_probs[t, j] to cell (i, j).
        matrix = moe_probs.T @ lora_probs  # [N_E, N_L]
        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        matrix /= row_sum
    else:
        # Top-1 assignment: count co-occurrences of argmax(MoE) and argmax(LoRA).
        moe_top1 = moe_probs.argmax(axis=1)
        lora_top1 = lora_probs.argmax(axis=1)
        for i, j in zip(moe_top1, lora_top1):
            matrix[i, j] += 1.0
        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        matrix /= row_sum

    return matrix


def aggregate(per_layer_data, use_top_k_weighted):
    """Build per-layer matrices and a globally pooled matrix."""
    per_layer_mats = {}
    pooled_moe = []
    pooled_lora = []
    for layer, (mp, lp) in sorted(per_layer_data.items()):
        per_layer_mats[layer] = build_cross_tab(mp, lp, use_top_k_weighted)
        pooled_moe.append(mp)
        pooled_lora.append(lp)
    pooled = build_cross_tab(np.concatenate(pooled_moe, axis=0),
                             np.concatenate(pooled_lora, axis=0),
                             use_top_k_weighted)
    return per_layer_mats, pooled


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    prompts = load_dataset(args.dataset, args.limit)
    print(f"Loaded {len(prompts)} prompts")

    # RCP
    print("\n[1/2] Loading RCP (ours)...")
    tok, model = load_model_with_adapter(args.base_model, args.ours_path)
    rcp_data = collect_per_layer_data(model, tok, prompts, args.batch_size, "RCP")
    del model
    torch.cuda.empty_cache()

    # Baseline
    print("\n[2/2] Loading independent-routing baseline...")
    _, model = load_model_with_adapter(args.base_model, args.baseline_path)
    bl_data = collect_per_layer_data(model, tok, prompts, args.batch_size, "Baseline")
    del model
    torch.cuda.empty_cache()

    print("\nAggregating cross-tabulation matrices...")
    rcp_per_layer, rcp_pooled = aggregate(rcp_data, args.use_top_k_weighted)
    bl_per_layer, bl_pooled = aggregate(bl_data, args.use_top_k_weighted)

    layers = sorted(rcp_per_layer.keys())
    n_layers = len(layers)
    N_E = rcp_pooled.shape[0]
    N_L = rcp_pooled.shape[1]

    save_dict = {
        "rcp_matrix": rcp_pooled,         # [N_E, N_L]
        "baseline_matrix": bl_pooled,
        "rcp_per_layer": np.stack([rcp_per_layer[L] for L in layers]),
        "baseline_per_layer": np.stack([bl_per_layer[L] for L in layers]),
        "layers": np.array(layers),
        "n_E": N_E,
        "n_L": N_L,
        "dataset": args.dataset,
        "n_prompts": len(prompts),
        "aggregation": "top_k_weighted" if args.use_top_k_weighted else "top_1",
        "ours_path": args.ours_path,
        "baseline_path": args.baseline_path,
    }
    np.savez_compressed(args.output, **save_dict)
    print(f"\nSaved to: {args.output}")
    print(f"  N_E={N_E}, N_L={N_L}, n_layers={n_layers}")
    print(f"  rcp_pooled sum per row (should be ~1): "
          f"{rcp_pooled.sum(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
