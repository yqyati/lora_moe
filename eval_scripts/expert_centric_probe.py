"""Expert-centric layer affinity analysis.

For each LoRA expert, compute:
  - Preferred depth = weighted mean of layer index (heavier = more used)
  - Layer entropy = spread of usage across layers (low = layer-specific, high = layer-agnostic)
  - Total usage = how often the expert is active overall

Then build:
  1. Sorted expert × layer heatmap (experts sorted by preferred depth)
     → visualizes whether experts cluster into "early-layer experts" / "late-layer experts"
  2. Histogram of preferred depths (distribution across experts)
  3. Scatter: total usage vs layer entropy (sees if low-entropy experts are also low-usage = dead specialists)
"""
import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe_dir", required=True, help="dir containing lora_sigs.npy")
    p.add_argument("--out_dir", default=None, help="defaults to probe_dir/expert_centric")
    p.add_argument("--title", default="", help="title prefix for plots")
    args = p.parse_args()

    probe_dir = Path(args.probe_dir)
    out_dir = Path(args.out_dir) if args.out_dir else probe_dir / "expert_centric"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigs = np.load(probe_dir / "lora_sigs.npy")  # [L, N]
    L, N = sigs.shape
    print(f"Loaded sigs: L={L} layers, N={N} experts")

    # Transpose: [N, L]
    expert_layer = sigs.T  # row = expert, col = layer activation freq

    # Normalize each expert's layer profile to a probability distribution
    expert_sum = expert_layer.sum(axis=1, keepdims=True)  # [N, 1]
    valid_expert = (expert_sum.squeeze() > 0)
    expert_dist = np.zeros_like(expert_layer)
    expert_dist[valid_expert] = expert_layer[valid_expert] / expert_sum[valid_expert]

    # Per-expert metrics
    layer_indices = np.arange(L)
    preferred_depth = (expert_dist * layer_indices).sum(axis=1)  # [N]
    # Layer entropy of usage distribution (low = layer-specific)
    layer_entropy = -(expert_dist * np.log(expert_dist + 1e-12)).sum(axis=1)
    # Total usage (raw count)
    total_usage = expert_layer.sum(axis=1)

    # Dead experts: never activated (or near-zero)
    n_dead = int((total_usage < 1.0).sum())
    print(f"Dead experts (usage < 1.0): {n_dead} / {N}")

    # Sort experts by preferred depth
    sort_idx = np.argsort(preferred_depth)
    sorted_layer_expert = expert_layer[sort_idx]  # [N_sorted, L]

    # For viz, also build a NORMALIZED-per-expert version so dead experts don't dominate
    sorted_norm = expert_dist[sort_idx]

    # ============ Figure ============
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30)

    # Panel 1 (large): expert × layer heatmap (raw counts)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im1 = ax1.imshow(sorted_layer_expert, aspect="auto", cmap="viridis",
                     interpolation="nearest")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Expert (sorted by preferred depth)")
    ax1.set_title(f"{args.title} | Expert × Layer activation (sorted)\nN={N} experts, L={L} layers")
    plt.colorbar(im1, ax=ax1, label="activation count")

    # Panel 2 (large): same but per-expert normalized (shows preference shape)
    ax2 = fig.add_subplot(gs[0:2, 2])
    im2 = ax2.imshow(sorted_norm, aspect="auto", cmap="magma",
                     interpolation="nearest", vmin=0, vmax=sorted_norm.max())
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Expert (sorted)")
    ax2.set_title("per-expert normalized\n(reveals layer preference)")
    plt.colorbar(im2, ax=ax2)

    # Panel 3: histogram of preferred depths
    ax3 = fig.add_subplot(gs[2, 0])
    used = preferred_depth[valid_expert]
    ax3.hist(used, bins=min(L, 30), color="steelblue", edgecolor="black", alpha=0.8)
    ax3.set_xlabel("Preferred depth (layer index)")
    ax3.set_ylabel("# experts")
    ax3.set_title(f"Distribution of expert preferred depth\n(n_alive={int(valid_expert.sum())}, n_dead={n_dead})")
    ax3.axvline(L / 2, ls="--", color="gray", alpha=0.5, label="mid")
    ax3.legend(loc="upper right", fontsize=8)

    # Panel 4: histogram of layer entropy
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(layer_entropy[valid_expert], bins=30, color="darkorange", edgecolor="black", alpha=0.8)
    ax4.axvline(np.log(L), ls="--", color="red", alpha=0.6, label="max entropy (uniform across layers)")
    ax4.axvline(0, ls="--", color="green", alpha=0.6, label="zero (single-layer specialist)")
    ax4.set_xlabel("Layer entropy of expert")
    ax4.set_ylabel("# experts")
    ax4.set_title("Layer-spread of each expert\n(low = layer-specific, high = layer-agnostic)")
    ax4.legend(loc="upper left", fontsize=7)

    # Panel 5: usage vs entropy scatter
    ax5 = fig.add_subplot(gs[2, 2])
    sc = ax5.scatter(layer_entropy[valid_expert], total_usage[valid_expert],
                     c=preferred_depth[valid_expert], cmap="coolwarm", s=14, alpha=0.7)
    ax5.set_xlabel("Layer entropy")
    ax5.set_ylabel("Total usage")
    ax5.set_yscale("log")
    ax5.set_title("Usage vs spread\n(color = preferred depth)")
    plt.colorbar(sc, ax=ax5, label="preferred depth")

    plt.savefig(out_dir / "expert_layer_affinity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Save numerical summary
    summary = {
        "L": int(L),
        "N": int(N),
        "n_dead": n_dead,
        "n_alive": int(valid_expert.sum()),
        "preferred_depth_mean": float(used.mean()),
        "preferred_depth_std": float(used.std()),
        "layer_entropy_mean_alive": float(layer_entropy[valid_expert].mean()),
        "layer_entropy_max_possible": float(np.log(L)),
        "fraction_below_half_entropy": float((layer_entropy[valid_expert] < 0.5 * np.log(L)).mean()),
    }
    with open(out_dir / "expert_centric_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFigure saved to: {out_dir / 'expert_layer_affinity.png'}")


if __name__ == "__main__":
    main()
