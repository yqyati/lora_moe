"""Generate single-column expert-affinity heatmap (math probe only).

Input:  lora_sigs.npy  [L, N]  per-layer expert usage frequency
Output: expert_affinity_math.{pdf,png}

Reproduces Figure (Section 6.2, F1) of the AnchorLoRA paper.

Usage:
    python plot_expert_affinity_math.py \
        --sigs /data/android/yqy/work/lora_moe/data/cfr_probe/deepseek_N512_topk/lora_sigs.npy \
        --out_dir /data/android/yqy/work/lora_moe/paper/figures
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigs", required=True, help="path to lora_sigs.npy [L, N]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--basename", default="expert_affinity_math")
    args = ap.parse_args()

    # 1. Load and transpose: [L, N] -> [N, L]
    math_sigs = np.load(args.sigs)
    math_E = math_sigs.T        # [N, L]
    L, N = math_E.shape[1], math_E.shape[0]

    # 2. Per-expert preferred depth (weighted-mean layer index)
    layer_idx = np.arange(L)
    pref_depth = (math_E * layer_idx).sum(axis=1) / (math_E.sum(axis=1) + 1e-12)

    # 3. Sort experts by preferred depth
    sort_order = np.argsort(pref_depth)
    math_sorted = math_E[sort_order]

    # 4. Per-row normalization (so each expert's pattern is visually comparable)
    math_norm = math_sorted / (math_sorted.max(axis=1, keepdims=True) + 1e-12)

    # 5. Plot — single-column figure (~3.5in wide for ACL/EMNLP)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
    })
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    im = ax.imshow(
        math_norm, aspect="auto", cmap="viridis",
        interpolation="nearest", vmin=0, vmax=1,
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"Expert (sorted by preferred depth), N={N}")
    ax.set_xticks(np.arange(0, L, 5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activation (per-row norm)", fontsize=9)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.basename}.pdf"
    png_path = out_dir / f"{args.basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
