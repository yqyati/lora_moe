"""Cross-layer routing correlation heatmap: (base expert × LoRA expert) Pearson r
across layers, using per-layer activation frequency profiles.

Stronger structure = AnchorLoRA's W_l successfully ties LoRA experts to specific
base-router signatures (across the depth dimension).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", default="/data/android/yqy/work/lora_moe/data/cfr_probe/deepseek_N512_topk")
    ap.add_argument("--out", default="/data/android/yqy/work/lora_moe/paper/figures/routing_corr_heatmap.pdf")
    args = ap.parse_args()

    probe = Path(args.probe_dir)
    base = np.load(probe / "base_sigs.npy")   # (L, N_E)
    lora = np.load(probe / "lora_sigs.npy")   # (L, N_L)
    L, N_E = base.shape
    _, N_L = lora.shape
    print(f"Loaded: base {base.shape}, lora {lora.shape}")

    # z-score per column (per-expert profile across layers)
    def zscore(x):
        mu = x.mean(0, keepdims=True)
        sd = x.std(0, keepdims=True) + 1e-12
        return (x - mu) / sd

    Bz = zscore(base)         # (L, N_E)
    Lz = zscore(lora)         # (L, N_L)
    corr = (Bz.T @ Lz) / L    # (N_E, N_L)  Pearson r across layers
    print(f"corr matrix: shape={corr.shape}, range [{corr.min():.3f}, {corr.max():.3f}]")

    # Sort LoRA experts by best-correlated base expert + sign
    best_e = np.argmax(np.abs(corr), axis=0)            # (N_L,)
    best_r = corr[best_e, np.arange(N_L)]               # (N_L,) signed
    order = np.lexsort((-np.abs(best_r), best_e))       # primary: best_e, tiebreak: |r| desc
    corr_sorted = corr[:, order]

    print(f"Per-LoRA |best r|: mean={np.abs(best_r).mean():.3f}, median={np.median(np.abs(best_r)):.3f}, p90={np.percentile(np.abs(best_r), 90):.3f}")

    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    im = ax.imshow(corr_sorted, aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7,
                   interpolation="nearest")
    ax.set_xlabel(f"LoRA expert (sorted by preferred base expert)  [N_L = {N_L}]")
    ax.set_ylabel(f"Base expert  [N_E = {N_E}]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Cross-layer routing correlation $r$", fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    # also PNG for quick view
    png = out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
