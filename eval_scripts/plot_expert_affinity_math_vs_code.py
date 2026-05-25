"""Math vs code expert-affinity comparison at N_L=512.

Two side-by-side panels: math probe (GSM8K) and code probe (MBPP).
Each sorted by its own preferred depth.
Same discrete colormap as the pool-size figure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

SIGS_MATH = Path("/data/android/yqy/work/lora_moe/data/cfr_probe/deepseek_N512_topk/lora_sigs.npy")
SIGS_CODE = Path("/data/android/yqy/work/lora_moe/data/cfr_probe/deepseek_N512_topk_code/lora_sigs.npy")
OUT = Path("/data/android/yqy/work/lora_moe/paper/figures/expert_affinity_math_vs_code")


def sort_by_depth(f):
    """f: [L, N]; return [N, L] sorted by per-expert preferred depth, per-row normalized."""
    fL = f.T
    L = fL.shape[1]
    layer_idx = np.arange(L)
    pref = (fL * layer_idx).sum(axis=1) / (fL.sum(axis=1) + 1e-12)
    order = np.argsort(pref)
    sorted_ = fL[order]
    norm = sorted_ / (sorted_.max(axis=1, keepdims=True) + 1e-12)
    return norm


# Discrete cmap (matches expert_affinity_pool_size figure)
bin_edges = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]
bin_colors = [
    "#ffffff",  # 0.00-0.50  white
    "#eef3f8",  # 0.50-0.70  near-white
    "#cee0f2",  # 0.70-0.85  light blue
    "#6baed6",  # 0.85-0.95  blue
    "#542788",  # 0.95-1.00  deep purple
]
cmap = mcolors.ListedColormap(bin_colors)
norm = mcolors.BoundaryNorm(bin_edges, cmap.N)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 6.5))

for ax, sigs_path, label in zip(
    axes,
    [SIGS_MATH, SIGS_CODE],
    ["Math (GSM8K probe)", "Code (MBPP probe)"],
):
    f = np.load(sigs_path)
    M = sort_by_depth(f)
    im = ax.imshow(
        M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
    )
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Layer index $\\ell$")

axes[0].set_ylabel("Expert (sorted by preferred depth), $N_L = 512$")

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.8)
cbar.set_label("Activation (per-row normalized)", rotation=270, labelpad=12)

plt.subplots_adjust(left=0.08, right=0.92, wspace=0.15)
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", bbox_inches="tight", dpi=130)
print("Saved:", str(OUT) + ".pdf")
print("Saved:", str(OUT) + ".png")
