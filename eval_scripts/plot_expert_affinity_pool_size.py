"""Combine 3 expert-affinity heatmaps (N=128, 256, 512) into one figure.

Output: paper/figures/expert_affinity_pool_size.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/data/android/yqy/work/lora_moe/data/cfr_probe")
SIGS = {
    128: ROOT / "deepseek_N128_r34_matched_topk" / "lora_sigs.npy",
    256: ROOT / "deepseek_N256_r17_matched_topk" / "lora_sigs.npy",
    512: ROOT / "deepseek_N512_topk"               / "lora_sigs.npy",
}
OUT = Path("/data/android/yqy/work/lora_moe/paper/figures/expert_affinity_pool_size")


def sort_by_depth(f):
    """f: [L, N]; return [N, L] sorted rows + normalised per row."""
    fL = f.T  # [N, L]
    L = fL.shape[1]
    layer_idx = np.arange(L)
    pref = (fL * layer_idx).sum(axis=1) / (fL.sum(axis=1) + 1e-12)
    order = np.argsort(pref)
    sorted_ = fL[order]
    norm = sorted_ / (sorted_.max(axis=1, keepdims=True) + 1e-12)
    return norm


plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10})

# 3 panels side by side, EQUAL width, TALLER for portrait aspect
fig, axes = plt.subplots(
    1, 3, figsize=(9.0, 6.5),
)

import matplotlib.colors as mcolors

# Discrete colormap: low values nearly white, only top values get deep color
bin_edges = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]
bin_colors = [
    "#ffffff",  # 0.00-0.50  pure white
    "#eef3f8",  # 0.50-0.70  nearly white
    "#cee0f2",  # 0.70-0.85  light blue
    "#6baed6",  # 0.85-0.95  blue
    "#542788",  # 0.95-1.00  deep purple
]
custom_cmap = mcolors.ListedColormap(bin_colors)
custom_norm = mcolors.BoundaryNorm(bin_edges, custom_cmap.N)

for ax, N in zip(axes, [128, 256, 512]):
    f = np.load(SIGS[N])
    M = sort_by_depth(f)
    im = ax.imshow(
        M, aspect="auto", cmap=custom_cmap, norm=custom_norm,
        interpolation="nearest",
    )
    ax.set_title(f"$N_L = {N}$", fontsize=11)
    ax.set_xlabel("Layer index $\\ell$")
    if N == 128:
        ax.set_ylabel("Expert (sorted by preferred depth)")

# single shared colorbar on the right
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.8)
cbar.set_label("Activation (per-row normalized)", rotation=270, labelpad=12)

plt.subplots_adjust(left=0.06, right=0.92, wspace=0.18)
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", bbox_inches="tight", dpi=130)
print("Saved:", str(OUT) + ".pdf")
print("Saved:", str(OUT) + ".png")
