"""Re-render cross-layer CKA heatmap with a cool palette
(no need to re-run the GPU probe; just reads the saved matrix).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/data/android/yqy/work/lora_moe/data/cka_cross_layer.json"
OUT = "/data/android/yqy/work/lora_moe/paper/figures/cka_cross_layer"

cka = np.array(json.load(open(SRC))["cka"])
L = cka.shape[0]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10})
fig, ax = plt.subplots(figsize=(5.5, 4.6))

# Simple cool monochrome: white (low) → deep blue (high). Calm, not vibrant.
import matplotlib.colors as mcolors

# UV-like gradient: pale lavender → blue → violet → deep purple
# No green, no red. Pure cool/UV spectrum.
cmap = mcolors.LinearSegmentedColormap.from_list("uv_spectrum", [
    (0.00, "#e8eef5"),  # very pale lavender-blue (near white)
    (0.25, "#a8bce0"),  # light blue
    (0.50, "#7370c2"),  # blue-violet
    (0.75, "#6845a8"),  # violet
    (1.00, "#4a2880"),  # deep UV purple
])

im = ax.imshow(cka, cmap=cmap, vmin=0, vmax=1, aspect="equal",
               interpolation="nearest")
ax.set_xlabel("Layer index $\\ell_j$")
ax.set_ylabel("Layer index $\\ell_i$")
ax.set_xticks([0, L//4, L//2, 3*L//4, L-1])
ax.set_yticks([0, L//4, L//2, 3*L//4, L-1])

cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("Linear CKA", fontsize=9)

plt.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", bbox_inches="tight", dpi=130)
print("Saved:", OUT + ".pdf")
print("Saved:", OUT + ".png")
