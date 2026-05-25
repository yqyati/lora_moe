"""DeepSeek matched-budget sweep: accuracy & inference throughput vs N_L.

Output: paper/figures/sweep_acc_vs_speed.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matched-budget sweep on DeepSeek-V2-Lite (~22M trainable)
N_L  = [64, 128, 256, 512]
rank = [70, 34,  17,  8  ]
gsm8k    = [52.24, 53.30, 50.87, 49.51]
math500  = [17.00, 18.40, 16.80, 16.60]
avg      = [34.62, 35.85, 33.84, 33.05]
tps      = [21.16, 19.59, 17.50, 14.41]

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                     "axes.titlesize": 11, "legend.fontsize": 9})
fig, ax1 = plt.subplots(figsize=(5.8, 3.6))

# left y: accuracy (Avg, GSM8K, MATH-500)
color_acc = "#1f6feb"
ax1.plot(N_L, avg, marker="o", markersize=7, linewidth=2.0,
         color=color_acc, label="Avg (GSM8K + MATH-500)")
ax1.plot(N_L, gsm8k, marker="s", markersize=5, linewidth=1.2,
         linestyle=":", color=color_acc, alpha=0.6, label="GSM8K")
ax1.plot(N_L, math500, marker="^", markersize=5, linewidth=1.2,
         linestyle=":", color=color_acc, alpha=0.4, label="MATH-500")
ax1.set_xlabel("Global pool size $N_L$ (log scale)")
ax1.set_ylabel("Math accuracy (%)", color=color_acc)
ax1.tick_params(axis="y", labelcolor=color_acc)
ax1.set_xscale("log", base=2)
ax1.set_xticks(N_L)
ax1.set_xticklabels([str(n) for n in N_L])
ax1.grid(axis="y", linestyle=":", alpha=0.4)

# right y: inference throughput
ax2 = ax1.twinx()
color_tps = "#ff8c00"
ax2.plot(N_L, tps, marker="D", markersize=7, linewidth=2.0,
         color=color_tps, label="tokens / sec")
ax2.set_ylabel("Tokens / sec (single H100, bs=1)", color=color_tps)
ax2.tick_params(axis="y", labelcolor=color_tps)

# annotate the sweet spot at N=128
ax1.axvline(128, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
ax1.annotate(
    "ours main\n($N_L{=}128$)",
    xy=(128, 35.85), xytext=(155, 36.4),
    fontsize=8.5, color="black",
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
)

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower left", frameon=False)

plt.tight_layout()
out = "/data/android/yqy/work/lora_moe/paper/figures/sweep_acc_vs_speed"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight", dpi=130)
print("Saved:", out + ".pdf")
print("Saved:", out + ".png")
