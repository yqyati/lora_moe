"""Per-layer F-statistic line plot (AnchorLoRA vs MoELoRA), single-column."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_fstats(path):
    d = json.load(open(path))
    ours_pl = d["ours"]["per_layer"]
    base_pl = d["baseline"]["per_layer"]
    layers = sorted([int(k) for k in ours_pl.keys()])
    ours_f = np.array([ours_pl[str(l)]["f_statistic"] for l in layers])
    base_f = np.array([base_pl[str(l)]["f_statistic"] for l in layers])
    ours_avg = d["ours"]["global"]["avg_f_statistic"]
    base_avg = d["baseline"]["global"]["avg_f_statistic"]
    return np.array(layers), ours_f, base_f, ours_avg, base_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment_json", required=True, help="alignment_*.json produced by analyze_routing.py")
    ap.add_argument("--ours_label", default="AnchorLoRA")
    ap.add_argument("--baseline_label", default="MoELoRA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--y_scale", default="linear", choices=["linear", "sqrt", "cbrt", "symlog"],
                    help="y-axis scale: linear / sqrt / cbrt (more aggressive compress) / symlog")
    args = ap.parse_args()

    layers, ours_f, base_f, ours_avg, base_avg = load_fstats(args.alignment_json)

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(layers, ours_f, marker="o", markersize=4, linewidth=1.7, color="#c0392b",
            label=f"{args.ours_label} (mean ${ours_avg:.1f}$)")
    ax.plot(layers, base_f, marker="s", markersize=4, linewidth=1.7, color="#7f8c8d",
            label=f"{args.baseline_label} (mean ${base_avg:.1f}$)")
    ax.axhline(ours_avg, linestyle="--", color="#c0392b", linewidth=0.8, alpha=0.7)
    ax.axhline(base_avg, linestyle="--", color="#7f8c8d", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Layer index $\\ell$")
    ax.set_ylabel("Per-layer F-statistic")
    ax.set_xlim(-0.5, len(layers) - 0.5)
    y_min = max(0, min(ours_f.min(), base_f.min()) * 0.7) if args.y_scale in ("sqrt", "cbrt", "symlog") else 0
    ax.set_ylim(y_min, max(ours_f.max(), base_f.max()) * 1.15)
    if args.y_scale == "sqrt":
        ax.set_yscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)),
                                              lambda x: x ** 2))
        import matplotlib.ticker as mt
        ax.set_yticks([15, 25, 50, 100, 200, 300])
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%d"))
    elif args.y_scale == "cbrt":
        ax.set_yscale("function", functions=(lambda x: np.cbrt(np.clip(x, 0, None)),
                                              lambda x: x ** 3))
        import matplotlib.ticker as mt
        ax.set_yticks([15, 25, 50, 100, 200, 300])
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%d"))
    elif args.y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=100)
        import matplotlib.ticker as mt
        ax.set_yticks([10, 25, 50, 75, 100, 200, 300])
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%d"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    print(f"Saved: {png}")
    print(f"{args.ours_label}: range [{ours_f.min():.1f}, {ours_f.max():.1f}], mean {ours_avg:.2f}")
    print(f"{args.baseline_label}: range [{base_f.min():.1f}, {base_f.max():.1f}], mean {base_avg:.2f}")


if __name__ == "__main__":
    main()
