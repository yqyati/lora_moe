"""Per-layer effective expert count line plot (math + code probes, DeepSeek-V2-Lite N=512)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def effective_N(f: np.ndarray) -> np.ndarray:
    f_norm = f / (f.sum(axis=-1, keepdims=True) + 1e-12)
    ent = -(f_norm * np.log(f_norm + 1e-12)).sum(axis=-1)
    return np.exp(ent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math_sigs", default="/data/android/yqy/work/lora_moe/data/cfr_probe/deepseek_N512_topk/lora_sigs.npy")
    ap.add_argument("--code_sigs", default="/data/android/yqy/work/lora_moe/data/cfr_probe/deepseek_N512_topk_code/lora_sigs.npy")
    ap.add_argument("--ref_N", type=float, default=20.0, help="MoELoRA fixed allocation reference")
    ap.add_argument("--out", default="/data/android/yqy/work/lora_moe/paper/figures/layer_allocation_math_vs_code.pdf")
    args = ap.parse_args()

    math = np.load(args.math_sigs)
    code = np.load(args.code_sigs)
    math_eN = effective_N(math)
    code_eN = effective_N(code)
    layers = np.arange(len(math_eN))

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(layers, math_eN, marker="o", markersize=4, linewidth=1.6, color="#1f6feb", label="math probe")
    ax.plot(layers, code_eN, marker="s", markersize=4, linewidth=1.6, color="#ff8c00", label="code probe")
    ax.axhline(args.ref_N, linestyle="--", color="gray", linewidth=1.0, label=f"MoELoRA fixed N={int(args.ref_N)}")

    ax.set_xlabel("Layer index $\\ell$")
    ax.set_ylabel("Effective expert count $\\exp(H(f_\\ell))$")
    ax.set_xlim(-0.5, len(math_eN) - 0.5)
    ax.set_ylim(0, max(math_eN.max(), code_eN.max()) * 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    r = np.corrcoef(math_eN, code_eN)[0, 1]
    print(f"math effN: min={math_eN.min():.2f}, max={math_eN.max():.2f}, mean={math_eN.mean():.2f}")
    print(f"code effN: min={code_eN.min():.2f}, max={code_eN.max():.2f}, mean={code_eN.mean():.2f}")
    print(f"Pearson r(math, code) = {r:.4f} over {len(math_eN)} layers")


if __name__ == "__main__":
    main()
