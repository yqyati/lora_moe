"""Hierarchical clustering + math-vs-code dual panel figure.

Inputs:  lora_sigs.npy from two probes (math + code) on the SAME trained model.
Outputs: PDF figure with 2 panels (math | code), experts sorted by cluster + depth,
         left color band = cluster membership.

Goal: Show that
  (a) experts cluster into K functional groups (data-driven, hierarchical),
  (b) the same cluster structure is stable across input domains.
"""
import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def normalize_rows(x):
    s = x.sum(axis=1, keepdims=True) + 1e-12
    return x / s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--math_probe", required=True, help="probe dir with lora_sigs.npy (math)")
    p.add_argument("--code_probe", required=True, help="probe dir with lora_sigs.npy (code)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--K", type=int, default=5, help="number of clusters")
    p.add_argument(
        "--linkage", default="average", choices=["average", "ward", "complete", "single"]
    )
    p.add_argument(
        "--metric", default="cosine", choices=["cosine", "correlation", "euclidean"]
    )
    p.add_argument("--filter_min_usage", type=float, default=0.0,
                   help="exclude experts with total usage < this (math probe). 0 = keep all")
    p.add_argument("--model_label", default="DeepSeek-V2-Lite N=512")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Load probe data ===
    math_sigs = np.load(Path(args.math_probe) / "lora_sigs.npy")  # [L, N]
    code_sigs = np.load(Path(args.code_probe) / "lora_sigs.npy")  # [L, N]
    L, N = math_sigs.shape
    assert code_sigs.shape == (L, N), f"shape mismatch: {math_sigs.shape} vs {code_sigs.shape}"
    print(f"L={L} layers, N={N} experts")

    math_expert = math_sigs.T  # [N, L]
    code_expert = code_sigs.T

    # Optionally filter dead experts based on math usage
    math_total = math_expert.sum(axis=1)
    if args.filter_min_usage > 0:
        keep = math_total >= args.filter_min_usage
        print(f"  filter usage >= {args.filter_min_usage}: keep {keep.sum()} / {N} experts")
        math_expert = math_expert[keep]
        code_expert = code_expert[keep]
        N = math_expert.shape[0]

    # === Hierarchical clustering on math (normalized for shape) ===
    math_normed = normalize_rows(math_expert)
    dist = pdist(math_normed, metric=args.metric)
    Z = linkage(dist, method=args.linkage)
    cluster_ids = fcluster(Z, t=args.K, criterion="maxclust")  # [N]

    # cluster sizes
    sizes = {c: int((cluster_ids == c).sum()) for c in range(1, args.K + 1)}
    print(f"  Cluster sizes: {sizes}")

    # === Sort: first by cluster centroid depth, then by individual depth within cluster ===
    layer_idx = np.arange(L)
    preferred_depth = (math_normed * layer_idx).sum(axis=1)

    # cluster centroid depth (for ordering clusters from early -> late)
    cluster_depth = {}
    for c in range(1, args.K + 1):
        mask = (cluster_ids == c)
        cluster_depth[c] = float(preferred_depth[mask].mean())
    # reassign cluster IDs so that 1 = earliest, K = latest
    sorted_old = sorted(cluster_depth.items(), key=lambda kv: kv[1])
    remap = {old: new for new, (old, _) in enumerate(sorted_old, start=1)}
    cluster_ids = np.array([remap[c] for c in cluster_ids])

    sort_order = np.lexsort((preferred_depth, cluster_ids))
    math_sorted = math_expert[sort_order]
    code_sorted = code_expert[sort_order]
    clusters_sorted = cluster_ids[sort_order]

    # === Plot ===
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 7),
        gridspec_kw={"width_ratios": [0.05, 1, 1], "wspace": 0.18},
    )
    ax_band, ax_math, ax_code = axes

    # Shared colormap and value range
    vmax = max(math_sorted.max(), code_sorted.max())

    # ---- color band: cluster membership ----
    cluster_palette = plt.cm.Set2(np.linspace(0, 1, max(args.K, 3)))
    cluster_cmap = ListedColormap(cluster_palette[:args.K])
    band = clusters_sorted.reshape(-1, 1)
    ax_band.imshow(band, aspect="auto", cmap=cluster_cmap, vmin=1, vmax=args.K,
                   interpolation="nearest")
    ax_band.set_xticks([])
    ax_band.set_ylabel("Expert (sorted: cluster, then preferred depth)")
    ax_band.set_title("Cluster", fontsize=10)
    # add cluster boundaries
    boundaries = []
    for i in range(1, len(clusters_sorted)):
        if clusters_sorted[i] != clusters_sorted[i - 1]:
            boundaries.append(i - 0.5)

    # ---- math heatmap ----
    im_m = ax_math.imshow(math_sorted, aspect="auto", cmap="viridis",
                          vmin=0, vmax=vmax, interpolation="nearest")
    ax_math.set_xlabel("Layer index")
    ax_math.set_title(f"Math (GSM8K) — {args.model_label}")
    ax_math.set_yticks([])
    for b in boundaries:
        ax_math.axhline(b, color="white", lw=0.6, alpha=0.6)

    # ---- code heatmap ----
    im_c = ax_code.imshow(code_sorted, aspect="auto", cmap="viridis",
                          vmin=0, vmax=vmax, interpolation="nearest")
    ax_code.set_xlabel("Layer index")
    ax_code.set_title("Code (MBPP) — same sort order")
    ax_code.set_yticks([])
    for b in boundaries:
        ax_code.axhline(b, color="white", lw=0.6, alpha=0.6)

    # Shared colorbar on right
    cbar = fig.colorbar(im_c, ax=ax_code, fraction=0.04, pad=0.02)
    cbar.set_label("Avg # top-k selections per token", rotation=270, labelpad=14)

    # Cluster legend below color band
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=cluster_palette[i],
              label=f"C{i + 1} (n={sizes.get(sorted_old[i][0], 0) if i < len(sorted_old) else 0}, "
                    f"avg depth={sorted_old[i][1]:.1f})")
        for i in range(args.K)
    ]
    # Actually use the remapped sizes
    sizes_remapped = {remap[old]: sz for old, sz in sizes.items()}
    legend_elems = []
    for new_id in range(1, args.K + 1):
        old_id = [old for old, new in remap.items() if new == new_id][0]
        legend_elems.append(Patch(
            facecolor=cluster_palette[new_id - 1],
            label=f"C{new_id} (n={sizes[old_id]}, depth={cluster_depth[old_id]:.1f})",
        ))
    ax_band.legend(handles=legend_elems, loc="upper left",
                   bbox_to_anchor=(-0.5, -0.04), fontsize=8, frameon=False, ncol=1)

    plt.suptitle(
        "Emergent expert clustering with cross-domain stability",
        y=0.995, fontsize=13,
    )

    pdf_path = out / "expert_clusters_math_vs_code.pdf"
    png_path = out / "expert_clusters_math_vs_code.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}\n       {png_path}")

    # === Compute cross-domain cluster stability stats ===
    # For each cluster, measure how much its experts behave the same on math vs code
    cluster_stats = {}
    for new_id in range(1, args.K + 1):
        mask = cluster_ids == new_id
        m_sub = math_expert[mask]
        c_sub = code_expert[mask]
        # mean cosine similarity of expert profiles between math and code
        cosines = []
        for i in range(m_sub.shape[0]):
            nm = np.linalg.norm(m_sub[i]) + 1e-12
            nc = np.linalg.norm(c_sub[i]) + 1e-12
            cosines.append(float((m_sub[i] @ c_sub[i]) / (nm * nc)))
        cluster_stats[f"C{new_id}"] = {
            "size": int(mask.sum()),
            "avg_depth": float(preferred_depth[mask].mean()),
            "math_vs_code_cosine_mean": float(np.mean(cosines)),
            "math_vs_code_cosine_std": float(np.std(cosines)),
        }

    summary = {
        "L": int(L),
        "N_after_filter": int(N),
        "K": int(args.K),
        "linkage": args.linkage,
        "metric": args.metric,
        "clusters": cluster_stats,
    }
    with open(out / "cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
