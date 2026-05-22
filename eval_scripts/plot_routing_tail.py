"""Bucket rarity into head / mid / tail and plot per-method mean loss per bucket."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(fp):
    rarities, losses = [], []
    with open(fp) as f:
        for line in f:
            d = json.loads(line)
            rarities.append(d["rarity"])
            losses.append(d["loss"])
    return np.array(rarities), np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", required=True,
                    help="<name>:<path.jsonl> pairs, e.g. ours:probes/ours.jsonl moelora:probes/moelora.jsonl")
    ap.add_argument("--ref_for_bucketing", default=None,
                    help="which probe to use for rarity quantile cutoffs (default = first one)")
    ap.add_argument("--n_buckets", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.probes:
        name, path = spec.split(":", 1)
        runs[name] = load(path)
        print(f"{name}: n_tokens={len(runs[name][0])}, rarity range [{runs[name][0].min():.1f}, {runs[name][0].max():.1f}]")

    # Bucket cutoffs come from a single reference (default: first probe)
    ref_name = args.ref_for_bucketing or list(runs.keys())[0]
    ref_rarities = runs[ref_name][0]
    quantiles = np.linspace(0, 1, args.n_buckets + 1)[1:-1]
    cutoffs = np.quantile(ref_rarities, quantiles)
    print(f"Using {ref_name} for cutoffs: {cutoffs.tolist()}")

    bucket_labels = ["head", "mid", "tail"] if args.n_buckets == 3 else [f"q{i+1}" for i in range(args.n_buckets)]

    # Compute per-bucket mean loss for each method
    # Caveat: each method's per-token records correspond to ITS OWN forward —
    # rarity is computed from base router which is identical across methods (frozen),
    # but token positions / counts must align. Assumes all methods used the same
    # prompts in the same order.
    n_ref = len(ref_rarities)
    bucket_ids = np.digitize(ref_rarities, cutoffs)  # [0, n_buckets-1] (0 = head)

    table = {}
    for name, (rar, loss) in runs.items():
        # sanity: lengths should match across methods
        if len(loss) != n_ref:
            print(f"WARN: {name} has {len(loss)} tokens vs ref {n_ref}; truncating to min")
            n = min(len(loss), n_ref)
            loss = loss[:n]; bids = bucket_ids[:n]
        else:
            bids = bucket_ids
        per_bucket = []
        for b in range(args.n_buckets):
            mask = bids == b
            per_bucket.append(float(loss[mask].mean()) if mask.any() else float("nan"))
        table[name] = per_bucket

    # Print table
    print(f"\nMean loss by rarity bucket:")
    print(f"{'method':<20}" + "".join(f"{lab:>10}" for lab in bucket_labels))
    for name, vals in table.items():
        print(f"{name:<20}" + "".join(f"{v:>10.4f}" for v in vals))

    # Compute Δ vs first method (e.g., ours)
    ref_method = list(table.keys())[0]
    print(f"\nΔ loss vs {ref_method}:")
    print(f"{'method':<20}" + "".join(f"{lab:>10}" for lab in bucket_labels))
    for name, vals in table.items():
        if name == ref_method:
            continue
        deltas = [v - r for v, r in zip(vals, table[ref_method])]
        print(f"{name:<20}" + "".join(f"{d:>+10.4f}" for d in deltas))

    # Plot
    n_methods = len(table)
    width = 0.8 / n_methods
    x = np.arange(args.n_buckets)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, (name, vals) in enumerate(table.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Routing rarity bucket")
    ax.set_ylabel("Per-token LM loss")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out}\nSaved: {png}")


if __name__ == "__main__":
    main()
