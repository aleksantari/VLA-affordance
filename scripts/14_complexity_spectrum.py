"""
Script 14: Binary peak-in-GT-region analysis on cached attention maps.

Reports a complementary binary metric to KLD/SIM/NSS on AGD20K: for each
(image, verb) pair, does the verb-attention peak land inside the GT
functional region? Gives a coarser but interpretable view that bypasses
distributional-metric sensitivities to map smoothness, scale, and noise.

  - Large KLD/SIM/NSS gap + small binary gap → metrics may be picking up
    map-smoothness differences, not real perceptual differences.
  - Both gaps agree → real deficit.

Inputs:
  results/cached_features/<system>_attention/*.npy   (per-sample attention maps)
  data/agd20k                                         (for GT heatmaps)

Outputs:
  results/tables/axis2_complexity_spectrum.csv
  results/figures/axis2/complexity_spectrum.png

Usage:
    python scripts/14_complexity_spectrum.py
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SYSTEMS = ["flux_attention", "cosmos_predict2_v2w_attention", "cosmos_policy_attention"]
SYSTEM_LABELS = {
    "flux_attention": "Flux",
    "cosmos_predict2_v2w_attention": "Cosmos-V2W",
    "cosmos_policy_attention": "Cosmos-Policy",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--data_dir", type=str, default="./data/agd20k")
    args = parser.parse_args()

    from data.agd20k_dataset import AGD20KDataset
    from interaction.verb_spatial_binding import peak_in_gt_region

    print("=" * 60)
    print("AXIS 2 — Binary peak-in-GT-region analysis")
    print("=" * 60)

    cache_root = Path(args.results_dir) / "cached_features"
    ds = AGD20KDataset(data_dir=args.data_dir, image_size=512)

    # Index dataset samples by sample_id (e.g. "cut_3")
    sample_by_id = {}
    by_aff = defaultdict(list)
    for idx, s in enumerate(ds.samples):
        aff = s["affordance"]
        by_aff[aff].append(idx)
    for aff, idxs in by_aff.items():
        for i, ds_idx in enumerate(idxs):
            sample_by_id[f"{aff}_{i}"] = ds_idx

    rows = []
    for system in SYSTEMS:
        cdir = cache_root / system
        if not cdir.exists():
            print(f"  ⚠ {system}: no cached attention dir, skip")
            continue
        maps = list(cdir.glob("*.npy"))
        if not maps:
            print(f"  ⚠ {system}: no .npy files, skip")
            continue
        print(f"  {system}: {len(maps)} cached maps")

        per_aff_hits = defaultdict(list)
        for npy in maps:
            sid = npy.stem
            if sid not in sample_by_id:
                continue
            pred = np.load(str(npy))
            sample = ds[sample_by_id[sid]]
            if sample["gt_heatmap"] is None:
                continue
            hit = peak_in_gt_region(pred, sample["gt_heatmap"])
            per_aff_hits[sample["affordance"]].append(hit)

        for aff, hits in per_aff_hits.items():
            if not hits:
                continue
            rate = float(np.mean(hits))
            rows.append({
                "system": system, "affordance": aff,
                "n": len(hits), "peak_in_gt_rate": rate,
            })

    if not rows:
        print("ERROR: no overlap between cached attention and AGD20K samples")
        sys.exit(1)

    # Save table
    out_csv = Path(args.results_dir) / "tables" / "axis2_complexity_spectrum.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "affordance", "n", "peak_in_gt_rate"])
        w.writeheader()
        for r in rows:
            r["peak_in_gt_rate"] = f"{r['peak_in_gt_rate']:.4f}"
            w.writerow(r)
    print(f"\nWrote {out_csv}")

    # Plot
    affordances = sorted({r["affordance"] for r in rows})
    systems_seen = sorted({r["system"] for r in rows})

    fig, ax = plt.subplots(figsize=(max(10, len(affordances) * 0.45), 5))
    bar_w = 0.8 / max(len(systems_seen), 1)
    x = np.arange(len(affordances))
    for i, system in enumerate(systems_seen):
        rates_by_aff = {r["affordance"]: float(r["peak_in_gt_rate"])
                        for r in rows if r["system"] == system}
        vals = [rates_by_aff.get(a, 0) for a in affordances]
        ax.bar(x + i * bar_w, vals, bar_w, label=SYSTEM_LABELS.get(system, system))
    ax.set_xticks(x + (len(systems_seen) - 1) * bar_w / 2)
    ax.set_xticklabels(affordances, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Peak-in-GT hit rate (binary)")
    ax.set_title("Axis 2 complexity spectrum: binary peak-in-GT per affordance")
    ax.axhline(0.2, ls="--", c="grey", lw=0.8, label="random baseline (~20%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_fig = Path(args.results_dir) / "figures" / "axis2" / "complexity_spectrum.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")

    # Overall summary per system
    print("\n── Overall binary peak-in-GT rates ──")
    for system in systems_seen:
        sys_rows = [r for r in rows if r["system"] == system]
        if not sys_rows:
            continue
        ns = [r["n"] for r in sys_rows]
        rates = [float(r["peak_in_gt_rate"]) for r in sys_rows]
        weighted = np.average(rates, weights=ns) if ns else 0
        print(f"  {SYSTEM_LABELS.get(system, system):>20}: "
              f"{weighted:.3f} (weighted across {len(sys_rows)} affordances, "
              f"n={sum(ns)} samples)")
    print()


if __name__ == "__main__":
    main()
