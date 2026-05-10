"""
Quick analysis pass on a partial Flux per-sample CSV.

Designed to run on partial CSVs that get auto-pushed from Colab during a
pilot (every --commit_every N samples). Prints a one-screen summary and
flags any sanity issues (NaNs, zero variance, suspicious metric values).

Usage:
    python scripts/dev_quick_analysis.py
    python scripts/dev_quick_analysis.py --csv results/tables/axis2_per_sample.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/tables/axis2_per_sample.csv")
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"ERROR: {path} not found. Has Colab pushed a checkpoint yet?")
        sys.exit(1)

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["kld"] = float(r["kld"])
                r["sim"] = float(r["sim"])
                r["nss"] = float(r["nss"])
            except (KeyError, ValueError):
                continue
            rows.append(r)

    if not rows:
        print(f"ERROR: {path} has no data rows.")
        sys.exit(1)

    print(f"\n{'=' * 60}\nQuick analysis: {path}\n{'=' * 60}")
    print(f"Total samples: {len(rows)}")

    klds = np.array([r["kld"] for r in rows])
    sims = np.array([r["sim"] for r in rows])
    nsss = np.array([r["nss"] for r in rows])

    def _summary(name, arr, target_op):
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            print(f"  {name}: ALL non-finite ({n_nan} NaN, {n_inf} Inf)")
            return
        print(
            f"  {name}: mean={valid.mean():.4f}  std={valid.std():.4f}  "
            f"median={np.median(valid):.4f}  min={valid.min():.4f}  max={valid.max():.4f}"
            f"  (n_nan={n_nan}, n_inf={n_inf})"
        )

    print("\nOverall metrics:")
    _summary("KLD↓", klds, min)
    _summary("SIM↑", sims, max)
    _summary("NSS↑", nsss, max)

    # By affordance
    by_aff = defaultdict(list)
    for r in rows:
        by_aff[r["affordance"]].append(r)
    print(f"\nPer-affordance (n={len(by_aff)}):")
    print(f"{'affordance':<18} {'n':>4} {'KLD↓':>8} {'SIM↑':>8} {'NSS↑':>8}")
    print("-" * 50)
    for aff in sorted(by_aff):
        arr = by_aff[aff]
        if not arr:
            continue
        k = np.mean([r["kld"] for r in arr])
        s = np.mean([r["sim"] for r in arr])
        n = np.mean([r["nss"] for r in arr])
        print(f"{aff:<18} {len(arr):>4} {k:>8.3f} {s:>8.3f} {n:>8.3f}")

    # H2a verdict on the partial data
    print("\n── H2a (locked thresholds: KLD≤1.7, SIM≥0.30, NSS≥1.0) ──")
    valid_klds = klds[np.isfinite(klds)]
    valid_sims = sims[np.isfinite(sims)]
    valid_nsss = nsss[np.isfinite(nsss)]
    if len(valid_klds) and len(valid_sims) and len(valid_nsss):
        kld_pass = valid_klds.mean() <= 1.7
        sim_pass = valid_sims.mean() >= 0.30
        nss_pass = valid_nsss.mean() >= 1.0
        print(f"  KLD mean: {valid_klds.mean():.3f} → {'PASS' if kld_pass else 'FAIL'} (≤ 1.7)")
        print(f"  SIM mean: {valid_sims.mean():.3f} → {'PASS' if sim_pass else 'FAIL'} (≥ 0.30)")
        print(f"  NSS mean: {valid_nsss.mean():.3f} → {'PASS' if nss_pass else 'FAIL'} (≥ 1.0)")
        if kld_pass and sim_pass and nss_pass:
            print("  → H2a CONFIRMED on partial data")
        else:
            print("  → H2a not yet confirmed (or partial data still ramping)")
    print()


if __name__ == "__main__":
    main()
