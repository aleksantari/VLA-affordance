"""
Script 12: Axis 2 Three-Way Comparison

Loads per-sample metrics from all systems probed in Axis 2 and produces:
  - Combined summary table (Flux, Cosmos Predict2 V2W, Cosmos Policy)
  - Statistical tests for each hypothesis (H2a, H2b, H2c)
  - Per-affordance comparison plots
  - Stratified analysis: manipulation verbs vs other verbs

Inputs (from scripts 10 and 10b):
  results/tables/axis2_per_sample.csv                   (Flux, default)
  results/tables/axis2_<system>_per_sample.csv          (Cosmos systems)

Output:
  results/tables/axis2_three_way_comparison.csv
  results/tables/axis2_hypothesis_tests.json
  results/figures/axis2/three_way_*.png

Usage:
    python scripts/12_three_way_comparison.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MANIPULATION_VERBS = {
    "hold", "pick_up", "pour", "push", "lift", "carry",
    "drag", "pack", "stir", "kick", "throw",
}

# Map system tag → display label
SYSTEM_LABELS = {
    "flux": "Flux (T2I baseline)",
    "flux_dev": "Flux-dev",
    "flux_schnell": "Flux-schnell",
    "cosmos_predict2_t2i": "Cosmos-Predict2 (T2I)",
    "cosmos_predict2_v2w": "Cosmos-Predict2 (V2W, base)",
    "cosmos_policy": "Cosmos-Policy (manipulation FT)",
}


def load_per_sample(path: Path, default_system: str) -> list:
    """Load per-sample CSV. The 'system' column is added if missing."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["kld"] = float(row["kld"])
            row["sim"] = float(row["sim"])
            row["nss"] = float(row["nss"])
            row.setdefault("system", default_system)
            rows.append(row)
    return rows


def aggregate(rows: list) -> dict:
    """Compute mean/std/CI for KLD/SIM/NSS across rows."""
    if not rows:
        return {}
    klds = np.array([r["kld"] for r in rows])
    sims = np.array([r["sim"] for r in rows])
    nsss = np.array([r["nss"] for r in rows])

    def _ci(x):
        # 95% bootstrap CI on the mean
        if len(x) < 2:
            return float(x.mean()) if len(x) else 0.0, float(x.mean()) if len(x) else 0.0
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(1000)])
        return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    return {
        "n": len(rows),
        "kld_mean": float(klds.mean()), "kld_std": float(klds.std()),
        "kld_ci_lo": _ci(klds)[0], "kld_ci_hi": _ci(klds)[1],
        "sim_mean": float(sims.mean()), "sim_std": float(sims.std()),
        "sim_ci_lo": _ci(sims)[0], "sim_ci_hi": _ci(sims)[1],
        "nss_mean": float(nsss.mean()), "nss_std": float(nsss.std()),
        "nss_ci_lo": _ci(nsss)[0], "nss_ci_hi": _ci(nsss)[1],
    }


def wilcoxon_signed_rank(x, y, alternative="greater"):
    """Wilcoxon signed-rank test (paired). Returns (statistic, p)."""
    try:
        from scipy.stats import wilcoxon
        diff = np.asarray(x) - np.asarray(y)
        diff = diff[diff != 0]
        if len(diff) == 0:
            return 0.0, 1.0
        stat, p = wilcoxon(diff, alternative=alternative)
        return float(stat), float(p)
    except ImportError:
        # Fall back to a simple sign test if scipy is unavailable
        diff = np.asarray(x) - np.asarray(y)
        n_pos = (diff > 0).sum()
        n_neg = (diff < 0).sum()
        n = n_pos + n_neg
        if n == 0:
            return 0.0, 1.0
        # Approximate normal under H0 (equal sign probability)
        z = (n_pos - n / 2) / np.sqrt(n / 4)
        p = 0.5 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) * 2
        return float(n_pos), float(min(1.0, p))


def evaluate_hypotheses(by_system: dict) -> dict:
    """Evaluate H2a, H2b, H2c using locked predictions."""
    results = {}

    # H2a — Flux replicates Zhang et al.
    flux_key = (
        "flux_dev" if "flux_dev" in by_system
        else ("flux" if "flux" in by_system else "flux_schnell")
    )
    if flux_key in by_system:
        agg = aggregate(by_system[flux_key])
        results["H2a"] = {
            "system": flux_key,
            "kld_mean": agg.get("kld_mean"),
            "sim_mean": agg.get("sim_mean"),
            "nss_mean": agg.get("nss_mean"),
            "thresholds": {"kld_le": 1.7, "sim_ge": 0.30, "nss_ge": 1.0},
            "kld_pass": agg.get("kld_mean", float("inf")) <= 1.7,
            "sim_pass": agg.get("sim_mean", -1) >= 0.30,
            "nss_pass": agg.get("nss_mean", -1) >= 1.0,
        }
        results["H2a"]["overall_pass"] = (
            results["H2a"]["kld_pass"]
            and results["H2a"]["sim_pass"]
            and results["H2a"]["nss_pass"]
        )

    # H2b — Cosmos Predict2 (V2W) shows binding (NSS > 0.5)
    if "cosmos_predict2_v2w" in by_system:
        rows = by_system["cosmos_predict2_v2w"]
        nsss = np.array([r["nss"] for r in rows])
        # one-sided test: NSS > 0 (well-bound predictions are positive)
        # We use the data themselves to test: median NSS > 0.5
        from numpy import median
        med = float(median(nsss))
        # Bootstrap p that median > 0.5
        rng = np.random.default_rng(0)
        boot = np.array([
            np.median(rng.choice(nsss, size=len(nsss), replace=True))
            for _ in range(1000)
        ])
        p_lt = float((boot <= 0.5).mean())
        results["H2b"] = {
            "system": "cosmos_predict2_v2w",
            "median_nss": med,
            "threshold_nss": 0.5,
            "bootstrap_p(median<=0.5)": p_lt,
            "pass": med > 0.5 and p_lt < 0.05,
            "n": len(rows),
        }

    # H2c — Cosmos Policy > Cosmos Predict2 (paired Wilcoxon)
    if "cosmos_predict2_v2w" in by_system and "cosmos_policy" in by_system:
        # Pair samples by sample_id (must come from the same AGD20K samples)
        pred = {r["sample_id"]: r for r in by_system["cosmos_predict2_v2w"]}
        policy = {r["sample_id"]: r for r in by_system["cosmos_policy"]}
        common = sorted(set(pred.keys()) & set(policy.keys()))

        if common:
            nss_pred = np.array([pred[k]["nss"] for k in common])
            nss_policy = np.array([policy[k]["nss"] for k in common])
            sim_pred = np.array([pred[k]["sim"] for k in common])
            sim_policy = np.array([policy[k]["sim"] for k in common])
            kld_pred = np.array([pred[k]["kld"] for k in common])
            kld_policy = np.array([policy[k]["kld"] for k in common])

            stat_nss, p_nss = wilcoxon_signed_rank(nss_policy, nss_pred, alternative="greater")
            stat_sim, p_sim = wilcoxon_signed_rank(sim_policy, sim_pred, alternative="greater")
            stat_kld, p_kld = wilcoxon_signed_rank(kld_pred, kld_policy, alternative="greater")  # lower KLD better

            delta_nss = float(np.median(nss_policy - nss_pred))
            rel_delta_nss = (
                float((nss_policy.mean() - nss_pred.mean()) / max(abs(nss_pred.mean()), 1e-6))
            )

            # Stratified by manipulation verbs
            man_mask = np.array([
                pred[k]["affordance"] in MANIPULATION_VERBS for k in common
            ])
            stratified = {}
            if man_mask.any():
                stat_man, p_man = wilcoxon_signed_rank(
                    nss_policy[man_mask], nss_pred[man_mask], alternative="greater"
                )
                stratified["manipulation"] = {
                    "n": int(man_mask.sum()),
                    "median_delta_nss": float(np.median(nss_policy[man_mask] - nss_pred[man_mask])),
                    "wilcoxon_stat": stat_man,
                    "wilcoxon_p": p_man,
                }
            if (~man_mask).any():
                stat_oth, p_oth = wilcoxon_signed_rank(
                    nss_policy[~man_mask], nss_pred[~man_mask], alternative="greater"
                )
                stratified["other"] = {
                    "n": int((~man_mask).sum()),
                    "median_delta_nss": float(np.median(nss_policy[~man_mask] - nss_pred[~man_mask])),
                    "wilcoxon_stat": stat_oth,
                    "wilcoxon_p": p_oth,
                }

            results["H2c"] = {
                "n_paired": len(common),
                "median_delta_nss": delta_nss,
                "relative_delta_nss": rel_delta_nss,
                "wilcoxon_nss": {"stat": stat_nss, "p": p_nss},
                "wilcoxon_sim": {"stat": stat_sim, "p": p_sim},
                "wilcoxon_kld": {"stat": stat_kld, "p": p_kld},
                "pass_nss": p_nss < 0.05 and delta_nss > 0,
                "pass_relative_10pct": rel_delta_nss > 0.10,
                "stratified": stratified,
            }

    return results


def plot_three_way_bar(by_system: dict, save_path: Path):
    """Bar chart with 95% bootstrap CIs for each system across 3 metrics."""
    if not by_system:
        return

    systems = list(by_system.keys())
    aggs = {s: aggregate(by_system[s]) for s in systems}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    metrics = [
        ("kld", "KLD ↓ (lower better)", "Reds"),
        ("sim", "SIM ↑ (higher better)", "Greens"),
        ("nss", "NSS ↑ (higher better)", "Blues"),
    ]

    for ax, (metric, title, cmap) in zip(axs, metrics):
        means = [aggs[s].get(f"{metric}_mean", 0) for s in systems]
        ci_lo = [aggs[s].get(f"{metric}_ci_lo", 0) for s in systems]
        ci_hi = [aggs[s].get(f"{metric}_ci_hi", 0) for s in systems]
        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for m, hi in zip(means, ci_hi)]

        labels = [SYSTEM_LABELS.get(s, s) for s in systems]
        x = np.arange(len(systems))
        ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5,
               color=plt.get_cmap(cmap)(np.linspace(0.4, 0.8, len(systems))))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_per_affordance_comparison(by_system: dict, metric: str, save_path: Path):
    """Per-affordance side-by-side bars across systems for one metric."""
    if not by_system:
        return

    systems = list(by_system.keys())
    affordances = sorted(set(
        r["affordance"] for rows in by_system.values() for r in rows
    ))

    fig, ax = plt.subplots(figsize=(max(12, len(affordances) * 0.5), 6))

    width = 0.8 / max(len(systems), 1)
    x = np.arange(len(affordances))

    for j, sys in enumerate(systems):
        vals_by_aff = defaultdict(list)
        for r in by_system[sys]:
            vals_by_aff[r["affordance"]].append(r[metric])
        means = [np.mean(vals_by_aff[a]) if vals_by_aff[a] else 0 for a in affordances]
        ax.bar(x + j * width, means, width=width,
               label=SYSTEM_LABELS.get(sys, sys))

    ax.set_xticks(x + (len(systems) - 1) * width / 2)
    ax.set_xticklabels(affordances, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel({"kld": "KLD ↓", "sim": "SIM ↑", "nss": "NSS ↑"}[metric])
    ax.set_title(f"Per-affordance {metric.upper()} comparison")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    tables_dir = Path(args.results_dir) / "tables"
    figures_dir = Path(args.results_dir) / "figures" / "axis2"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Discover all per-sample CSVs
    by_system: dict = {}

    # Flux (script 10 outputs default name)
    flux_csv = tables_dir / "axis2_per_sample.csv"
    rows = load_per_sample(flux_csv, default_system="flux")
    if rows:
        by_system["flux"] = rows

    for sys_id in ("cosmos_predict2_t2i", "cosmos_predict2_v2w", "cosmos_policy"):
        path = tables_dir / f"axis2_{sys_id}_per_sample.csv"
        rows = load_per_sample(path, default_system=sys_id)
        if rows:
            by_system[sys_id] = rows

    if not by_system:
        print("ERROR: No per-sample CSVs found in", tables_dir)
        print("Run script 10 (Flux) and 10b (Cosmos systems) first.")
        sys.exit(1)

    print("=" * 60)
    print("AXIS 2 — Three-Way Comparison")
    print("=" * 60)
    for s, rows in by_system.items():
        print(f"  {s}: {len(rows)} samples")
    print()

    # Combined per-system summary
    summary_path = tables_dir / "axis2_three_way_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "system", "n",
            "kld_mean", "kld_std", "kld_ci_lo", "kld_ci_hi",
            "sim_mean", "sim_std", "sim_ci_lo", "sim_ci_hi",
            "nss_mean", "nss_std", "nss_ci_lo", "nss_ci_hi",
        ])
        for s, rows in by_system.items():
            agg = aggregate(rows)
            w.writerow([
                s, agg["n"],
                f"{agg['kld_mean']:.4f}", f"{agg['kld_std']:.4f}",
                f"{agg['kld_ci_lo']:.4f}", f"{agg['kld_ci_hi']:.4f}",
                f"{agg['sim_mean']:.4f}", f"{agg['sim_std']:.4f}",
                f"{agg['sim_ci_lo']:.4f}", f"{agg['sim_ci_hi']:.4f}",
                f"{agg['nss_mean']:.4f}", f"{agg['nss_std']:.4f}",
                f"{agg['nss_ci_lo']:.4f}", f"{agg['nss_ci_hi']:.4f}",
            ])
    print(f"Wrote {summary_path}")

    # Hypothesis tests
    h_results = evaluate_hypotheses(by_system)
    h_path = tables_dir / "axis2_hypothesis_tests.json"
    with open(h_path, "w") as f:
        json.dump(h_results, f, indent=2)
    print(f"Wrote {h_path}")

    # Print hypothesis summary
    print("\n── Hypothesis Tests ──")
    for hk, hv in h_results.items():
        print(f"\n{hk}:")
        for k, v in hv.items():
            print(f"  {k}: {v}")

    # Figures
    print("\nGenerating figures...")
    plot_three_way_bar(by_system, figures_dir / "three_way_overall.png")
    for metric in ("kld", "sim", "nss"):
        plot_per_affordance_comparison(
            by_system, metric, figures_dir / f"three_way_per_affordance_{metric}.png"
        )

    print()
    print("=" * 60)
    print("THREE-WAY COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Outputs in {tables_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
