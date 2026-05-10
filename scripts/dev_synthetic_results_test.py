"""
Synthetic-results integration test for Axis 2 analysis pipeline.

Generates fake per-sample CSVs that mimic the Colab pilot output, then runs
script 12 (three-way comparison) end-to-end against them. This catches
import errors, schema mismatches, and obvious bugs in the analysis path
before real Colab results arrive (which take hours to produce).

Not a unit test framework — just a self-contained `python scripts/dev_synthetic_results_test.py`
runner. Lives outside the production pipeline (filename starts with `dev_`).

It does NOT exercise script 13 (qualitative panel) because that needs
.npy attention maps + AGD20K image pairs, which would require fabricating
both. Script 12 is the higher-leverage check — its statistical tests are
the harder code path.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np


# Affordance categories matching AGD20K loader
AFFORDANCES = [
    "beat", "boxing", "brush_with", "carry", "catch", "cut",
    "cut_with", "drag", "drink_with", "eat", "hit", "hold",
    "jump", "kick", "lie_on", "lift", "look_out", "open",
    "pack", "peel", "pick_up", "pour", "push", "ride",
    "sip", "sit_on", "stick", "stir", "swing", "take_photo",
    "talk_on", "text_on", "throw", "type_on", "wash", "write",
]

MANIPULATION_SET = {"hold", "pick_up", "pour", "push", "lift", "carry",
                    "drag", "pack", "stir", "kick", "throw"}


def make_csv(
    path: Path,
    system: str,
    samples_per_aff: int,
    rng: np.random.Generator,
    kld_loc: float,
    kld_scale: float,
    sim_loc: float,
    sim_scale: float,
    nss_loc: float,
    nss_scale: float,
    manipulation_boost: float = 0.0,
):
    """
    Write a synthetic per-sample CSV. Distributions chosen to roughly
    resemble what we expect:
      - flux:      KLD ≈ 1.5, SIM ≈ 0.32, NSS ≈ 1.1 (Zhang et al.)
      - cosmos_v2w: KLD ≈ 1.7, SIM ≈ 0.25, NSS ≈ 0.7 (a guess, weaker than Flux)
      - cosmos_pol: same + manipulation boost on hold/pick_up/pour/...
                    so H2c is detectable in synthetic data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "system", "affordance",
                                          "prompt", "kld", "sim", "nss"])
        w.writeheader()
        for aff in AFFORDANCES:
            for i in range(samples_per_aff):
                kld = max(0.0, rng.normal(kld_loc, kld_scale))
                sim = float(np.clip(rng.normal(sim_loc, sim_scale), 0, 1))
                nss = rng.normal(nss_loc, nss_scale)

                if manipulation_boost and aff in MANIPULATION_SET:
                    nss += manipulation_boost
                    sim += 0.5 * manipulation_boost
                    kld -= 0.3 * manipulation_boost

                w.writerow({
                    "sample_id": f"{aff}_{i}",
                    "system": system,
                    "affordance": aff,
                    "prompt": f"a person {aff.replace('_', ' ')}ing a thing",
                    "kld": f"{kld:.6f}",
                    "sim": f"{sim:.6f}",
                    "nss": f"{nss:.6f}",
                })
    return path


def main():
    repo_root = Path(__file__).parent.parent
    samples_per_aff = 30  # match pilot scale

    # Use a temp dir layout that mirrors results/, then copy into the real
    # results/ for script 12 to find. Restore from backup at the end.
    tables_dir = repo_root / "results" / "tables"
    figures_dir = repo_root / "results" / "figures" / "axis2"

    backup = None
    results_existed_before = (repo_root / "results").exists()
    if results_existed_before:
        backup = Path(tempfile.mkdtemp(prefix="axis2_results_backup_"))
        shutil.copytree(str(repo_root / "results"), str(backup / "results"))
        print(f"Backed up existing results to {backup}/results")
    else:
        print("No pre-existing results/ — will fully remove synthetic artifacts on exit.")

    try:
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)

        flux_csv = tables_dir / "axis2_per_sample.csv"
        v2w_csv = tables_dir / "axis2_cosmos_predict2_v2w_per_sample.csv"
        pol_csv = tables_dir / "axis2_cosmos_policy_per_sample.csv"

        # Flux ≈ Zhang et al. (close to thresholds)
        make_csv(flux_csv, "flux", samples_per_aff, rng,
                 kld_loc=1.50, kld_scale=0.35,
                 sim_loc=0.32, sim_scale=0.10,
                 nss_loc=1.10, nss_scale=0.40)
        # Cosmos V2W base — weaker
        make_csv(v2w_csv, "cosmos_predict2_v2w", samples_per_aff, rng,
                 kld_loc=1.80, kld_scale=0.40,
                 sim_loc=0.22, sim_scale=0.10,
                 nss_loc=0.65, nss_scale=0.45)
        # Cosmos Policy — base + manipulation boost on manip verbs
        make_csv(pol_csv, "cosmos_policy", samples_per_aff, rng,
                 kld_loc=1.65, kld_scale=0.40,
                 sim_loc=0.26, sim_scale=0.10,
                 nss_loc=0.85, nss_scale=0.45,
                 manipulation_boost=0.40)

        print(f"Wrote synthetic CSVs:")
        for p in (flux_csv, v2w_csv, pol_csv):
            n = sum(1 for _ in open(p)) - 1  # minus header
            print(f"  {p}  ({n} rows)")

        # Run script 12
        print("\nRunning scripts/12_three_way_comparison.py ...")
        r = subprocess.run(
            [sys.executable, "scripts/12_three_way_comparison.py"],
            cwd=str(repo_root),
            capture_output=True, text=True,
        )
        if r.stdout:
            print("--- stdout ---\n" + r.stdout)
        if r.stderr:
            print("--- stderr ---\n" + r.stderr)
        if r.returncode != 0:
            print(f"❌ FAIL: script 12 returned {r.returncode}")
            sys.exit(1)

        # Verify outputs
        expected = [
            tables_dir / "axis2_three_way_summary.csv",
            tables_dir / "axis2_hypothesis_tests.json",
            figures_dir / "three_way_overall.png",
            figures_dir / "three_way_per_affordance_kld.png",
            figures_dir / "three_way_per_affordance_sim.png",
            figures_dir / "three_way_per_affordance_nss.png",
        ]
        missing = [p for p in expected if not p.exists()]
        if missing:
            print("❌ FAIL: missing expected outputs:")
            for p in missing:
                print(f"   {p}")
            sys.exit(2)

        # Verify hypothesis tests JSON has the keys we expect
        import json
        with open(tables_dir / "axis2_hypothesis_tests.json") as f:
            h = json.load(f)
        for hk in ("H2a", "H2b", "H2c"):
            if hk not in h:
                print(f"❌ FAIL: hypothesis test {hk} not produced")
                sys.exit(3)
            print(f"  {hk}: {list(h[hk].keys())}")

        # H2a: Flux should be borderline (we set kld_loc near 1.5, threshold is 1.7)
        h2a = h["H2a"]
        print(f"  H2a verdict — kld_pass={h2a['kld_pass']} sim_pass={h2a['sim_pass']} nss_pass={h2a['nss_pass']}")
        # H2c: with the manipulation_boost, Wilcoxon should fire — let's check
        h2c = h["H2c"]
        p_nss = h2c.get("wilcoxon_nss", {}).get("p", 1.0)
        delta = h2c.get("median_delta_nss", 0)
        print(f"  H2c verdict — median_delta_nss={delta:.3f}, wilcoxon_p(NSS)={p_nss:.4g}, pass_nss={h2c['pass_nss']}")
        if not h2c["pass_nss"]:
            print("⚠ Sanity issue: with synthetic manipulation boost = +0.40, H2c should easily reject H0. "
                  "If it doesn't, the Wilcoxon plumbing in script 12 has a bug.")
            sys.exit(4)

        print("\n✓ Synthetic-results integration test PASSED.")
        print("  Script 12 produces expected outputs and statistics fire correctly.")

    finally:
        if backup is not None:
            print(f"\nRestoring real results from {backup}/results ...")
            if (repo_root / "results").exists():
                shutil.rmtree(str(repo_root / "results"))
            shutil.copytree(str(backup / "results"), str(repo_root / "results"))
            shutil.rmtree(str(backup))
        elif not results_existed_before and (repo_root / "results").exists():
            print(f"\nRemoving synthetic results/ (none existed before) ...")
            shutil.rmtree(str(repo_root / "results"))


if __name__ == "__main__":
    main()
