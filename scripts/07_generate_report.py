"""Step 7: Compile all results into figures & tables for the report."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path

from evaluation.visualization import plot_miou_comparison


def load_json(path):
    """Load JSON results file."""
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def generate_report(tables_dir="./results/tables", figures_dir="./results/figures"):
    """Compile all results into a summary report."""
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    print("=" * 60)
    print("AFFORDANCE PROBING RESULTS REPORT")
    print("=" * 60)

    # 1. Linear probing results
    print("\n1. LINEAR PROBING (mIoU)")
    print("-" * 40)
    lp_results = load_json(tables_dir / "linear_probing_results.json")
    miou_dict = {}
    for name, res in lp_results.items():
        miou = res.get("mIoU", 0)
        miou_dict[name] = miou
        print(f"  {name:20s}: {miou:.4f}")

    if miou_dict:
        plot_miou_comparison(miou_dict, save_path=figures_dir / "miou_comparison.png")

    # 2. Weight divergence
    print("\n2. WEIGHT DIVERGENCE")
    print("-" * 40)
    wd_results = load_json(tables_dir / "weight_divergence_results.json")
    for name, summary in wd_results.items():
        if "error" in summary:
            print(f"  {name}: ERROR - {summary['error']}")
        else:
            identical = summary.get("weights_identical", "unknown")
            mean_change = summary.get("mean_relative_change", "unknown")
            print(f"  {name}:")
            print(f"    Weights identical: {identical}")
            print(f"    Mean relative change: {mean_change}")

    # 3. Cosine similarity
    print("\n3. COSINE SIMILARITY (Part Correspondence)")
    print("-" * 40)
    cs_results = load_json(tables_dir / "cosine_similarity_results.json")
    for name, res in cs_results.items():
        hit_k = res.get("hit_at_k", "N/A")
        ratio = res.get("similarity_ratio", "N/A")
        print(f"  {name:20s}: Hit@K={hit_k}, Ratio={ratio}")

    # 4. Interpretation
    print("\n4. INTERPRETATION GUIDE")
    print("-" * 40)
    if miou_dict:
        raw_miou = miou_dict.get("raw_siglip", 0)
        pi0_miou = miou_dict.get("pi0_siglip", 0)
        dinov2_miou = miou_dict.get("dinov2", 0)

        gap_pi0_raw = pi0_miou - raw_miou
        gap_pi0_dinov2 = dinov2_miou - pi0_miou

        print(f"  pi0 improvement over raw SigLIP: {gap_pi0_raw:+.4f}")
        print(f"  Gap between pi0 SigLIP and DINOv2: {gap_pi0_dinov2:.4f}")

        if gap_pi0_raw < 0.05:
            print("  -> VLA fine-tuning doesn't fix geometric limitations")
        elif gap_pi0_dinov2 > 0.15:
            print("  -> Partial recovery, but gap is structural")
        elif gap_pi0_dinov2 < 0.05:
            print("  -> Near full recovery! Training > architecture")

    print("\n" + "=" * 60)
    print("Report generation complete.")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to: {tables_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_report()
