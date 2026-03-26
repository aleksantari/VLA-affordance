"""Step 8: Compile all results into figures & tables for the report."""

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

    # 3. Depth/normal augmentation
    print("\n3. DEPTH/NORMAL AUGMENTATION (mIoU delta)")
    print("-" * 60)
    da_results = load_json(tables_dir / "depth_augmentation_results.json")
    if da_results:
        print(f"  {'Encoder':25s} {'Visual':>8s} {'+ Depth':>8s} {'Delta':>8s}")
        print(f"  {'-'*55}")
        for name, res in da_results.items():
            v = f"{res['visual_only_mIoU']:.4f}" if res.get("visual_only_mIoU") else "  N/A "
            d = f"{res['delta']:+.4f}" if res.get("delta") is not None else "  N/A "
            print(f"  {name:25s} {v:>8s} {res['mIoU']:8.4f} {d:>8s}")

    # 4. Interpretation
    print("\n4. INTERPRETATION GUIDE")
    print("-" * 40)
    if miou_dict:
        raw_miou = miou_dict.get("raw_siglip", 0)
        pali_miou = miou_dict.get("paligemma_siglip", 0)
        pi0_miou = miou_dict.get("pi0_siglip", 0)
        pi05_miou = miou_dict.get("pi05_siglip", 0)
        dinov2_miou = miou_dict.get("dinov2", 0)

        print("  SigLIP progression:")
        print(f"    Raw SigLIP:       {raw_miou:.4f}")
        print(f"    PaliGemma SigLIP: {pali_miou:.4f} ({pali_miou - raw_miou:+.4f} from raw)")
        print(f"    pi0 SigLIP:       {pi0_miou:.4f} ({pi0_miou - pali_miou:+.4f} from PaliGemma)")
        print(f"    pi0.5 SigLIP:     {pi05_miou:.4f} ({pi05_miou - pali_miou:+.4f} from PaliGemma)")
        print(f"    DINOv2 ceiling:   {dinov2_miou:.4f}")

        gap_pali_raw = pali_miou - raw_miou
        gap_pi0_pali = pi0_miou - pali_miou

        if gap_pali_raw > 0.05:
            print("  -> Multimodal VL training improves geometric features")
        if gap_pi0_pali < -0.03:
            print("  -> Robot training damages geometric features (pi0)")
        elif gap_pi0_pali > 0.03:
            print("  -> Robot training further improves geometric features")

    print("\n" + "=" * 60)
    print("Report generation complete.")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to: {tables_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_report()
