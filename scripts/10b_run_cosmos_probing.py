"""
Script 10b: Run Interaction Affordance Probing — Cosmos systems (Axis 2)

Cosmos-specific counterpart to scripts/10_run_interaction_probing.py.
Supports both Cosmos-Predict2-2B-Video2World (H2b base video diffusion)
and Cosmos-Policy-ALOHA-Predict2-2B (H2c manipulation-fine-tuned).

The output CSV/JSON layout matches script 10 exactly, so script 12
(three-way comparison) can ingest results from all systems uniformly.

Usage:
    python scripts/10b_run_cosmos_probing.py --system cosmos_predict2_v2w
    python scripts/10b_run_cosmos_probing.py --system cosmos_policy --max_per_category 30

The --system flag selects:
    cosmos_predict2_t2i  → nvidia/Cosmos-Predict2-2B-Text2Image
    cosmos_predict2_v2w  → nvidia/Cosmos-Predict2-2B-Video2World  (H2b)
    cosmos_policy        → nvidia/Cosmos-Policy-ALOHA-Predict2-2B  (H2c)
"""

import argparse
import json
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


SYSTEM_CONFIG = {
    "cosmos_predict2_t2i": {
        "model_name": "nvidia/Cosmos-Predict2-2B-Text2Image",
        "pipeline_type": "text2image",
        "needs_image_input": False,
        "default_height": 768,
        "default_width": 768,
    },
    "cosmos_predict2_v2w": {
        "model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
        "pipeline_type": "video2world",
        "needs_image_input": True,
        "default_height": 480,
        "default_width": 704,
    },
    "cosmos_policy": {
        "model_name": "nvidia/Cosmos-Policy-ALOHA-Predict2-2B",
        "pipeline_type": "video2world",  # same base architecture
        "needs_image_input": True,
        "default_height": 480,
        "default_width": 704,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Cosmos verb-spatial binding probing (Axis 2)"
    )
    parser.add_argument(
        "--system", type=str, required=True, choices=list(SYSTEM_CONFIG.keys()),
        help="Cosmos system to probe",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/agd20k",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
    )
    parser.add_argument(
        "--max_per_category", type=int, default=30,
        help="Max samples per affordance category (default 30 for pilot)",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None,
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument(
        "--save_attention_maps", action="store_true",
        help="Save raw per-sample attention maps to disk (uses space)",
    )
    args = parser.parse_args()

    cfg = SYSTEM_CONFIG[args.system]
    height = args.height or cfg["default_height"]
    width = args.width or cfg["default_width"]

    tables_dir = Path(args.output_dir) / "tables"
    figures_dir = Path(args.output_dir) / "figures" / "axis2" / args.system
    cache_dir = Path(args.output_dir) / "cached_features" / f"{args.system}_attention"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"AXIS 2 — Cosmos Interaction Probing: {args.system}")
    print("=" * 60)
    print(f"Model: {cfg['model_name']}")
    print(f"Pipeline type: {cfg['pipeline_type']}")
    print(f"Resolution: {height}×{width}")
    if cfg["pipeline_type"] == "video2world":
        print(f"Num frames: {args.num_frames}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Max per category: {args.max_per_category}")
    print()

    # Step 1: dataset
    print("Step 1: Loading AGD20K dataset...")
    from data.agd20k_dataset import AGD20KDataset
    dataset = AGD20KDataset(
        data_dir=args.data_dir,
        image_size=max(height, width),
    )
    categories = args.categories or dataset.get_affordance_categories()
    print(f"  {len(categories)} affordance categories")
    print()

    # Step 2: extractor
    print("Step 2: Loading Cosmos pipeline...")
    from interaction.cosmos_attention import CosmosVerbAttentionExtractor

    extractor = CosmosVerbAttentionExtractor(
        model_name=cfg["model_name"],
        pipeline_type=cfg["pipeline_type"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        enable_cpu_offload=args.cpu_offload,
    )

    # Step 3: probe
    print("\nStep 3: Running verb-spatial binding probing...")
    from interaction.verb_spatial_binding import (
        evaluate_single,
        aggregate_metrics_by_affordance,
        print_metrics_table,
    )

    all_results = {}
    total = 0
    skipped = 0

    for cat_idx, category in enumerate(categories):
        sample_indices = dataset.get_samples_by_affordance(category)
        if args.max_per_category:
            sample_indices = sample_indices[:args.max_per_category]

        print(f"\n[{cat_idx+1}/{len(categories)}] {category} ({len(sample_indices)} samples)")

        for i, sample_idx in enumerate(sample_indices):
            sample = dataset[sample_idx]
            sample_id = f"{category}_{i}"

            if sample["gt_heatmap"] is None:
                skipped += 1
                continue

            prompt = sample["prompt"]
            verb = sample["verb_gerund"]
            img = sample["image"] if cfg["needs_image_input"] else None

            try:
                result = extractor.extract(
                    prompt=prompt,
                    verbs=[verb],
                    image=img,
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    num_frames=args.num_frames,
                    seed=args.seed + i,
                    store_per_timestep=False,
                )

                if verb in result.verb_attention_maps:
                    pred_map = result.verb_attention_maps[verb]
                else:
                    vals = list(result.verb_attention_maps.values())
                    pred_map = vals[0] if vals else np.zeros((32, 32), dtype=np.float32)

                metrics = evaluate_single(
                    pred_map=pred_map,
                    gt_heatmap=sample["gt_heatmap"],
                    affordance=category,
                    prompt=prompt,
                )
                all_results[sample_id] = metrics
                total += 1

                if args.save_attention_maps:
                    np.save(str(cache_dir / f"{sample_id}.npy"), pred_map)

                if (i + 1) % 10 == 0:
                    print(
                        f"    {i+1}/{len(sample_indices)} — "
                        f"KLD={metrics.kld:.3f} SIM={metrics.sim:.3f} NSS={metrics.nss:.3f}"
                    )

            except Exception as e:
                print(f"    ⚠ Error on {sample_id}: {e}")
                skipped += 1

    print(f"\nProcessed: {total}, skipped: {skipped}")

    # Step 4: aggregate + save
    aggregated = aggregate_metrics_by_affordance(all_results)
    print_metrics_table(aggregated)

    per_sample_path = tables_dir / f"axis2_{args.system}_per_sample.csv"
    with open(per_sample_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "system", "affordance", "prompt", "kld", "sim", "nss"])
        for sid, m in sorted(all_results.items()):
            w.writerow([sid, args.system, m.affordance, m.prompt,
                        f"{m.kld:.6f}", f"{m.sim:.6f}", f"{m.nss:.6f}"])

    agg_path = tables_dir / f"axis2_{args.system}_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    agg_csv_path = tables_dir / f"axis2_{args.system}_metrics.csv"
    with open(agg_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["affordance", "count", "kld_mean", "kld_std",
                    "sim_mean", "sim_std", "nss_mean", "nss_std"])
        for aff, stats in sorted(aggregated.items()):
            w.writerow([aff, stats["count"],
                        f"{stats['kld_mean']:.4f}", f"{stats['kld_std']:.4f}",
                        f"{stats['sim_mean']:.4f}", f"{stats['sim_std']:.4f}",
                        f"{stats['nss_mean']:.4f}", f"{stats['nss_std']:.4f}"])

    print()
    print("=" * 60)
    print("PROBING COMPLETE")
    print("=" * 60)
    overall = aggregated.get("OVERALL", {})
    print(f"System: {args.system}")
    print(f"Total samples: {total}")
    print(f"Overall KLD↓: {overall.get('kld_mean', 0):.4f} ± {overall.get('kld_std', 0):.4f}")
    print(f"Overall SIM↑: {overall.get('sim_mean', 0):.4f} ± {overall.get('sim_std', 0):.4f}")
    print(f"Overall NSS↑: {overall.get('nss_mean', 0):.4f} ± {overall.get('nss_std', 0):.4f}")
    print()
    print(f"Outputs:")
    print(f"  {per_sample_path}")
    print(f"  {agg_path}")
    print(f"  {agg_csv_path}")


if __name__ == "__main__":
    main()
