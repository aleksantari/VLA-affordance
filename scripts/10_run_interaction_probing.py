"""
Script 10: Run Interaction Affordance Probing (Axis 2)

Full evaluation pipeline:
1. Load AGD20K egocentric dataset
2. For each affordance category, construct prompts and extract verb attention
3. Compute KLD/SIM/NSS against ground truth affordance heatmaps
4. Save per-category and overall metrics
5. Cache attention maps for visualization

Usage:
    python scripts/10_run_interaction_probing.py
    python scripts/10_run_interaction_probing.py --model dev --max_per_category 50
    python scripts/10_run_interaction_probing.py --categories cut hold pour
"""

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run Axis 2 interaction affordance probing"
    )
    parser.add_argument(
        "--model", type=str, default="schnell",
        choices=["schnell", "dev"],
        help="Flux variant"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/agd20k",
        help="Path to AGD20K dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Output directory for metrics and cached attention"
    )
    parser.add_argument(
        "--max_per_category", type=int, default=None,
        help="Max samples per affordance category (for quick testing)"
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None,
        help="Specific affordance categories to evaluate (default: all)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Image generation size"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true",
        help="Use CPU offloading"
    )
    parser.add_argument(
        "--save_attention_maps", action="store_true",
        help="Save raw attention maps (uses disk space)"
    )
    parser.add_argument(
        "--num_vis_examples", type=int, default=3,
        help="Number of visualization examples per category"
    )
    args = parser.parse_args()
    
    # Paths
    model_map = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "dev": "black-forest-labs/FLUX.1-dev",
    }
    model_name = model_map[args.model]
    
    tables_dir = Path(args.output_dir) / "tables"
    figures_dir = Path(args.output_dir) / "figures" / "axis2"
    cache_dir = Path(args.output_dir) / "cached_features" / "flux_attention"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AXIS 2 — Interaction Affordance Probing")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Image size: {args.image_size}")
    if args.max_per_category:
        print(f"Max per category: {args.max_per_category}")
    if args.categories:
        print(f"Categories: {args.categories}")
    print()
    
    # ── Step 1: Load dataset ──
    print("Step 1: Loading AGD20K dataset...")
    
    from data.agd20k_dataset import AGD20KDataset
    
    dataset = AGD20KDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
    )
    
    categories = args.categories or dataset.get_affordance_categories()
    print(f"  Evaluating {len(categories)} affordance categories")
    print()
    
    # ── Step 2: Initialize Flux extractor ──
    print("Step 2: Loading Flux pipeline...")
    
    from interaction.flux_attention import FluxVerbAttentionExtractor
    
    extractor = FluxVerbAttentionExtractor(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        enable_cpu_offload=args.cpu_offload,
    )
    
    # ── Step 3: Run probing ──
    print("\nStep 3: Running verb-spatial binding probing...")
    
    from interaction.verb_spatial_binding import (
        evaluate_single,
        aggregate_metrics_by_affordance,
        print_metrics_table,
        BindingMetrics,
    )
    from interaction.visualization import (
        plot_gt_vs_predicted,
        plot_attention_overlay,
    )
    
    all_results = {}  # sample_id -> BindingMetrics
    vis_examples = defaultdict(list)  # affordance -> list of vis dicts
    
    total_samples = 0
    skipped = 0
    
    for cat_idx, category in enumerate(categories):
        sample_indices = dataset.get_samples_by_affordance(category)
        
        if args.max_per_category:
            sample_indices = sample_indices[:args.max_per_category]
        
        print(f"\n[{cat_idx+1}/{len(categories)}] {category} "
              f"({len(sample_indices)} samples)")
        
        for i, sample_idx in enumerate(sample_indices):
            sample = dataset[sample_idx]
            sample_id = f"{category}_{i}"
            
            # Skip if no ground truth heatmap
            if sample["gt_heatmap"] is None:
                skipped += 1
                continue
            
            prompt = sample["prompt"]
            verb_roots = sample["verb_roots"]
            verb = sample["verb_gerund"]
            
            try:
                # Extract verb attention
                result = extractor.extract(
                    prompt=prompt,
                    verbs=[verb],
                    height=args.image_size,
                    width=args.image_size,
                    seed=args.seed + i,
                    store_per_timestep=False,
                )
                
                # Get the attention map for the verb
                if verb in result.verb_attention_maps:
                    pred_map = result.verb_attention_maps[verb]
                else:
                    # Try with first available key
                    pred_map = list(result.verb_attention_maps.values())[0] \
                        if result.verb_attention_maps else np.zeros((32, 32))
                
                # Evaluate
                metrics = evaluate_single(
                    pred_map=pred_map,
                    gt_heatmap=sample["gt_heatmap"],
                    affordance=category,
                    prompt=prompt,
                )
                
                all_results[sample_id] = metrics
                total_samples += 1
                
                # Save visualization examples
                if len(vis_examples[category]) < args.num_vis_examples:
                    vis_examples[category].append({
                        "image": sample["image"],
                        "gt_heatmap": sample["gt_heatmap"],
                        "pred_heatmap": pred_map,
                        "affordance": category,
                        "prompt": prompt,
                        "metrics": {
                            "kld": metrics.kld,
                            "sim": metrics.sim,
                            "nss": metrics.nss,
                        },
                    })
                
                # Save raw attention map if requested
                if args.save_attention_maps:
                    map_path = cache_dir / f"{sample_id}.npy"
                    np.save(str(map_path), pred_map)
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"    {i+1}/{len(sample_indices)} — "
                          f"KLD={metrics.kld:.3f} SIM={metrics.sim:.3f} "
                          f"NSS={metrics.nss:.3f}")
                    
            except Exception as e:
                print(f"    ⚠ Error on sample {sample_id}: {e}")
                skipped += 1
                continue
    
    print(f"\nProcessed: {total_samples} samples, skipped: {skipped}")
    
    # ── Step 4: Aggregate and save metrics ──
    print("\nStep 4: Aggregating metrics...")
    
    aggregated = aggregate_metrics_by_affordance(all_results)
    print_metrics_table(aggregated)
    
    # Save per-sample metrics
    per_sample_path = tables_dir / "axis2_per_sample.csv"
    with open(per_sample_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "affordance", "prompt", "kld", "sim", "nss"])
        for sid, m in sorted(all_results.items()):
            writer.writerow([sid, m.affordance, m.prompt, 
                           f"{m.kld:.6f}", f"{m.sim:.6f}", f"{m.nss:.6f}"])
    print(f"  Saved per-sample metrics: {per_sample_path}")
    
    # Save aggregated metrics
    agg_path = tables_dir / "axis2_metrics.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"  Saved aggregated metrics: {agg_path}")
    
    # Save as CSV too
    agg_csv_path = tables_dir / "axis2_metrics.csv"
    with open(agg_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["affordance", "count", "kld_mean", "kld_std",
                         "sim_mean", "sim_std", "nss_mean", "nss_std"])
        for aff, stats in sorted(aggregated.items()):
            writer.writerow([
                aff, stats["count"],
                f"{stats['kld_mean']:.4f}", f"{stats['kld_std']:.4f}",
                f"{stats['sim_mean']:.4f}", f"{stats['sim_std']:.4f}",
                f"{stats['nss_mean']:.4f}", f"{stats['nss_std']:.4f}",
            ])
    print(f"  Saved aggregated CSV: {agg_csv_path}")
    
    # ── Step 5: Generate visualizations ──
    print("\nStep 5: Generating visualizations...")
    
    for category, examples in vis_examples.items():
        for j, ex in enumerate(examples):
            fig_path = figures_dir / f"{category}_{j}_comparison.png"
            plot_gt_vs_predicted(
                image=ex["image"],
                gt_heatmap=ex["gt_heatmap"],
                pred_heatmap=ex["pred_heatmap"],
                affordance=category,
                metrics=ex["metrics"],
                save_path=str(fig_path),
            )
            plt_module = sys.modules.get("matplotlib.pyplot")
            if plt_module:
                plt_module.close("all")
    
    print(f"  Saved visualizations to {figures_dir}")
    
    # ── Summary ──
    print()
    print("=" * 60)
    print("PROBING COMPLETE")
    print("=" * 60)
    
    overall = aggregated.get("OVERALL", {})
    print(f"Total samples: {total_samples}")
    print(f"Overall KLD↓: {overall.get('kld_mean', 0):.4f} ± {overall.get('kld_std', 0):.4f}")
    print(f"Overall SIM↑: {overall.get('sim_mean', 0):.4f} ± {overall.get('sim_std', 0):.4f}")
    print(f"Overall NSS↑: {overall.get('nss_mean', 0):.4f} ± {overall.get('nss_std', 0):.4f}")
    print(f"\nOutput files:")
    print(f"  {agg_csv_path}")
    print(f"  {agg_path}")
    print(f"  {per_sample_path}")
    print(f"  {figures_dir}/")


if __name__ == "__main__":
    main()
