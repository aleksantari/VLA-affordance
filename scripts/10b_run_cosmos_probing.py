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
    # NOTE: 2026-05-10 — Cosmos-Policy-ALOHA-Predict2-2B is NOT a diffusers
    # pipeline. The HF repo lacks `model_index.json`; it ships under the
    # NVlabs/cosmos-policy custom Python package with action heads + proprio
    # input. It cannot be loaded with Cosmos2VideoToWorldPipeline.from_pretrained.
    # H2c (manipulation amplification) is therefore deferred — see
    # axis2_research/research-log.md for the decision rationale.
    "cosmos_policy": {
        "model_name": "nvidia/Cosmos-Policy-ALOHA-Predict2-2B",
        "pipeline_type": "video2world",
        "needs_image_input": True,
        "default_height": 480,
        "default_width": 704,
        "deferred": True,
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
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip samples already present in the per-sample CSV (default: on)",
    )
    parser.add_argument(
        "--no_resume", dest="resume", action="store_false",
        help="Disable resume — overwrite existing per-sample CSV",
    )
    parser.add_argument(
        "--commit_every", type=int, default=0,
        help="If >0, run `git add+commit+push` every N samples (Colab durability)",
    )
    args = parser.parse_args()

    cfg = SYSTEM_CONFIG[args.system]
    if cfg.get("deferred"):
        print(
            f"⚠ System '{args.system}' is DEFERRED.\n"
            f"  {cfg['model_name']} is not a diffusers pipeline (no model_index.json);\n"
            f"  it requires the NVlabs/cosmos-policy custom Python package.\n"
            f"  H2c is paused — proceed with cosmos_predict2_v2w (H2b) only.\n"
            f"  See axis2_research/research-log.md for the decision."
        )
        sys.exit(0)
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
        BindingMetrics,
    )
    from interaction.incremental_results import IncrementalCSVWriter

    per_sample_path = tables_dir / f"axis2_{args.system}_per_sample.csv"
    if not args.resume and per_sample_path.exists():
        per_sample_path.unlink()
    csv_writer = IncrementalCSVWriter(
        per_sample_path,
        columns=["sample_id", "system", "affordance", "prompt", "kld", "sim", "nss"],
    )
    done_ids = csv_writer.done_sample_ids() if args.resume else set()
    if done_ids:
        print(f"  Resume: {len(done_ids)} samples already done — skipping")

    all_results = {}
    if done_ids:
        import csv as _csv
        with open(per_sample_path) as _f:
            for r in _csv.DictReader(_f):
                all_results[r["sample_id"]] = BindingMetrics(
                    kld=float(r["kld"]), sim=float(r["sim"]), nss=float(r["nss"]),
                    affordance=r.get("affordance", ""), prompt=r.get("prompt", ""),
                )

    total = len(done_ids)
    skipped = 0

    def _maybe_git_push(n_done: int):
        if not args.commit_every or n_done % args.commit_every != 0:
            return
        try:
            import subprocess
            subprocess.run(["git", "add", str(per_sample_path)], check=False)
            subprocess.run(
                ["git", "commit", "-m",
                 f"results({args.system}): incremental {n_done} samples"],
                check=False,
            )
            subprocess.run(["git", "push", "origin", "HEAD"], check=False)
        except Exception as _e:
            print(f"    ⚠ git push failed: {_e}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, *a, **k):
            return it

    # Total work units = sum of per-category sample budgets, minus already done.
    total_to_do = sum(
        min(len(dataset.get_samples_by_affordance(c)),
            args.max_per_category or 10**9)
        for c in categories
    ) - len(done_ids)
    pbar = tqdm(
        total=max(total_to_do, 0),
        desc=f"{args.system}",
        unit="sample",
        dynamic_ncols=True,
    )

    for cat_idx, category in enumerate(categories):
        sample_indices = dataset.get_samples_by_affordance(category)
        if args.max_per_category:
            sample_indices = sample_indices[:args.max_per_category]

        for i, sample_idx in enumerate(sample_indices):
            sample = dataset[sample_idx]
            sample_id = f"{category}_{i}"

            if sample_id in done_ids:
                continue

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

                csv_writer.append({
                    "sample_id": sample_id, "system": args.system,
                    "affordance": category, "prompt": prompt,
                    "kld": metrics.kld, "sim": metrics.sim, "nss": metrics.nss,
                })
                _maybe_git_push(total)

                if args.save_attention_maps:
                    np.save(str(cache_dir / f"{sample_id}.npy"), pred_map)

                pbar.set_postfix({
                    "cat": category[:6],
                    "KLD": f"{metrics.kld:.2f}",
                    "SIM": f"{metrics.sim:.2f}",
                    "NSS": f"{metrics.nss:.2f}",
                })
                pbar.update(1)

            except Exception as e:
                print(f"    ⚠ Error on {sample_id}: {e}")
                skipped += 1
                pbar.update(1)

    pbar.close()
    print(f"\nProcessed: {total}, skipped: {skipped}")

    # Step 4: aggregate + save
    aggregated = aggregate_metrics_by_affordance(all_results)
    print_metrics_table(aggregated)

    csv_writer.close()  # per-sample CSV is durable on disk via incremental writes

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
