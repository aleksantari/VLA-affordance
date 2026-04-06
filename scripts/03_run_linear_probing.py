"""Step 3: Train & evaluate linear probes using cached features."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path

from encoders.feature_extractor import ENCODER_REGISTRY
from probing.linear_probe import train_probe_cached, evaluate_probe_cached
from data.umd_dataset import UMD_CATEGORIES


ENCODER_NAMES = [
    "raw_siglip", "paligemma_siglip", "pi0_siglip", "pi05_siglip",
    "dinov2", "dino_wm",
]


def run_linear_probing(encoder_name, cache_dir, device="cuda",
                       epochs=50, lr=1e-3, weight_decay=0.05, batch_size=32,
                       wandb_project=None):
    """Train and evaluate a linear probe for one encoder using cached features."""
    print(f"\n{'='*50}")
    print(f"Linear Probing: {encoder_name}")
    print(f"{'='*50}")

    train_feat = Path(cache_dir) / encoder_name / "train" / "features_multilayer.npy"
    train_mask = Path(cache_dir) / encoder_name / "train" / "masks.npy"
    test_feat = Path(cache_dir) / encoder_name / "test" / "features_multilayer.npy"
    test_mask = Path(cache_dir) / encoder_name / "test" / "masks.npy"

    for p in [train_feat, train_mask, test_feat, test_mask]:
        if not p.exists():
            print(f"  Missing cached features: {p}")
            print("  Run 02_extract_features.py first.")
            return None

    feature_dim = ENCODER_REGISTRY[encoder_name]["fused_feature_dim"]

    # Init wandb run for this encoder
    wandb_run = None
    if wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=encoder_name,
            config={
                "encoder": encoder_name,
                "feature_dim": feature_dim,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "optimizer": "adamw",
                "lr_schedule": "cosine_with_warmup",
                "warmup_fraction": 0.1,
                "num_classes": 8,
                "miou_classes": 7,
            },
            reinit=True,
        )

    # Train with validation monitoring
    probe, history = train_probe_cached(
        str(train_feat), str(train_mask), feature_dim,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, device=device,
        val_features_path=str(test_feat),
        val_masks_path=str(test_mask),
        wandb_run=wandb_run,
    )

    # Final evaluation
    results = evaluate_probe_cached(
        probe, str(test_feat), str(test_mask),
        batch_size=batch_size, device=device,
    )

    print(f"\n  mIoU (7-class): {results['mIoU']:.4f}")
    print(f"  mIoU (all):     {results['mIoU_all']:.4f}")
    print("  Per-class IoU:")
    for c, iou in results['per_class_iou'].items():
        name = UMD_CATEGORIES.get(c, f"class_{c}")
        print(f"    {name:12s}: {iou:.4f}")

    # Log final test metrics to wandb
    if wandb_run:
        final_metrics = {
            "test/mIoU": results["mIoU"],
            "test/mIoU_all": results["mIoU_all"],
        }
        for c, iou in results["per_class_iou"].items():
            name = UMD_CATEGORIES.get(c, f"class_{c}")
            final_metrics[f"test/iou_{name}"] = iou
        wandb_run.log(final_metrics)
        wandb_run.finish()

    # Save probe checkpoint
    probe_dir = Path("./results/probes")
    probe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), probe_dir / f"{encoder_name}_probe.pt")

    del probe
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./results/cached_features")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    parser.add_argument("--wandb_project", default="affordance-probing")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project = None if args.no_wandb else args.wandb_project

    all_results = {}
    for encoder_name in args.encoders:
        results = run_linear_probing(
            encoder_name, args.cache_dir,
            device=device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
            wandb_project=wandb_project,
        )
        if results:
            all_results[encoder_name] = {
                "mIoU": float(results["mIoU"]),
                "mIoU_all": float(results["mIoU_all"]),
                "per_class_iou": {str(k): v for k, v in results["per_class_iou"].items()},
            }

    # Save results
    results_dir = Path("./results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "linear_probing_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nFinal Results:")
    print("-" * 40)
    for name, res in all_results.items():
        print(f"  {name:20s}: mIoU = {res['mIoU']:.4f}")
