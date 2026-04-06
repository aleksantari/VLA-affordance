"""Step 3: Train & evaluate linear probes for all encoders."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path

from encoders.feature_extractor import UnifiedFeatureExtractor, ENCODER_REGISTRY
from data.umd_dataset import UMDAffordanceDataset
from probing.linear_probe import train_probe, evaluate_probe


ENCODER_NAMES = [
    "raw_siglip", "paligemma_siglip", "pi0_siglip", "pi05_siglip",
    "dinov2", "dino_wm",
]


def run_linear_probing(encoder_name, train_dataset, test_dataset, device="cuda",
                       epochs=50, lr=1e-3, weight_decay=0.05, batch_size=32):
    """Train and evaluate a linear probe for one encoder."""
    print(f"\n{'='*50}")
    print(f"Linear Probing: {encoder_name}")
    print(f"{'='*50}")

    try:
        extractor = UnifiedFeatureExtractor(encoder_name, device=device)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    feature_dim = ENCODER_REGISTRY[encoder_name]["fused_feature_dim"]

    # Train
    probe = train_probe(
        extractor, train_dataset, feature_dim,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, device=device,
    )

    # Evaluate
    results = evaluate_probe(extractor, probe, test_dataset,
                             batch_size=batch_size, device=device)

    print(f"\n  mIoU: {results['mIoU']:.4f}")
    print(f"  Per-class IoU: {results['per_class_iou']}")

    # Save probe checkpoint
    probe_dir = Path("./results/probes")
    probe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), probe_dir / f"{encoder_name}_probe.pt")

    del extractor, probe
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/umd_dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = UMDAffordanceDataset(args.data_dir, split="train")
    test_dataset = UMDAffordanceDataset(args.data_dir, split="test")

    all_results = {}
    for encoder_name in args.encoders:
        results = run_linear_probing(
            encoder_name, train_dataset, test_dataset,
            device=device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
        )
        if results:
            all_results[encoder_name] = {
                "mIoU": results["mIoU"],
                "mIoU_all": results["mIoU_all"],
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
