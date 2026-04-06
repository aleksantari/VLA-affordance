"""Step 6: Run depth/normal augmentation experiment.

Concatenates pre-cached multi-layer encoder features with depth/normal
features, trains a linear probe on the augmented features, and computes
the mIoU delta vs visual-only probing.

Delta = mIoU(with_depth) - mIoU(without_depth)
- Small delta: encoder already encodes strong geometry
- Large delta: encoder relies on external geometric help
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from encoders.feature_extractor import ENCODER_REGISTRY
from probing.linear_probe import AffordanceLinearProbe


ENCODER_NAMES = [
    "raw_siglip", "paligemma_siglip", "pi0_siglip", "pi05_siglip",
    "dinov2", "dino_wm",
]


def load_cached_features(cache_dir, encoder_name, split):
    """Load pre-cached multi-layer features and masks."""
    cache_path = Path(cache_dir) / encoder_name
    features = np.load(cache_path / "features_multilayer.npy")  # (N, 256, C_fused) float16
    masks = np.load(cache_path / "masks.npy")  # (N, 224, 224)
    return features.astype(np.float32), masks


def load_depth_normal(cache_dir, split):
    """Load pre-cached depth/normal features."""
    path = Path(cache_dir) / "depth_normal" / split / "depth_normal.npy"
    return np.load(str(path))  # (N, 4, 16, 16)


def features_to_spatial(features):
    """Reshape token features to spatial grid: (N, 256, C) -> (N, C, 16, 16)."""
    N, num_patches, C = features.shape
    H = W = int(num_patches ** 0.5)
    return features.reshape(N, H, W, C).transpose(0, 3, 1, 2)


def train_augmented_probe(train_features, train_masks, feature_dim, num_classes=8,
                          epochs=50, lr=1e-3, weight_decay=0.05, warmup_fraction=0.1,
                          batch_size=32, device="cuda"):
    """Train a linear probe on depth-augmented features from cached arrays.

    Uses AdamW with cosine LR schedule + linear warmup, following Probe3D.
    """
    probe = AffordanceLinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Create tensor dataset
    features_tensor = torch.from_numpy(train_features).float()
    masks_tensor = torch.from_numpy(train_masks).long()
    dataset = TensorDataset(features_tensor, masks_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Cosine LR schedule with linear warmup (Probe3D protocol)
    total_steps = epochs * len(dataloader)
    warmup_steps = int(warmup_fraction * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    probe.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for features_batch, masks_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            features_batch = features_batch.to(device)
            masks_batch = masks_batch.to(device)

            logits = probe(features_batch)
            loss = criterion(logits, masks_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

    return probe


def evaluate_augmented_probe(test_features, test_masks, probe, num_classes=8,
                             batch_size=32, device="cuda"):
    """Evaluate probe on test set, return mIoU."""
    features_tensor = torch.from_numpy(test_features).float()
    masks_tensor = torch.from_numpy(test_masks).long()
    dataset = TensorDataset(features_tensor, masks_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    probe.eval()
    with torch.no_grad():
        for features_batch, masks_batch in dataloader:
            features_batch = features_batch.to(device)
            logits = probe(features_batch)
            preds = logits.argmax(dim=1).cpu()

            valid = masks_batch != 255
            pred_valid = preds[valid].long()
            mask_valid = masks_batch[valid].long()
            indices = mask_valid * num_classes + pred_valid
            confusion += torch.bincount(
                indices, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # Compute mIoU — exclude background (class 0), matching Zhang et al.
    per_class_iou = {}
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
        per_class_iou[c] = iou

    affordance_ious = [v for k, v in per_class_iou.items() if k != 0]
    miou = np.mean(affordance_ious) if affordance_ious else 0.0
    miou_all = np.mean(list(per_class_iou.values()))
    return {"mIoU": miou, "mIoU_all": miou_all, "per_class_iou": per_class_iou}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./results/cached_features")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load visual-only results for delta computation
    visual_results_path = Path("./results/tables/linear_probing_results.json")
    visual_results = {}
    if visual_results_path.exists():
        with open(visual_results_path) as f:
            visual_results = json.load(f)

    # Load depth/normal features (shared across encoders)
    print("Loading depth/normal features...")
    train_depth = load_depth_normal(args.cache_dir, "train")
    test_depth = load_depth_normal(args.cache_dir, "test")
    print(f"  Train: {train_depth.shape}, Test: {test_depth.shape}")

    all_results = {}

    for encoder_name in args.encoders:
        print(f"\n{'='*50}")
        print(f"Depth Augmentation: {encoder_name}")
        print(f"{'='*50}")

        try:
            # Load cached multi-layer features
            train_features, train_masks = load_cached_features(args.cache_dir, encoder_name, "train")
            test_features, test_masks = load_cached_features(args.cache_dir, encoder_name, "test")
        except FileNotFoundError as e:
            print(f"  SKIP: cached features not found ({e})")
            continue

        # Reshape to spatial: (N, 256, C) -> (N, C, 16, 16)
        train_spatial = features_to_spatial(train_features)
        test_spatial = features_to_spatial(test_features)

        # Concatenate with depth/normal: (N, C+4, 16, 16)
        train_augmented = np.concatenate([train_spatial, train_depth], axis=1)
        test_augmented = np.concatenate([test_spatial, test_depth], axis=1)

        augmented_dim = train_augmented.shape[1]
        print(f"  Augmented feature dim: {augmented_dim} (visual {train_spatial.shape[1]} + depth 4)")

        # Train and evaluate
        probe = train_augmented_probe(
            train_augmented, train_masks, augmented_dim,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device,
        )
        results = evaluate_augmented_probe(
            test_augmented, test_masks, probe, device=device,
        )

        # Compute delta
        visual_miou = visual_results.get(encoder_name, {}).get("mIoU", None)
        delta = results["mIoU"] - visual_miou if visual_miou is not None else None

        results["visual_only_mIoU"] = visual_miou
        results["delta"] = delta
        all_results[encoder_name] = results

        print(f"  mIoU (with depth): {results['mIoU']:.4f}")
        if visual_miou is not None:
            print(f"  mIoU (visual only): {visual_miou:.4f}")
            print(f"  Delta: {delta:+.4f}")

        del probe
        torch.cuda.empty_cache()

    # Save results
    results_dir = Path("./results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "depth_augmentation_results.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "per_class_iou"}
             for k, v in all_results.items()},
            f, indent=2,
        )

    print("\n\nDepth Augmentation Results:")
    print("-" * 60)
    print(f"  {'Encoder':25s} {'Visual':>8s} {'+ Depth':>8s} {'Delta':>8s}")
    print("-" * 60)
    for name, res in all_results.items():
        v = f"{res['visual_only_mIoU']:.4f}" if res["visual_only_mIoU"] else "  N/A "
        d = f"{res['delta']:+.4f}" if res["delta"] is not None else "  N/A "
        print(f"  {name:25s} {v:>8s} {res['mIoU']:8.4f} {d:>8s}")
