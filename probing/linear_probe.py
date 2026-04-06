"""Affordance segmentation linear probe following Zhang et al. / Probe3D protocol.

Probe architecture: Upsample 4x → BatchNorm → 1x1 Conv → resize to target
- Input: frozen patch features (B, C_fused, 16, 16) from multi-layer fusion
- Output: per-pixel affordance class prediction (7 categories + background)
- Loss: cross-entropy (ignore_index=255)
- Optimizer: AdamW with cosine LR schedule + linear warmup
- Metric: mIoU over 7 affordance classes (excluding background)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class AffordanceLinearProbe(nn.Module):
    """Linear probe following Zhang et al.'s protocol."""

    def __init__(self, feature_dim, num_classes=8, image_size=224):
        super().__init__()
        self.bn = nn.BatchNorm2d(feature_dim)
        self.conv = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.image_size = image_size

    def forward(self, features):
        # features: (B, C, H_feat, W_feat) — e.g. (B, 4608, 16, 16)
        # Step 1: Upsample features 4x (16x16 → 64x64), matching Probe3D
        x = F.interpolate(features, scale_factor=4, mode='bilinear', align_corners=True)
        # Step 2: BN + 1x1 Conv (Zhang et al.'s addition to Probe3D)
        x = self.bn(x)
        x = self.conv(x)
        # Step 3: Final resize to target resolution for pixel-wise loss
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=True)
        return x


class CachedFeatureDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-extracted features from numpy files."""

    def __init__(self, features_path, masks_path):
        self.features = np.load(features_path, mmap_mode='r')
        self.masks = np.load(masks_path, mmap_mode='r')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features: (256, C_fused) tokens → reshape to (C_fused, 16, 16) spatial
        feat = torch.from_numpy(self.features[idx].copy()).float()
        feat = feat.permute(1, 0).reshape(feat.shape[1], 16, 16)
        mask = torch.from_numpy(self.masks[idx].copy()).long()
        return feat, mask


def _compute_miou(confusion, num_classes=8):
    """Compute mIoU from confusion matrix. Returns (mIoU_7class, mIoU_all, per_class)."""
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
    return miou, miou_all, per_class_iou


def _run_validation(probe, val_dataloader, device, num_classes=8):
    """Run validation and return mIoU metrics."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    probe.eval()
    with torch.no_grad():
        for features, masks in val_dataloader:
            features = features.to(device)
            masks = masks.to(device)
            logits = probe(features)
            preds = logits.argmax(dim=1)
            preds_cpu = preds.cpu()
            masks_cpu = masks.cpu()
            valid = masks_cpu != 255
            pred_valid = preds_cpu[valid].long()
            mask_valid = masks_cpu[valid].long()
            indices = mask_valid * num_classes + pred_valid
            confusion += torch.bincount(
                indices, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)
    probe.train()
    return _compute_miou(confusion, num_classes)


def train_probe_cached(train_features_path, train_masks_path, feature_dim,
                       num_classes=8, epochs=50, lr=1e-3, weight_decay=0.05,
                       warmup_fraction=0.1, batch_size=32, device="cuda",
                       num_workers=4, val_features_path=None,
                       val_masks_path=None, val_every=5, wandb_run=None):
    """Train a linear probe on cached features (no encoder needed).

    Args:
        train_features_path: path to cached features .npy (N, 256, C_fused)
        train_masks_path: path to cached masks .npy (N, 224, 224)
        feature_dim: fused feature dimension
        num_classes: number of output classes
        epochs, lr, weight_decay, warmup_fraction: Probe3D training protocol
        batch_size: training batch size
        device: torch device
        num_workers: dataloader workers
        val_features_path: optional path to val/test cached features
        val_masks_path: optional path to val/test cached masks
        val_every: compute val mIoU every N epochs (also always on last epoch)
        wandb_run: optional active wandb run for logging

    Returns:
        (probe, history) — trained probe and list of per-epoch metric dicts
    """
    dataset = CachedFeatureDataset(train_features_path, train_masks_path)
    probe = AffordanceLinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    # Validation dataloader (if paths provided)
    val_dataloader = None
    if val_features_path and val_masks_path:
        val_dataset = CachedFeatureDataset(val_features_path, val_masks_path)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    total_steps = epochs * len(dataloader)
    warmup_steps = int(warmup_fraction * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    probe.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for features, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            masks = masks.to(device)

            logits = probe(features)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        epoch_metrics = {"epoch": epoch + 1, "train/loss": avg_loss, "train/lr": current_lr}

        # Validation check
        is_last = (epoch == epochs - 1)
        do_val = val_dataloader and ((epoch + 1) % val_every == 0 or is_last)

        if do_val:
            miou, miou_all, per_class = _run_validation(
                probe, val_dataloader, device, num_classes)
            epoch_metrics["val/mIoU"] = miou
            epoch_metrics["val/mIoU_all"] = miou_all
            for c, iou in per_class.items():
                epoch_metrics[f"val/iou_class_{c}"] = iou
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"LR: {current_lr:.6f} - val mIoU: {miou:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"LR: {current_lr:.6f}")

        if wandb_run:
            wandb_run.log(epoch_metrics, step=epoch + 1)

        history.append(epoch_metrics)

    return probe, history


def evaluate_probe_cached(probe, test_features_path, test_masks_path,
                          batch_size=32, device="cuda", num_workers=4,
                          num_classes=8):
    """Evaluate probe on cached test features, compute mIoU."""
    dataset = CachedFeatureDataset(test_features_path, test_masks_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    miou, miou_all, per_class_iou = _run_validation(
        probe, dataloader, device, num_classes)

    return {
        "mIoU": miou,
        "mIoU_all": miou_all,
        "per_class_iou": per_class_iou,
    }
