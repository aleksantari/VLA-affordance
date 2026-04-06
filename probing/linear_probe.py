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


def train_probe(encoder, dataset, feature_dim, num_classes=8, epochs=50,
                lr=1e-3, weight_decay=0.05, warmup_fraction=0.1,
                batch_size=32, device="cuda", num_workers=4):
    """Train a linear probe on frozen encoder features.

    Uses AdamW with cosine LR schedule + linear warmup, following Probe3D.

    Args:
        encoder: UnifiedFeatureExtractor instance
        dataset: UMDAffordanceDataset (train split)
        feature_dim: dimension of encoder features
        num_classes: number of output classes
        epochs: training epochs
        lr: peak learning rate
        weight_decay: AdamW weight decay
        warmup_fraction: fraction of total steps for linear warmup
        batch_size: batch size
        device: torch device
        num_workers: dataloader workers

    Returns:
        Trained AffordanceLinearProbe
    """
    probe = AffordanceLinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

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

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            masks = masks.to(device)

            # Extract features (frozen encoder, no gradients)
            with torch.no_grad():
                features = encoder.extract_multilayer_spatial(images)  # (B, C_fused, 16, 16)

            # Forward through probe
            logits = probe(features)  # (B, num_classes, 224, 224)

            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

    return probe


def evaluate_probe(encoder, probe, dataset, batch_size=32, device="cuda",
                   num_workers=4, num_classes=8):
    """Evaluate probe on test set, compute mIoU.

    Returns:
        dict with mIoU and per-class IoU
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    # Confusion matrix for mIoU computation
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    probe.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            masks = masks.to(device)

            features = encoder.extract_multilayer_spatial(images)
            logits = probe(features)
            preds = logits.argmax(dim=1)  # (B, H, W)

            # Update confusion matrix (vectorized)
            preds_cpu = preds.cpu()
            masks_cpu = masks.cpu()
            valid = masks_cpu != 255
            pred_valid = preds_cpu[valid].long()
            mask_valid = masks_cpu[valid].long()
            indices = mask_valid * num_classes + pred_valid
            confusion += torch.bincount(
                indices, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # Compute per-class IoU
    per_class_iou = {}
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
        per_class_iou[c] = iou

    # mIoU over 7 affordance classes (exclude background=0), matching Zhang et al.
    affordance_ious = [v for k, v in per_class_iou.items() if k != 0]
    miou = np.mean(affordance_ious) if affordance_ious else 0.0
    miou_all = np.mean(list(per_class_iou.values()))

    return {
        "mIoU": miou,
        "mIoU_all": miou_all,
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion.numpy(),
    }
