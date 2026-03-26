"""Affordance segmentation linear probe following Zhang et al.'s protocol.

Probe architecture: BatchNorm + 1x1 Conv (single linear layer)
- Input: frozen patch features from encoder (upsampled to image resolution)
- Output: per-pixel affordance class prediction (7 categories + background)
- Loss: cross-entropy
- Metric: mIoU (mean Intersection over Union)
"""

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
        # features: (B, C, H_feat, W_feat) — patch features reshaped to spatial grid
        x = self.bn(features)
        x = self.conv(x)
        # Upsample to original image resolution for pixel-wise comparison
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode='bilinear', align_corners=False)
        return x


def train_probe(encoder, dataset, feature_dim, num_classes=8, epochs=50,
                lr=1e-3, batch_size=32, device="cuda", num_workers=4):
    """Train a linear probe on frozen encoder features.

    Args:
        encoder: UnifiedFeatureExtractor instance
        dataset: UMDAffordanceDataset (train split)
        feature_dim: dimension of encoder features
        num_classes: number of output classes
        epochs: training epochs
        lr: learning rate
        batch_size: batch size
        device: torch device
        num_workers: dataloader workers

    Returns:
        Trained AffordanceLinearProbe
    """
    probe = AffordanceLinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    probe.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            masks = masks.to(device)

            # Extract features (frozen encoder, no gradients)
            with torch.no_grad():
                features = encoder.extract_spatial(images)  # (B, C, 16, 16)

            # Forward through probe
            logits = probe(features)  # (B, num_classes, 224, 224)

            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

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

            features = encoder.extract_spatial(images)
            logits = probe(features)
            preds = logits.argmax(dim=1)  # (B, H, W)

            # Update confusion matrix
            for pred, mask in zip(preds.cpu(), masks.cpu()):
                valid = mask != 255
                pred_valid = pred[valid]
                mask_valid = mask[valid]
                for c_pred, c_true in zip(pred_valid.flatten(), mask_valid.flatten()):
                    confusion[c_true, c_pred] += 1

    # Compute per-class IoU
    per_class_iou = {}
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
        per_class_iou[c] = iou

    miou = np.mean(list(per_class_iou.values()))

    return {
        "mIoU": miou,
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion.numpy(),
    }
