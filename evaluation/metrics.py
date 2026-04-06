"""Evaluation metrics: mIoU computation, separation ratios."""

import numpy as np
import torch


def compute_miou(confusion_matrix, num_classes=8, ignore_classes=None):
    """Compute mean Intersection over Union from confusion matrix.

    Args:
        confusion_matrix: (num_classes, num_classes) array where
                         confusion[true, pred] = count
        num_classes: number of classes
        ignore_classes: list of class indices to exclude from mIoU average
                       (default: [0] to exclude background, matching Zhang et al.)

    Returns:
        dict with mIoU (over non-ignored classes), mIoU_all, and per-class IoU
    """
    if ignore_classes is None:
        ignore_classes = [0]

    per_class_iou = {}

    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        per_class_iou[c] = float(iou)

    all_ious = list(per_class_iou.values())
    affordance_ious = [v for k, v in per_class_iou.items() if k not in ignore_classes]

    return {
        "mIoU": float(np.mean(affordance_ious)) if affordance_ious else 0.0,
        "mIoU_all": float(np.mean(all_ious)) if all_ious else 0.0,
        "per_class_iou": per_class_iou,
    }


def compute_separation_ratio(projected_patches, part_masks):
    """Compute inter-part / intra-part variance ratio.

    Args:
        projected_patches: (H, W, n_components) PCA-projected features
        part_masks: dict mapping part_name -> (H, W) binary mask

    Returns:
        float: separation ratio (higher = better part separation)
    """
    part_centroids = {}
    part_variances = {}

    for part_name, mask in part_masks.items():
        features = projected_patches[mask.astype(bool)]
        if len(features) > 0:
            part_centroids[part_name] = features.mean(axis=0)
            part_variances[part_name] = features.var(axis=0).mean()

    parts = list(part_centroids.keys())
    inter_distances = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            dist = np.linalg.norm(part_centroids[parts[i]] - part_centroids[parts[j]])
            inter_distances.append(dist)

    avg_inter = np.mean(inter_distances) if inter_distances else 0
    avg_intra = np.mean(list(part_variances.values())) if part_variances else 1

    return float(avg_inter / (avg_intra + 1e-8))


def aggregate_results(all_results):
    """Aggregate results across multiple encoders into a comparison table.

    Args:
        all_results: dict mapping encoder_name -> result_dict

    Returns:
        Formatted comparison dict
    """
    comparison = {}
    for encoder_name, results in all_results.items():
        comparison[encoder_name] = {
            "mIoU": results.get("mIoU", None),
            "separation_ratio": results.get("separation_ratio", None),
            "correspondence_accuracy": results.get("hit_at_k", None),
        }
    return comparison
