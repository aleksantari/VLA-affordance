"""Cosine similarity for part correspondence analysis.

Protocol (from Zhang et al.):
1. Select a patch from a functional part of one object (e.g., handle of mug A)
2. Compute cosine similarity between this query patch and all patches of another object
3. Visualize: heatmap showing which regions are most similar to the query patch
4. Measure: does the query patch activate the functionally corresponding part?
"""

import torch
import numpy as np


def cosine_similarity_map(encoder, query_image, query_patch_idx, target_image):
    """Compute cosine similarity between a query patch and all patches of a target image.

    Args:
        encoder: UnifiedFeatureExtractor instance
        query_image: PIL image containing the query object
        query_patch_idx: (row, col) in the 16x16 patch grid
        target_image: PIL image containing the target object

    Returns:
        (16, 16) similarity heatmap
    """
    query_features = encoder.extract_spatial(query_image)
    target_features = encoder.extract_spatial(target_image)

    # Extract query patch feature
    row, col = query_patch_idx
    query_vec = query_features[0, :, row, col]  # (C,)

    # Compute cosine similarity with all target patches
    target_flat = target_features[0]  # (C, H, W)
    C, H, W = target_flat.shape
    target_flat = target_flat.reshape(C, -1)  # (C, H*W)

    query_norm = query_vec / (query_vec.norm() + 1e-8)
    target_norm = target_flat / (target_flat.norm(dim=0, keepdim=True) + 1e-8)

    similarity = (query_norm.unsqueeze(1) * target_norm).sum(dim=0)  # (H*W,)
    similarity_map = similarity.reshape(H, W)

    return similarity_map.cpu().numpy()


def compute_correspondence_accuracy(encoder, query_image, query_patch_idx,
                                     target_image, target_part_mask):
    """Measure if a query patch activates the correct corresponding part.

    Args:
        encoder: UnifiedFeatureExtractor instance
        query_image: PIL image of query object
        query_patch_idx: (row, col) of a specific functional part
        target_image: PIL image of target object
        target_part_mask: (16, 16) binary mask of the corresponding part in target

    Returns:
        dict with accuracy metrics
    """
    sim_map = cosine_similarity_map(encoder, query_image, query_patch_idx, target_image)

    # Metrics
    # 1. Is the max-similarity patch inside the target part?
    max_idx = np.unravel_index(sim_map.argmax(), sim_map.shape)
    hit_at_1 = bool(target_part_mask[max_idx[0], max_idx[1]])

    # 2. What fraction of top-k patches are inside the target part?
    flat_sim = sim_map.flatten()
    flat_mask = target_part_mask.flatten().astype(bool)
    top_k = max(int(flat_mask.sum()), 1)  # use part size as k
    top_k_indices = np.argsort(flat_sim)[-top_k:]
    hit_at_k = flat_mask[top_k_indices].mean()

    # 3. Average similarity inside vs outside the part
    inside_sim = sim_map[target_part_mask.astype(bool)].mean() if target_part_mask.any() else 0
    outside_sim = sim_map[~target_part_mask.astype(bool)].mean() if (~target_part_mask.astype(bool)).any() else 0

    return {
        "hit_at_1": hit_at_1,
        "hit_at_k": float(hit_at_k),
        "inside_similarity": float(inside_sim),
        "outside_similarity": float(outside_sim),
        "similarity_ratio": float(inside_sim / (outside_sim + 1e-8)),
        "similarity_map": sim_map,
    }
