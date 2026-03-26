"""PCA subspace projection analysis following Zhang et al.

This is Zhang et al.'s primary visual analysis tool.
Protocol:
1. Select a reference object from the dataset
2. Extract patch features from the encoder for the reference image
3. Run PCA on patch features -> principal components defining a geometric subspace
4. Project patch features from NOVEL scenes into this subspace
5. Visualize: color each patch by its PCA projection coordinates
"""

import torch
import numpy as np
from sklearn.decomposition import PCA


def extract_patch_features(encoder, image):
    """Extract spatial patch features from an encoder.

    Returns: (B, C, H_grid, W_grid) tensor of patch features arranged spatially.
    """
    return encoder.extract_spatial(image)


def pca_subspace_analysis(encoder, reference_image, test_images, n_components=3):
    """Compute PCA on reference object's patch features, then project test images.

    Args:
        encoder: UnifiedFeatureExtractor instance
        reference_image: PIL image of reference object (e.g., a mug)
        test_images: list of PIL images to project
        n_components: number of PCA components

    Returns:
        results: list of (16, 16, n_components) arrays for each test image
        pca: fitted PCA object
        ref_projected: (16, 16, n_components) array for reference image
    """
    # 1. Extract features from reference object
    ref_features = extract_patch_features(encoder, reference_image)
    # ref_features: (1, C, H, W)
    ref_flat = ref_features.squeeze(0).reshape(ref_features.shape[1], -1).T.cpu().numpy()
    # ref_flat: (N_patches, C)

    # 2. Fit PCA on reference features
    pca = PCA(n_components=n_components)
    pca.fit(ref_flat)

    # Project reference image
    ref_projected = pca.transform(ref_flat)  # (N_patches, n_components)
    ref_projected = ref_projected.reshape(16, 16, n_components)

    # 3. Project test images
    results = []
    for test_img in test_images:
        test_features = extract_patch_features(encoder, test_img)
        test_flat = test_features.squeeze(0).reshape(test_features.shape[1], -1).T.cpu().numpy()
        projected = pca.transform(test_flat)  # (N_patches, n_components)
        results.append(projected.reshape(16, 16, n_components))

    return results, pca, ref_projected


def quantify_part_separation(projected_patches, part_masks):
    """Measure how well PCA separates functional parts.

    Args:
        projected_patches: (16, 16, n_components) array
        part_masks: dict mapping part_name -> binary mask over the 16x16 patch grid

    Returns:
        Separation ratio (inter-part distance / intra-part variance).
        Higher = better separation.
    """
    part_centroids = {}
    part_variances = {}

    for part_name, mask in part_masks.items():
        part_features = projected_patches[mask]
        if len(part_features) > 0:
            part_centroids[part_name] = part_features.mean(axis=0)
            part_variances[part_name] = part_features.var(axis=0).mean()

    # Inter-part distance: average pairwise distance between centroids
    parts = list(part_centroids.keys())
    inter_distances = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            dist = np.linalg.norm(part_centroids[parts[i]] - part_centroids[parts[j]])
            inter_distances.append(dist)

    avg_inter = np.mean(inter_distances) if inter_distances else 0
    avg_intra = np.mean(list(part_variances.values())) if part_variances else 1

    return avg_inter / (avg_intra + 1e-8)  # Separation ratio
