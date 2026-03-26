"""Depth and surface normal extraction for augmentation experiments.

Uses DPT (Dense Prediction Transformer) from HuggingFace for monocular depth
estimation, then derives surface normals from depth via finite differences.

The depth/normal augmentation experiment measures how much geometric
information the encoder already has vs needs from external 3D cues.
Delta = mIoU(with_depth) - mIoU(without_depth). Small delta = encoder
already encodes strong geometry.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def load_depth_model(model_name="Intel/dpt-large", device="cuda"):
    """Load DPT depth estimation model.

    Returns:
        model: DPTForDepthEstimation
        processor: DPTImageProcessor
    """
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    model = DPTForDepthEstimation.from_pretrained(model_name)
    model = model.to(device).eval()
    processor = DPTImageProcessor.from_pretrained(model_name)

    return model, processor


def compute_normals_from_depth(depth_map):
    """Compute surface normals from a depth map using finite differences.

    Args:
        depth_map: (H, W) numpy array of depth values

    Returns:
        normals: (H, W, 3) numpy array of unit surface normals (x, y, z)
    """
    # Finite differences for gradient
    dz_dy = np.gradient(depth_map, axis=0)  # vertical gradient
    dz_dx = np.gradient(depth_map, axis=1)  # horizontal gradient

    # Normal vector: (-dz/dx, -dz/dy, 1), then normalize
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth_map)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)

    return normals


def extract_depth_normal_features(
    image,
    depth_model,
    depth_processor,
    patch_grid_size=16,
    device="cuda",
):
    """Extract depth + surface normal features downsampled to patch grid.

    Args:
        image: PIL Image
        depth_model: DPTForDepthEstimation
        depth_processor: DPTImageProcessor
        patch_grid_size: target spatial resolution (16 for 16x16 patch grid)
        device: torch device

    Returns:
        features: (1, 4, patch_grid_size, patch_grid_size) tensor
            Channel 0: normalized depth
            Channels 1-3: surface normal (x, y, z)
    """
    # Run DPT depth estimation
    inputs = depth_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, H, W)

    # Interpolate to a consistent resolution for normal computation
    depth_map = F.interpolate(
        predicted_depth.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()  # (224, 224)

    # Normalize depth to [0, 1]
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 1e-8:
        depth_normalized = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_normalized = np.zeros_like(depth_map)

    # Compute surface normals from depth
    normals = compute_normals_from_depth(depth_normalized)  # (224, 224, 3)

    # Stack depth + normals: (224, 224, 4)
    depth_normal = np.concatenate(
        [depth_normalized[:, :, np.newaxis], normals], axis=-1
    )

    # Convert to tensor and downsample to patch grid
    depth_normal_tensor = torch.from_numpy(depth_normal).float()
    depth_normal_tensor = depth_normal_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 4, 224, 224)
    depth_normal_tensor = F.adaptive_avg_pool2d(
        depth_normal_tensor, (patch_grid_size, patch_grid_size)
    )  # (1, 4, 16, 16)

    return depth_normal_tensor
