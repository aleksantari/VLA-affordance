"""Unified feature extraction interface for all encoders."""

import torch
from . import raw_siglip, pi0_siglip, pi05_siglip, dinov2, dino_wm


ENCODER_REGISTRY = {
    "raw_siglip": {
        "loader": raw_siglip.load_raw_siglip,
        "extractor": raw_siglip.extract_features,
        "feature_dim": 1152,
    },
    "pi0_siglip": {
        "loader": pi0_siglip.load_pi0_siglip,
        "extractor": pi0_siglip.extract_features,
        "feature_dim": 1152,
    },
    "pi05_siglip": {
        "loader": pi05_siglip.load_pi05_siglip,
        "extractor": pi05_siglip.extract_features,
        "feature_dim": 1152,
    },
    "dinov2": {
        "loader": dinov2.load_dinov2,
        "extractor": dinov2.extract_features,
        "feature_dim": 768,
    },
    "dino_wm": {
        "loader": dino_wm.load_dino_wm,
        "extractor": dino_wm.extract_ground_truth_features,
        "feature_dim": 768,
    },
}


class UnifiedFeatureExtractor:
    """Unified interface to load any encoder and extract patch features."""

    def __init__(self, encoder_name, device="cuda", **kwargs):
        if encoder_name not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder: {encoder_name}. "
                f"Available: {list(ENCODER_REGISTRY.keys())}"
            )

        self.encoder_name = encoder_name
        self.device = device
        self.feature_dim = ENCODER_REGISTRY[encoder_name]["feature_dim"]

        registry = ENCODER_REGISTRY[encoder_name]

        # Load model
        if encoder_name == "dino_wm":
            self.components = registry["loader"](device=device, **kwargs)
            self.model = self.components["encoder"]
            self.processor = self.components["processor"]
        else:
            self.model, self.processor = registry["loader"](device=device)

        self._extract_fn = registry["extractor"]

    def extract(self, images):
        """Extract patch features from images.

        Args:
            images: PIL image or list of PIL images

        Returns:
            Patch tokens of shape (B, 256, feature_dim)
        """
        if self.encoder_name == "dino_wm":
            return self._extract_fn(self.components, images, device=self.device)
        return self._extract_fn(self.model, self.processor, images, device=self.device)

    def extract_spatial(self, images):
        """Extract features reshaped to spatial grid.

        Returns:
            Features of shape (B, feature_dim, H_grid, W_grid) = (B, C, 16, 16)
        """
        patch_tokens = self.extract(images)  # (B, 256, C)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # 16
        features = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, 16, 16)
        return features

    def get_raw_model(self):
        """Get the underlying model for weight analysis."""
        return self.model
