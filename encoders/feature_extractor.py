"""Unified feature extraction interface for all encoders."""

import torch
from . import raw_siglip, paligemma_siglip, pi0_siglip, pi05_siglip, dinov2, dino_wm
from .multilayer import fuse_hidden_states, get_probe_layer_indices


ENCODER_REGISTRY = {
    "raw_siglip": {
        "loader": raw_siglip.load_raw_siglip,
        "extractor": raw_siglip.extract_features,
        "hidden_states_extractor": raw_siglip.extract_hidden_states,
        "feature_dim": 1152,
        "fused_feature_dim": 4608,  # 1152 * 4
        "encoder_type": "siglip",
        "num_layers": 27,
    },
    "paligemma_siglip": {
        "loader": paligemma_siglip.load_paligemma_siglip,
        "extractor": paligemma_siglip.extract_features,
        "hidden_states_extractor": paligemma_siglip.extract_hidden_states,
        "feature_dim": 1152,
        "fused_feature_dim": 4608,
        "encoder_type": "siglip",
        "num_layers": 27,
    },
    "pi0_siglip": {
        "loader": pi0_siglip.load_pi0_siglip,
        "extractor": pi0_siglip.extract_features,
        "hidden_states_extractor": pi0_siglip.extract_hidden_states,
        "feature_dim": 1152,
        "fused_feature_dim": 4608,
        "encoder_type": "siglip",
        "num_layers": 27,
    },
    "pi05_siglip": {
        "loader": pi05_siglip.load_pi05_siglip,
        "extractor": pi05_siglip.extract_features,
        "hidden_states_extractor": pi05_siglip.extract_hidden_states,
        "feature_dim": 1152,
        "fused_feature_dim": 4608,
        "encoder_type": "siglip",
        "num_layers": 27,
    },
    "dinov2": {
        "loader": dinov2.load_dinov2,
        "extractor": dinov2.extract_features,
        "hidden_states_extractor": dinov2.extract_hidden_states,
        "feature_dim": 768,
        "fused_feature_dim": 3072,  # 768 * 4
        "encoder_type": "dinov2",
        "num_layers": 12,
    },
    "dino_wm": {
        "loader": dino_wm.load_dino_wm,
        "extractor": dino_wm.extract_ground_truth_features,
        "hidden_states_extractor": dino_wm.extract_ground_truth_hidden_states,
        "feature_dim": 768,
        "fused_feature_dim": 3072,
        "encoder_type": "dinov2",
        "num_layers": 12,
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

        registry = ENCODER_REGISTRY[encoder_name]
        self.feature_dim = registry["feature_dim"]
        self.fused_feature_dim = registry["fused_feature_dim"]
        self.encoder_type = registry["encoder_type"]
        self.num_layers = registry["num_layers"]

        # Load model
        if encoder_name == "dino_wm":
            self.components = registry["loader"](device=device, **kwargs)
            self.model = self.components["encoder"]
            self.processor = self.components["processor"]
        else:
            self.model, self.processor = registry["loader"](device=device)

        self._extract_fn = registry["extractor"]
        self._hidden_states_fn = registry["hidden_states_extractor"]

    def extract(self, images):
        """Extract patch features from images (single layer, final).

        Returns:
            Patch tokens of shape (B, 256, feature_dim)
        """
        if self.encoder_name == "dino_wm":
            return self._extract_fn(self.components, images, device=self.device)
        return self._extract_fn(self.model, self.processor, images, device=self.device)

    def extract_spatial(self, images):
        """Extract features reshaped to spatial grid (single layer).

        Returns:
            Features of shape (B, feature_dim, H_grid, W_grid) = (B, C, 16, 16)
        """
        patch_tokens = self.extract(images)  # (B, 256, C)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        features = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return features

    def extract_hidden_states(self, images):
        """Extract all hidden states from the encoder.

        Returns:
            Tuple of tensors, one per layer + embedding layer.
        """
        if self.encoder_name == "dino_wm":
            return self._hidden_states_fn(self.components, images, device=self.device)
        return self._hidden_states_fn(self.model, self.processor, images, device=self.device)

    def extract_multilayer(self, images):
        """Extract multi-layer fused patch features (4 equally-spaced layers).

        Returns:
            Patch tokens of shape (B, 256, fused_feature_dim)
        """
        hidden_states = self.extract_hidden_states(images)
        # fuse_hidden_states returns (B, C_fused, H, W)
        spatial = fuse_hidden_states(hidden_states, self.encoder_type)
        # Convert back to token format: (B, C_fused, H, W) -> (B, N, C_fused)
        B, C, H, W = spatial.shape
        tokens = spatial.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return tokens

    def extract_multilayer_spatial(self, images):
        """Extract multi-layer fused features in spatial grid format.

        Returns:
            Features of shape (B, fused_feature_dim, H_grid, W_grid) = (B, C*4, 16, 16)
        """
        hidden_states = self.extract_hidden_states(images)
        return fuse_hidden_states(hidden_states, self.encoder_type)

    def get_probe_layers(self):
        """Get the layer indices used for multi-layer probing."""
        return get_probe_layer_indices(self.num_layers)

    def get_raw_model(self):
        """Get the underlying model for weight analysis."""
        return self.model
