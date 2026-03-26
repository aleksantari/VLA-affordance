"""Step 1: Verify all encoder checkpoints load and produce correct output shapes.

Tests both single-layer and multi-layer (fused) extraction for each encoder.
Prints layer counts, probe indices, and fused dimensions.

Encoders:
- Raw SigLIP (google/siglip-so400m-patch14-224)
- PaliGemma SigLIP (google/paligemma-3b-pt-224)
- pi0 SigLIP (lerobot/pi0_base)
- pi0.5 SigLIP (lerobot/pi05_base)
- DINOv2 (facebook/dinov2-base)
- DINO-WM (frozen DINOv2 + transition model)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
import numpy as np

from encoders.feature_extractor import UnifiedFeatureExtractor, ENCODER_REGISTRY
from encoders.multilayer import get_probe_layer_indices


def create_test_image():
    """Create a dummy 224x224 test image."""
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def verify_encoder(encoder_name, device="cuda"):
    """Verify single-layer and multi-layer extraction for one encoder."""
    registry = ENCODER_REGISTRY[encoder_name]
    feature_dim = registry["feature_dim"]
    fused_dim = registry["fused_feature_dim"]
    encoder_type = registry["encoder_type"]
    num_layers = registry["num_layers"]
    probe_layers = get_probe_layer_indices(num_layers)

    print(f"\n  Type: {encoder_type} | Layers: {num_layers} | Probe layers: {probe_layers}")
    print(f"  Single-layer dim: {feature_dim} | Fused dim: {fused_dim}")

    try:
        extractor = UnifiedFeatureExtractor(encoder_name, device=device)
    except Exception as e:
        print(f"  SKIP (load failed): {e}")
        return False

    test_img = create_test_image()

    # Test single-layer extraction
    features = extractor.extract(test_img)
    assert features.shape == (1, 256, feature_dim), \
        f"Single-layer: expected (1, 256, {feature_dim}), got {features.shape}"
    print(f"  Single-layer: {features.shape} PASS")

    # Test multi-layer hidden states
    hidden_states = extractor.extract_hidden_states(test_img)
    expected_states = num_layers + 1  # layers + embedding
    assert len(hidden_states) == expected_states, \
        f"Hidden states: expected {expected_states}, got {len(hidden_states)}"
    print(f"  Hidden states: {len(hidden_states)} states PASS")

    # Test multi-layer fused extraction
    fused = extractor.extract_multilayer(test_img)
    assert fused.shape == (1, 256, fused_dim), \
        f"Fused: expected (1, 256, {fused_dim}), got {fused.shape}"
    print(f"  Fused tokens: {fused.shape} PASS")

    # Test spatial fused extraction
    spatial = extractor.extract_multilayer_spatial(test_img)
    assert spatial.shape == (1, fused_dim, 16, 16), \
        f"Spatial: expected (1, {fused_dim}, 16, 16), got {spatial.shape}"
    print(f"  Fused spatial: {spatial.shape} PASS")

    del extractor
    torch.cuda.empty_cache()
    return True


# Encoders that don't require special access or installs
CORE_ENCODERS = ["raw_siglip", "dinov2"]
# Encoders that may need gated access or extra dependencies
OPTIONAL_ENCODERS = ["paligemma_siglip", "pi0_siglip", "pi05_siglip", "dino_wm"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    passed = 0
    skipped = 0
    total = len(CORE_ENCODERS) + len(OPTIONAL_ENCODERS)

    for i, name in enumerate(CORE_ENCODERS + OPTIONAL_ENCODERS, 1):
        print(f"\n[{i}/{total}] Setting up {name}...")
        if verify_encoder(name, device):
            passed += 1
        else:
            skipped += 1

    print(f"\n{'='*50}")
    print(f"Encoder setup complete: {passed} passed, {skipped} skipped")
    print(f"{'='*50}")
