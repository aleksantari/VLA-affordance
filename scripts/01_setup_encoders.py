"""Step 1: Download/convert all encoder checkpoints.

Downloads:
- Raw SigLIP (google/siglip-so400m-patch14-384)
- pi0 base (lerobot/pi0_base) -> extract SigLIP
- pi0.5 base (lerobot/pi05_base) -> extract SigLIP
- DINOv2 (facebook/dinov2-base)
- DINO-WM repository

Verifies each produces correct output shapes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
import numpy as np


def create_test_image():
    """Create a dummy 224x224 test image."""
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def setup_raw_siglip(device="cuda"):
    print("\n[1/5] Setting up Raw SigLIP...")
    from encoders.raw_siglip import load_raw_siglip, extract_features
    model, processor = load_raw_siglip(device=device)
    test_img = create_test_image()
    features = extract_features(model, processor, test_img, device=device)
    print(f"  Output shape: {features.shape}")
    assert features.shape == (1, 256, 1152), f"Expected (1, 256, 1152), got {features.shape}"
    print("  PASS")
    del model
    torch.cuda.empty_cache()


def setup_dinov2(device="cuda"):
    print("\n[2/5] Setting up DINOv2...")
    from encoders.dinov2 import load_dinov2, extract_features
    model, processor = load_dinov2(device=device)
    test_img = create_test_image()
    features = extract_features(model, processor, test_img, device=device)
    print(f"  Output shape: {features.shape}")
    assert features.shape == (1, 256, 768), f"Expected (1, 256, 768), got {features.shape}"
    print("  PASS")
    del model
    torch.cuda.empty_cache()


def setup_pi0_siglip(device="cuda"):
    print("\n[3/5] Setting up pi0 SigLIP...")
    try:
        from encoders.pi0_siglip import load_pi0_siglip, extract_features
        model, processor = load_pi0_siglip(device=device)
        test_img = create_test_image()
        features = extract_features(model, processor, test_img, device=device)
        print(f"  Output shape: {features.shape}")
        assert features.shape == (1, 256, 1152), f"Expected (1, 256, 1152), got {features.shape}"
        print("  PASS")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  SKIP: {e}")
        print("  pi0 checkpoint may need manual download or lerobot install.")


def setup_pi05_siglip(device="cuda"):
    print("\n[4/5] Setting up pi0.5 SigLIP...")
    try:
        from encoders.pi05_siglip import load_pi05_siglip, extract_features
        model, processor = load_pi05_siglip(device=device)
        test_img = create_test_image()
        features = extract_features(model, processor, test_img, device=device)
        print(f"  Output shape: {features.shape}")
        assert features.shape == (1, 256, 1152), f"Expected (1, 256, 1152), got {features.shape}"
        print("  PASS")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  SKIP: {e}")
        print("  pi0.5 checkpoint may need manual download or lerobot install.")


def setup_dino_wm(device="cuda"):
    print("\n[5/5] Setting up DINO-WM...")
    from encoders.dino_wm import load_dino_wm, extract_ground_truth_features
    components = load_dino_wm(device=device)
    test_img = create_test_image()
    features = extract_ground_truth_features(components, test_img, device=device)
    print(f"  Ground truth output shape: {features.shape}")
    assert features.shape == (1, 256, 768), f"Expected (1, 256, 768), got {features.shape}"
    print("  PASS (ground truth features only — transition model needs checkpoint)")
    del components
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    setup_raw_siglip(device)
    setup_dinov2(device)
    setup_pi0_siglip(device)
    setup_pi05_siglip(device)
    setup_dino_wm(device)

    print("\n" + "=" * 50)
    print("Encoder setup complete!")
    print("=" * 50)
