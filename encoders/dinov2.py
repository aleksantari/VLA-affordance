"""Load frozen DINOv2 encoder (geometric affordance ceiling)."""

import torch
from transformers import Dinov2Model, AutoImageProcessor


def load_dinov2(model_name="facebook/dinov2-base", device="cuda"):
    """Load DINOv2 ViT-B/14 encoder.

    This is the reference for what a self-supervised encoder achieves on affordance.
    Zhang et al. showed DINOv2 achieves 0.670 mIoU on UMD affordance segmentation.

    Architecture: ViT-B/14 with 768-dim features, patch size 14.
    For 224x224 input: 16x16 = 256 patch tokens + 1 CLS token.
    DINOv2 is trained with DINO loss (CLS self-distillation) + iBOT loss
    (patch-level masked image modeling), giving spatially rich patch features.
    """
    model = Dinov2Model.from_pretrained(model_name)
    model = model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = {"height": 224, "width": 224}

    return model, processor


def extract_features(model, processor, images, device="cuda"):
    """Extract patch-level features from DINOv2.

    IMPORTANT: Skip index 0 (CLS token). Use indices 1: for patch tokens.

    Returns:
        Patch tokens tensor of shape (B, 256, 768)
    """
    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        # DINOv2 returns CLS at index 0, patch tokens at indices 1:
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, 256, 768)

    return patch_tokens


def extract_hidden_states(model, processor, images, device="cuda"):
    """Extract all hidden states for multi-layer fusion.

    Returns:
        Tuple of (B, 257, 768) tensors (includes CLS token), one per layer
        + embedding layer. CLS token handling is done by fuse_hidden_states().
        For DINOv2 ViT-B: 13 states (1 embedding + 12 layers).
    """
    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)

    return outputs.hidden_states
