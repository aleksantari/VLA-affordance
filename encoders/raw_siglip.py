"""Load raw (pretrained) SigLIP vision encoder."""

import torch
from transformers import SiglipVisionModel, SiglipImageProcessor


def load_raw_siglip(model_name="google/siglip-so400m-patch14-384", device="cuda"):
    """Load the pretrained SigLIP-So400m/14 encoder before any VLA fine-tuning.

    This is the same SigLIP variant used inside PaliGemma, which pi0 builds on.
    Architecture: SiglipVisionTransformer with 27 encoder layers, hidden dim 1152,
    patch size 14x14, input resolution 224x224 -> 256 patch tokens (16x16 grid).
    """
    model = SiglipVisionModel.from_pretrained(model_name)
    model = model.to(device).eval()

    processor = SiglipImageProcessor.from_pretrained(model_name)

    # Override to 224x224 for consistent patch grid across all encoders
    processor.size = {"height": 224, "width": 224}

    return model, processor


def extract_features(model, processor, images, device="cuda"):
    """Extract patch-level features from raw SigLIP.

    Args:
        model: SiglipVisionModel
        processor: SiglipImageProcessor
        images: list of PIL images or single PIL image

    Returns:
        Patch tokens tensor of shape (B, 256, 1152)
    """
    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        # SigLIP returns all patch tokens (no separate CLS)
        patch_tokens = outputs.last_hidden_state  # (B, 256, 1152)

    return patch_tokens
