"""Load raw (pretrained) SigLIP vision encoder."""

import torch
from transformers import SiglipVisionModel, SiglipImageProcessor


def load_raw_siglip(model_name="google/siglip-so400m-patch14-224", device="cuda"):
    """Load the pretrained SigLIP-So400m/14 encoder before any VLA fine-tuning.

    Uses the 224-native checkpoint (not the 384 model) to avoid position
    embedding interpolation artifacts. All downstream encoders (PaliGemma,
    pi0, pi0.5) also run at 224x224, making this a fair comparison.

    Architecture: SiglipVisionTransformer with 27 encoder layers, hidden dim 1152,
    patch size 14x14, input resolution 224x224 -> 256 patch tokens (16x16 grid).
    """
    model = SiglipVisionModel.from_pretrained(model_name)
    model = model.to(device).eval()

    processor = SiglipImageProcessor.from_pretrained(model_name)

    return model, processor


def extract_features(model, processor, images, device="cuda"):
    """Extract patch-level features from raw SigLIP (final layer only).

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
        patch_tokens = outputs.last_hidden_state  # (B, 256, 1152)

    return patch_tokens


def extract_hidden_states(model, processor, images, device="cuda"):
    """Extract all hidden states for multi-layer fusion.

    Returns:
        Tuple of (B, 256, 1152) tensors, one per layer + embedding layer.
        hidden_states[0] = embedding output, hidden_states[i+1] = layer i output.
        For SigLIP So400m: 28 states (1 embedding + 27 layers).
    """
    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)

    return outputs.hidden_states
