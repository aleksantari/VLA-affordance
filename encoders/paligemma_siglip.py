"""Extract SigLIP vision encoder from PaliGemma-3B pretrained checkpoint.

PaliGemma jointly trains SigLIP on 1B multimodal examples (captioning, VQA,
detection, segmentation) with a slow LR warm-up. SigLIP is fully unfrozen.
This gives us the intermediate point between raw SigLIP and pi0/pi0.5.

Requires HuggingFace gated access to google/paligemma-3b-pt-224.
"""

import torch


def load_paligemma_siglip(model_name="google/paligemma-3b-pt-224", device="cuda"):
    """Load SigLIP vision tower from PaliGemma-3B pretrained at 224x224.

    Loads the full PaliGemma model, extracts the vision tower,
    and frees the LLM + projector to save memory.
    """
    from transformers import PaliGemmaForConditionalGeneration, SiglipImageProcessor

    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
    vision_tower = model.vision_tower
    vision_tower = vision_tower.to(device).eval()

    processor = SiglipImageProcessor.from_pretrained(model_name)

    # Free LLM and projector
    del model
    torch.cuda.empty_cache()

    return vision_tower, processor


def extract_features(model, processor, images, device="cuda"):
    """Extract patch-level features from PaliGemma's SigLIP (final layer only).

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
        For SigLIP So400m: 28 states (1 embedding + 27 layers).
    """
    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)

    return outputs.hidden_states
