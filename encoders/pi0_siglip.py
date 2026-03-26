"""Extract SigLIP vision encoder from pi0 base checkpoint."""

import torch


def load_pi0_siglip(model_name="lerobot/pi0_base", device="cuda"):
    """Load SigLIP encoder weights from the pi0 base model.

    The pi0 model uses PaliGemma internally. The vision encoder is at:
    model.paligemma_with_expert.paligemma.vision_tower (SiglipVisionModel)

    Uses the LeRobot HuggingFace PyTorch port (no JAX dependency).
    """
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    policy = PI0Policy.from_pretrained(model_name)

    # Extract the vision tower from the PaliGemma backbone
    vision_tower = policy.model.paligemma_with_expert.paligemma.vision_tower
    vision_tower = vision_tower.to(device).eval()

    # Get the processor for image preprocessing
    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    processor.size = {"height": 224, "width": 224}

    return vision_tower, processor


def extract_features(model, processor, images, device="cuda"):
    """Extract patch-level features from pi0's SigLIP encoder.

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
