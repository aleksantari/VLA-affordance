"""Extract SigLIP vision encoder from pi0.5 base checkpoint."""

import torch


def load_pi05_siglip(model_name="lerobot/pi05_base", device="cuda"):
    """Load SigLIP encoder weights from the pi0.5 base model.

    Same as pi0 but from pi0.5, which adds hierarchical reasoning with
    discrete subtask prediction. The hypothesis is that pi0.5's stronger
    language-grounded spatial reasoning demands may push the encoder harder.
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(model_name)

    # Extract the vision tower from the PaliGemma backbone
    vision_tower = policy.model.paligemma_with_expert.paligemma.vision_tower
    vision_tower = vision_tower.to(device).eval()

    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    processor.size = {"height": 224, "width": 224}

    return vision_tower, processor


def extract_features(model, processor, images, device="cuda"):
    """Extract patch-level features from pi0.5's SigLIP encoder.

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
