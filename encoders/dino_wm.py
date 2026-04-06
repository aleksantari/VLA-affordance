"""Load DINO-WM and extract predicted world state features.

DINO-WM uses a frozen DINOv2 ViT-B/14 encoder to map RGB -> patch embeddings,
then a ViT-based transition model with causal attention predicts future states.

Repository: https://github.com/mazpie/dino-wm
"""

import torch
import sys
import os


def load_dino_wm(repo_path="./third_party/dino-wm", checkpoint_path=None, device="cuda"):
    """Load DINO-WM model components.

    Args:
        repo_path: path to cloned dino-wm repository
        checkpoint_path: path to trained checkpoint (e.g., PushT)
        device: torch device

    Returns:
        dict with 'encoder' (frozen DINOv2) and 'transition_model'
    """
    # Add dino-wm to path
    sys.path.insert(0, repo_path)

    # The frozen DINOv2 encoder (same as encoders/dinov2.py)
    from transformers import Dinov2Model, AutoImageProcessor
    encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
    encoder = encoder.to(device).eval()

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    processor.size = {"height": 224, "width": 224}

    components = {
        "encoder": encoder,
        "processor": processor,
        "transition_model": None,
    }

    # Load transition model if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Transition model loading depends on DINO-WM's specific architecture
        # This will need to be adapted based on the checkpoint structure
        components["checkpoint"] = checkpoint
        print(f"Loaded DINO-WM checkpoint from {checkpoint_path}")

    return components


def extract_ground_truth_features(components, images, device="cuda"):
    """Extract ground truth DINOv2 features (raw encoding of actual frame).

    Returns:
        Patch tokens tensor of shape (B, 256, 768)
    """
    encoder = components["encoder"]
    processor = components["processor"]

    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = encoder(**inputs, return_dict=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, 256, 768)

    return patch_tokens


def extract_predicted_features(components, current_frames, actions, device="cuda"):
    """Extract predicted world state features from DINO-WM's transition model.

    Pipeline:
    1. Encode current frame with frozen DINOv2 -> z_t
    2. Feed z_t + action into transition model -> z_{t+1}_predicted
    3. Return predicted features for probing

    Args:
        components: dict from load_dino_wm()
        current_frames: list of PIL images (current observation)
        actions: action tensor for transition prediction
        device: torch device

    Returns:
        Predicted patch tokens of shape (B, 256, 768)
    """
    if components["transition_model"] is None:
        raise ValueError(
            "Transition model not loaded. Provide a checkpoint_path to load_dino_wm()."
        )

    # Encode current frame
    z_t = extract_ground_truth_features(components, current_frames, device)

    # Pass through transition model to get predicted next state
    # Implementation depends on DINO-WM's specific API
    transition_model = components["transition_model"]

    with torch.no_grad():
        z_t1_predicted = transition_model(z_t, actions)

    return z_t1_predicted


def extract_ground_truth_hidden_states(components, images, device="cuda"):
    """Extract all hidden states from DINO-WM's frozen DINOv2 for multi-layer fusion.

    Returns:
        Tuple of (B, 257, 768) tensors (includes CLS token).
        For DINOv2 ViT-B: 13 states (1 embedding + 12 layers).
    """
    encoder = components["encoder"]
    processor = components["processor"]

    if not isinstance(images, list):
        images = [images]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = encoder(**inputs, return_dict=True, output_hidden_states=True)

    return outputs.hidden_states
