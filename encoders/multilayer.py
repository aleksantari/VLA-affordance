"""Multi-layer feature extraction and fusion following Zhang et al. / Probing3D protocol.

Extracts patch tokens from 4 equally-spaced intermediate encoder layers
and concatenates along the feature dimension.
"""

import torch
from typing import List, Tuple


def get_probe_layer_indices(num_layers: int, num_probes: int = 4) -> List[int]:
    """Get equally-spaced layer indices for probing (0-indexed).

    For SigLIP So400m (27 layers): [6, 13, 20, 26]
    For DINOv2 ViT-B (12 layers):  [2, 5, 8, 11]
    For DINOv2 ViT-L (24 layers):  [5, 11, 17, 23]
    """
    indices = [
        int(round((i + 1) * (num_layers - 1) / num_probes))
        for i in range(num_probes)
    ]
    return indices


def fuse_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    encoder_type: str,
    num_probes: int = 4,
) -> torch.Tensor:
    """Fuse multi-layer hidden states into a single feature tensor.

    Args:
        hidden_states: Tuple of (B, N, C) tensors from encoder.
            hidden_states[0] = embedding output (pre-transformer).
            hidden_states[i+1] = output after transformer layer i.
        encoder_type: "siglip" or "dinov2"
        num_probes: number of layers to extract (default 4)

    Returns:
        features: (B, C_fused, H_grid, W_grid) where C_fused = C * num_probes
    """
    # Subtract 1 for the embedding layer
    num_layers = len(hidden_states) - 1
    layer_indices = get_probe_layer_indices(num_layers, num_probes)

    selected = []
    for idx in layer_indices:
        # Layer i output is at hidden_states[i+1]
        layer_output = hidden_states[idx + 1]

        if encoder_type == "dinov2":
            # DINOv2: skip CLS token at index 0
            patch_tokens = layer_output[:, 1:, :]
        elif encoder_type == "siglip":
            # SigLIP: all tokens are patch tokens (no CLS)
            patch_tokens = layer_output
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        selected.append(patch_tokens)

    # Concatenate along feature dimension: (B, N_patches, C * num_probes)
    fused = torch.cat(selected, dim=-1)

    # Reshape to spatial grid: (B, N, C_fused) -> (B, C_fused, H, W)
    B, N, C = fused.shape
    H = W = int(N ** 0.5)
    assert H * W == N, f"Non-square patch grid: {N} patches"
    features = fused.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return features
