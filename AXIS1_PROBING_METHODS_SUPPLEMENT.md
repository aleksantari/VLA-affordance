# Axis 1 probing methods — supplementary implementation guide

This document supplements `AXIS1_IMPLEMENTATION_GUIDE.md` with corrections, clarifications, and detailed implementation guidance for the three probing methods we are using. Read the original guide first for encoder loading, dataset setup, weight extraction, and project structure. This document focuses exclusively on the probing methodology.

---

## Context: what changed from the original guide

After reviewing Zhang et al. (arXiv 2602.20501) in detail and discussing which methods are necessary for our specific thesis, we made two changes:

1. **Dropped cosine similarity correspondence (Method 3 in original guide).** It is redundant with PCA analysis for our thesis — if PCA shows SigLIP doesn't decompose objects into parts, cosine similarity will show the same. It can be added later as a stretch goal but is not necessary.

2. **Identified a critical methodological gap: multi-layer feature fusion.** Zhang et al. do NOT probe only the final encoder layer. They extract features from four equally-spaced intermediate layers and fuse them. The original guide's `extract_patch_features` function uses only `last_hidden_state`, which would systematically undercount geometric information in intermediate layers. This must be fixed.

We converge on three methods, each answering a distinct question:

| Method | Question it answers | Output type |
|--------|-------------------|-------------|
| Linear probing on UMD | How much geometric affordance is in these features? | Quantitative (mIoU number) |
| PCA subspace projection | What kind of geometric structure does the encoder learn? | Qualitative (visualizations) + quantitative (separation ratio) |
| Depth/normal augmentation delta | How much geometric information does the encoder already have vs. need externally? | Quantitative (mIoU delta) |

---

## Critical fix: multi-layer feature extraction

### The problem

The original guide extracts features only from the final encoder layer:

```python
# WRONG — only final layer
outputs = encoder(image, return_dict=True)
patch_tokens = outputs.last_hidden_state  # Only layer 27 of 27
```

Zhang et al. extract from four equally-spaced layers and fuse them. This follows the Probing3D protocol (El Banani et al.). The rationale: different layers capture different levels of spatial information. Early layers retain more local geometric detail; later layers are more semantic. Fusing gives the probe the best chance to extract whatever geometric information exists.

### The fix

For every encoder, extract patch tokens from 4 equally-spaced layers and concatenate along the feature dimension.

```python
import torch
from typing import List

def get_probe_layer_indices(num_layers: int, num_probes: int = 4) -> List[int]:
    """
    Get equally-spaced layer indices for probing.
    Following Zhang et al. / Probing3D protocol.
    
    For SigLIP So400m (27 layers): layers [6, 13, 20, 26] (0-indexed)
    For DINOv2 ViT-B (12 layers):  layers [2, 5, 8, 11] (0-indexed)
    For DINOv2 ViT-L (24 layers):  layers [5, 11, 17, 23] (0-indexed)
    """
    # Equally spaced, ending at the last layer
    indices = [
        int(round((i + 1) * (num_layers - 1) / num_probes))
        for i in range(num_probes)
    ]
    return indices


def extract_multilayer_features(
    encoder,
    image: torch.Tensor,
    encoder_type: str,
    num_probes: int = 4,
) -> torch.Tensor:
    """
    Extract patch features from multiple equally-spaced encoder layers
    and concatenate them, following Zhang et al.'s probing protocol.
    
    Args:
        encoder: the vision encoder model
        image: input tensor (B, 3, 224, 224)
        encoder_type: "dinov2" or "siglip"
        num_probes: number of intermediate layers to extract (default 4)
    
    Returns:
        features: (B, C_fused, H_grid, W_grid) where C_fused = C * num_probes
        
    For SigLIP So400m: C_fused = 1152 * 4 = 4608
    For DINOv2 ViT-B:  C_fused = 768 * 4 = 3072
    """
    with torch.no_grad():
        # Run encoder with output_hidden_states=True to get all intermediate layers
        outputs = encoder(image, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (B, N, C) per layer
        # hidden_states[0] = embedding output (pre-transformer)
        # hidden_states[1] = after layer 0
        # hidden_states[L] = after layer L-1 = final output
        
        if encoder_type == "dinov2":
            num_layers = len(hidden_states) - 1  # subtract embedding layer
            layer_indices = get_probe_layer_indices(num_layers, num_probes)
            
            selected = []
            for idx in layer_indices:
                # hidden_states is 0-indexed with embedding at position 0
                # So layer i output is at hidden_states[i+1]
                layer_output = hidden_states[idx + 1]
                # DINOv2: skip CLS token at index 0
                patch_tokens = layer_output[:, 1:, :]  # (B, 256, 768)
                selected.append(patch_tokens)
                
        elif encoder_type == "siglip":
            num_layers = len(hidden_states) - 1
            layer_indices = get_probe_layer_indices(num_layers, num_probes)
            
            selected = []
            for idx in layer_indices:
                layer_output = hidden_states[idx + 1]
                # SigLIP: all tokens are patch tokens (no CLS)
                patch_tokens = layer_output  # (B, 256, 1152)
                selected.append(patch_tokens)
        
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Concatenate along feature dimension
        fused = torch.cat(selected, dim=-1)  # (B, 256, C * num_probes)
        
        # Reshape to spatial grid
        B, N, C = fused.shape
        H = W = int(N ** 0.5)  # 16 for 256 patches
        features = fused.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
    
    return features
```

### Important: verify `output_hidden_states` works for each encoder

Before running any probing, confirm that `output_hidden_states=True` returns the expected number of layers:

```python
# Verification script — run this first for each encoder
def verify_hidden_states(encoder, encoder_type, image):
    outputs = encoder(image, return_dict=True, output_hidden_states=True)
    hs = outputs.hidden_states
    print(f"{encoder_type}: {len(hs)} hidden states (expect num_layers + 1)")
    print(f"  Each shape: {hs[0].shape}")
    print(f"  Probe layers: {get_probe_layer_indices(len(hs) - 1)}")
    
    # For SigLIP So400m: expect 28 states (embedding + 27 layers)
    # For DINOv2 ViT-B: expect 13 states (embedding + 12 layers)
```

### Feature dimensions after fusion

| Encoder | Layers | Hidden dim | Fused dim (4 layers) | Probe layers (0-indexed) |
|---------|--------|-----------|---------------------|--------------------------|
| SigLIP So400m/14 | 27 | 1152 | 4608 | [6, 13, 20, 26] |
| DINOv2 ViT-B/14 | 12 | 768 | 3072 | [2, 5, 8, 11] |
| DINOv2 ViT-L/14 | 24 | 1024 | 4096 | [5, 11, 17, 23] |

The linear probe input dimension must match the fused dimension, not the single-layer dimension. Each encoder gets its own probe with the correct input size.

---

## Method 1: linear probing for affordance segmentation

### What it measures

A single mIoU number per encoder that directly answers "how much geometric affordance information is extractable from these features?" This is the primary quantitative result — the backbone of the paper.

### Protocol (following Zhang et al. exactly)

1. Freeze the encoder completely — no gradient flow to encoder parameters.
2. Extract multi-layer fused patch features for every image in UMD train and test sets.
3. Train a single linear head (BatchNorm2d + Conv2d 1x1) on the train set.
4. Evaluate mIoU on the test set across 7 affordance categories + background.

### Probe architecture

```python
import torch
import torch.nn as nn

class AffordanceLinearProbe(nn.Module):
    """
    Linear probe following Zhang et al.'s protocol.
    BatchNorm + 1x1 Conv, trained on frozen features.
    """
    def __init__(self, feature_dim: int, num_classes: int = 8):
        """
        Args:
            feature_dim: fused feature dimension (e.g. 4608 for SigLIP, 3072 for DINOv2-B)
            num_classes: 7 affordance categories + 1 background = 8
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(feature_dim)
        self.conv = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
    
    def forward(self, features: torch.Tensor, target_size: tuple = (224, 224)) -> torch.Tensor:
        """
        Args:
            features: (B, C_fused, H_grid, W_grid) e.g. (B, 4608, 16, 16)
            target_size: spatial resolution to upsample to for pixel-wise loss
        Returns:
            logits: (B, num_classes, target_H, target_W)
        """
        x = self.bn(features)
        x = self.conv(x)
        x = nn.functional.interpolate(
            x, size=target_size, mode='bilinear', align_corners=False
        )
        return x
```

### Training configuration

```python
# Hyperparameters — match Zhang et al. / Probing3D protocol
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)  # 255 = unlabeled pixels if any
num_epochs = 50
batch_size = 32  # adjust for GPU memory with fused features
```

### Evaluation

```python
def compute_miou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 8) -> float:
    """
    Compute mean Intersection over Union.
    
    Args:
        predictions: (N, H, W) integer class predictions
        targets: (N, H, W) integer ground truth labels
        num_classes: number of classes including background
    Returns:
        mIoU: mean IoU across all classes present in ground truth
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        gt_mask = (targets == cls)
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return sum(ious) / len(ious) if ious else 0.0
```

### Validation target

DINOv2 should reproduce approximately 0.670 mIoU on UMD (Zhang et al.'s reported number). If your DINOv2 result is significantly different, debug the probing infrastructure before running other encoders. Common issues: wrong layer indices, forgetting to skip CLS token, incorrect image preprocessing, wrong number of classes.

### Systems to run

Run this for all 5 encoder states, producing a table like:

| System | mIoU | Delta from raw SigLIP |
|--------|------|----------------------|
| Raw SigLIP | ? | baseline |
| π0 SigLIP | ? | ? |
| π0.5 SigLIP | ? | ? |
| Frozen DINOv2 | ~0.670 (validate) | — |
| DINO-WM predicted states | ? | — |

---

## Method 2: PCA subspace projection

### What it measures

Qualitatively: what kind of geometric structure does each encoder learn? Does it decompose objects into functional parts (handle, blade, rim) or just do foreground/background separation?

Quantitatively: inter-part separation ratio — how cleanly do functional parts cluster in the principal component space.

### Protocol (following Zhang et al.)

1. Select reference manipulation objects from UMD (mug, knife, screwdriver — objects with clearly distinct functional parts).
2. Extract multi-layer fused patch features for the reference object.
3. Compute PCA on the reference object's patch features (N_patches x C_fused) → extract top 3 principal components.
4. Project patch features from novel scenes (different objects, different categories) into the same PCA subspace.
5. Visualize: color each patch in the 16x16 grid by its (PC1, PC2, PC3) coordinates mapped to RGB.
6. Quantify: compute inter-part distance / intra-part variance ratio using ground-truth part annotations.

### Implementation

```python
from sklearn.decomposition import PCA
import numpy as np

def compute_pca_subspace(
    encoder,
    reference_image: torch.Tensor,
    encoder_type: str,
    n_components: int = 3,
) -> PCA:
    """
    Fit PCA on a reference object's patch features.
    Returns the fitted PCA model for projecting other images.
    """
    features = extract_multilayer_features(encoder, reference_image, encoder_type)
    # features: (1, C_fused, 16, 16)
    flat = features.squeeze(0).reshape(features.shape[1], -1).T.cpu().numpy()
    # flat: (256, C_fused)
    
    pca = PCA(n_components=n_components)
    pca.fit(flat)
    return pca


def project_into_subspace(
    encoder,
    image: torch.Tensor,
    encoder_type: str,
    pca: PCA,
) -> np.ndarray:
    """
    Project an image's patch features into a fitted PCA subspace.
    Returns: (16, 16, n_components) array for visualization.
    """
    features = extract_multilayer_features(encoder, image, encoder_type)
    flat = features.squeeze(0).reshape(features.shape[1], -1).T.cpu().numpy()
    projected = pca.transform(flat)  # (256, n_components)
    return projected.reshape(16, 16, pca.n_components_)


def visualize_pca_rgb(projected: np.ndarray) -> np.ndarray:
    """
    Map 3-component PCA projection to RGB image for visualization.
    Normalizes each component to [0, 1] independently.
    Returns: (16, 16, 3) RGB array ready for matplotlib imshow.
    """
    rgb = projected.copy()
    for c in range(3):
        channel = rgb[:, :, c]
        cmin, cmax = channel.min(), channel.max()
        if cmax - cmin > 1e-8:
            rgb[:, :, c] = (channel - cmin) / (cmax - cmin)
        else:
            rgb[:, :, c] = 0.5
    return rgb


def quantify_part_separation(
    projected: np.ndarray,
    part_masks: dict,
) -> float:
    """
    Measure how well PCA separates functional parts.
    
    Args:
        projected: (16, 16, n_components) PCA-projected patch features
        part_masks: dict mapping part_name -> (16, 16) boolean mask
        
    Returns:
        separation_ratio: inter-part distance / intra-part variance
        Higher = cleaner part separation.
    """
    flat = projected.reshape(-1, projected.shape[-1])  # (256, n_components)
    
    centroids = {}
    variances = {}
    
    for part_name, mask in part_masks.items():
        mask_flat = mask.reshape(-1)  # (256,)
        part_features = flat[mask_flat]
        if len(part_features) > 1:
            centroids[part_name] = part_features.mean(axis=0)
            variances[part_name] = part_features.var(axis=0).mean()
    
    if len(centroids) < 2:
        return 0.0
    
    # Inter-part: average pairwise L2 distance between centroids
    parts = list(centroids.keys())
    inter_dists = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            dist = np.linalg.norm(centroids[parts[i]] - centroids[parts[j]])
            inter_dists.append(dist)
    
    avg_inter = np.mean(inter_dists)
    avg_intra = np.mean(list(variances.values()))
    
    return avg_inter / (avg_intra + 1e-8)
```

### What to look for in the visualizations

For each encoder, produce a grid of PCA RGB images across several UMD objects. Expected patterns:

- **DINOv2**: clean part-level color separation. Handle patches are one color, blade patches another, body patches a third. This separation should generalize across different objects.
- **Raw SigLIP**: noisy blobs, foreground/background separation only. No clean part decomposition.
- **π0 SigLIP**: the key question. Does it shift toward DINOv2-like part structure, or just get slightly better foreground detection?
- **π0.5 SigLIP**: does knowledge insulation produce different geometric structure than π0's naive fine-tuning?

### Part masks for quantification

The UMD dataset provides pixel-level affordance annotations. To create 16x16 patch-level part masks, downsample the affordance masks to the patch grid resolution:

```python
def create_patch_part_masks(
    affordance_mask: np.ndarray,
    patch_grid_size: int = 16,
) -> dict:
    """
    Convert pixel-level UMD affordance annotations to patch-level masks.
    
    Args:
        affordance_mask: (H, W) integer array with affordance category per pixel
        patch_grid_size: spatial resolution of patch grid (16 for 224px / 14px patches)
    
    Returns:
        dict mapping affordance_name -> (16, 16) boolean mask
        A patch belongs to a part if the majority of its pixels have that label.
    """
    from scipy.ndimage import zoom
    
    categories = {
        0: "background", 1: "grasp", 2: "cut", 3: "scoop",
        4: "contain", 5: "pound", 6: "support", 7: "wrap-grasp"
    }
    
    H, W = affordance_mask.shape
    patch_h = H // patch_grid_size
    patch_w = W // patch_grid_size
    
    masks = {}
    for cat_id, cat_name in categories.items():
        if cat_id == 0:
            continue  # skip background for part separation analysis
        patch_mask = np.zeros((patch_grid_size, patch_grid_size), dtype=bool)
        for i in range(patch_grid_size):
            for j in range(patch_grid_size):
                region = affordance_mask[
                    i * patch_h : (i + 1) * patch_h,
                    j * patch_w : (j + 1) * patch_w
                ]
                # Majority vote: patch belongs to this part if >50% of pixels match
                if (region == cat_id).sum() > (patch_h * patch_w * 0.5):
                    patch_mask[i, j] = True
        if patch_mask.any():
            masks[cat_name] = patch_mask
    
    return masks
```

---

## Method 3: depth/normal augmentation delta

### What it measures

How much geometric information does the encoder already have vs. need from external 3D cues? This is measured as the delta in mIoU between probing with and without concatenated depth/normal features from Metric3Dv2. A small delta means the encoder already encodes strong geometry. A large delta means it relies on external geometric help.

This method reuses the entire Method 1 pipeline — same linear probe architecture, same training, same evaluation. The only difference is the input features.

### Why this matters for our thesis

If raw SigLIP has a large delta (say +15 mIoU with depth) but π0-fine-tuned SigLIP has a smaller delta (say +8 mIoU), that is direct evidence that VLA fine-tuning injected geometric understanding into the encoder. If both have the same large delta, fine-tuning did not help geometry. If both have a small delta... that would be surprising and would mean SigLIP already has decent geometry (contradicting Zhang et al.).

### Implementation

```python
import torch

def extract_depth_normal_features(
    image_path: str,
    patch_grid_size: int = 16,
) -> torch.Tensor:
    """
    Extract depth and surface normal estimates using Metric3Dv2,
    then downsample to patch grid resolution.
    
    Returns:
        features: (1, 4, patch_grid_size, patch_grid_size)
        Channel 0: depth
        Channels 1-3: surface normal (x, y, z)
    """
    # Load and run Metric3Dv2
    # The exact loading code depends on the Metric3Dv2 installation
    # See: https://github.com/YvanYin/Metric3D
    #
    # Pseudocode:
    # model = load_metric3dv2()
    # depth, normals = model.predict(image)
    # depth: (H, W, 1), normals: (H, W, 3)
    #
    # Downsample to patch grid:
    # depth_patches = F.adaptive_avg_pool2d(depth, (patch_grid_size, patch_grid_size))
    # normal_patches = F.adaptive_avg_pool2d(normals, (patch_grid_size, patch_grid_size))
    # 
    # return torch.cat([depth_patches, normal_patches], dim=1)  # (1, 4, 16, 16)
    raise NotImplementedError("Implement with your Metric3Dv2 installation")


def extract_features_with_depth(
    encoder,
    image: torch.Tensor,
    image_path: str,
    encoder_type: str,
) -> torch.Tensor:
    """
    Extract multi-layer fused visual features AND concatenate
    Metric3Dv2 depth/normal features.
    
    Returns:
        features: (B, C_fused + 4, H_grid, W_grid)
    """
    # Visual features from encoder
    visual_features = extract_multilayer_features(encoder, image, encoder_type)
    # visual_features: (1, C_fused, 16, 16)
    
    # Depth/normal features
    depth_normal = extract_depth_normal_features(image_path)
    # depth_normal: (1, 4, 16, 16)
    
    # Concatenate along channel dimension
    combined = torch.cat([visual_features, depth_normal.to(visual_features.device)], dim=1)
    # combined: (1, C_fused + 4, 16, 16)
    
    return combined
```

### Running the experiment

This is identical to Method 1 but with an augmented feature dimension:

1. Extract features WITH depth/normal concatenation for all UMD images.
2. Train a new linear probe with `feature_dim = C_fused + 4` (e.g. 4612 for SigLIP, 3076 for DINOv2).
3. Evaluate mIoU on test set.
4. Compute delta: `delta = mIoU_with_depth - mIoU_without_depth`.

### Expected results table

| System | mIoU (visual only) | mIoU (+ depth/normal) | Delta | Interpretation |
|--------|--------------------|-----------------------|-------|----------------|
| Raw SigLIP | ? | ? | ? | Large delta = weak inherent geometry |
| π0 SigLIP | ? | ? | ? | Smaller delta than raw = VLA helped |
| π0.5 SigLIP | ? | ? | ? | Compare to π0 |
| DINOv2 | ~0.670 | ~0.67-0.68 | ~0 | Zhang et al.: barely benefits from depth |
| DINO-WM states | ? | ? | ? | Does dynamics prediction affect geometry? |

### Metric3Dv2 setup

```bash
# Install Metric3Dv2
git clone https://github.com/YvanYin/Metric3D.git
cd Metric3D
pip install -r requirements.txt --break-system-packages
# Download pretrained weights (check repo for latest links)
```

If Metric3Dv2 setup proves too complex for the timeline, an acceptable fallback is to use DPT (Dense Prediction Transformer) from the `transformers` library for monocular depth estimation. The depth quality will be lower than Metric3Dv2 but the diagnostic (delta between with/without) is still meaningful. The key is using the same depth source for all encoders so the comparison is fair.

---

## Execution order (revised)

### Week 1-2: infrastructure

Follow the original guide's week 1-2 plan, but add:

- Verify `output_hidden_states=True` works for each encoder (run the verification script above)
- Print the layer indices that will be used for multi-layer extraction
- Cache the FUSED multi-layer features, not single-layer features
- Set up Metric3Dv2 (or DPT fallback) and verify it produces depth/normal maps

Validation checkpoint (in addition to original guide's):
- For each encoder, print: number of hidden states, probe layer indices, fused feature dimension
- For one test image, extract fused features and confirm shape matches expected dimensions
- Metric3Dv2 produces (1, 4, 16, 16) depth/normal patches for a test image

### Week 3-4: core probing

1. Cache fused features from all 5 encoders on entire UMD dataset (train + test)
2. Cache depth-augmented fused features for all 5 encoders
3. **Method 1**: train linear probes on visual-only features, report mIoU for all 5 systems
4. **Method 3**: train linear probes on depth-augmented features, report mIoU, compute deltas
5. **Method 2**: run PCA analysis on reference objects, generate visualizations for all 5 systems
6. Compute part separation ratios for PCA analysis

Validation checkpoint:
- DINOv2 mIoU matches ~0.670 (validates probing infrastructure)
- DINOv2 depth delta is near zero (validates depth augmentation is working correctly)
- PCA visualizations for DINOv2 show clean part-level separation (validates PCA pipeline)

### Week 5-6: analysis and report

1. Compile mIoU table (with and without depth augmentation)
2. Generate publication-quality PCA visualization grids
3. Compute weight divergence between raw and fine-tuned SigLIP variants
4. Statistical analysis: run linear probing 3x with different seeds, report mean ± std
5. Write report framing results in context of Zhang et al.

---

## Key numbers to reproduce / validate against

| Measurement | Expected value | Source |
|-------------|---------------|--------|
| DINOv2 mIoU on UMD | ~0.670 | Zhang et al. Table/Figure 4 |
| DINOv2 depth augmentation delta | ~0 (very small) | Zhang et al. — "barely benefits from depth" |
| SigLIP/CLIP geometric affordance | ~3/10 informal, weak mIoU | Zhang et al. observations |
| UMD dataset size | 11,800 train / 14,020 test | Zhang et al. Section 3.1 |
| UMD categories | 7 affordance classes + background | grasp, cut, scoop, contain, pound, support, wrap-grasp |
| Probe architecture | BatchNorm2d + Conv2d(C, 8, 1) | Zhang et al. following Probing3D |
| Feature fusion | 4 equally-spaced layers concatenated | Zhang et al. Section 3.1 |

---

## SigLIP-specific technical notes

### Architecture recap

- Model: SigLIP-So400m/14 (shape-optimized, ~400M params)
- "So" = shape-optimized architecture from "Getting ViT in Shape" paper
- 27 encoder layers, 1152 hidden dim, 16 attention heads, head dim = 72, MLP intermediate = 4304
- Patch size 14x14, input 224x224 → 16x16 grid = 256 patch tokens
- No CLS token — uses MAP head (multi-head attention pooling) for contrastive loss
- MAP head is discarded when used inside PaliGemma/π0 (pool_type="none")
- Patch tokens were trained as keys/values for a global contrastive query, never directly supervised for local spatial content

### SigLIP inside π0

- PaliGemma wraps SigLIP + linear projector (1152→2048) + Gemma 2B LLM
- π0 adds ~300M action expert (randomly initialized Gemma-based transformer)
- π0 training: VLM backbone (including SigLIP) is fine-tuned end-to-end via flow matching loss — this is the "naive" approach where randomly-initialized action expert gradients flow backward through the entire model
- π0.5 training: uses "knowledge insulation" — gradients from the action expert are STOPPED before reaching the VLM backbone. Instead, the VLM learns about actions via discrete FAST tokens (next-token prediction, native to the LLM objective). Co-trained on VLM data (captioning, VQA) to preserve pretrained knowledge

### What this means for probing

We are comparing three gradient regimes on the same SigLIP architecture:
1. **Raw SigLIP**: no robot gradients at all
2. **π0 SigLIP**: receives noisy gradients from randomly-initialized action expert (may DAMAGE spatial features)
3. **π0.5 SigLIP**: receives gradients only from discrete action token prediction + VLM co-training (protects representations)

The weight divergence analysis (see original guide Section 4) should be run FIRST to confirm the encoders actually changed. If π0's SigLIP weights are identical to raw SigLIP, that means the encoder was frozen and the entire probing comparison collapses to a single data point.

---

## DINOv2-specific technical notes

### Architecture options

- ViT-B/14: 12 layers, 768 hidden dim, 12 heads — smaller, faster
- ViT-L/14: 24 layers, 1024 hidden dim, 16 heads — closer to SigLIP's scale

Recommendation: use ViT-B/14 as the primary comparison (it's what DINO-WM uses), but run ViT-L/14 as well if time permits since it's a fairer parameter-count comparison with SigLIP's 400M.

### Why DINOv2 is the geometric ceiling

DINOv2 is trained with two self-distillation objectives:
- DINO loss: image-level cross-entropy between student and EMA-teacher CLS tokens (provides global context)
- iBOT loss: patch-level masked image modeling where student predicts teacher's features at masked positions (directly supervises individual patches)

The iBOT loss is the key difference from SigLIP. Every DINOv2 patch token receives a direct learning signal about what it should locally represent. SigLIP patch tokens only receive diluted indirect gradient from the global MAP pooling operation.

### DINO-WM world states

When probing DINO-WM predicted states, the features come from the transition model's output, not from DINOv2 directly. The transition model predicts future DINOv2 patch features given action history. These predicted features should be probed with the same methodology as raw DINOv2 features — same multi-layer extraction is not applicable here since the transition model has its own architecture. Instead, probe the transition model's output directly (it should already be in DINOv2 feature space). Use the DINOv2 single-layer feature dimension (768 for ViT-B) for the probe on predicted states, and compare against raw DINOv2 single-layer features for the same frames.
