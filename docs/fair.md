# Fairness Analysis: Zhang et al.'s SigLIP vs DINOv2 Comparison

Zhang et al. compare SigLIP and DINOv2 via linear probing on UMD affordance segmentation. These encoders differ in architecture (patch size, training objective). This document assesses: what design decisions make this comparison fair, and where does it fall short?

All source references are relative to `Probing_Bridging_Affordance/geometry_probing/umd_linear_probing/`.

---

## 1. The Core Asymmetry: What Differs Between SigLIP and DINOv2

| Property | DINOv2 (ViT-B/14) | SigLIP (ViT-B/16) |
|----------|-------------------|-------------------|
| Patch size | 14 | 16 |
| Hidden dim | 768 | 768 |
| Layers | 12 | 12 |
| Parameters | ~86M | ~86M |
| Training objective | DINO + iBOT (self-supervised, patch-level MIM) | SigLIP (contrastive image-text, sigmoid loss) |
| Native resolution | 224×224 | 384×384 |
| CLS token | Yes | No (or optional) |

Zhang et al. chose **ViT-B variants of both** — same depth (12 layers), same hidden dim (768), same parameter count (~86M). This is a deliberate capacity-matching decision. The only architectural difference is patch size (14 vs 16).

Sources: `configs/dinov2.yaml` → `dinov2_vitb14`, `configs/siglip.yaml` → `google/siglip-base-patch16-384`

---

## 2. Design Decisions That Promote Fairness

### 2.1 Matched Model Capacity

Both are ViT-B (~86M params, 12 layers, 768 dim). This is the single most important fairness decision — it isolates the training objective (self-supervised geometric vs contrastive language) rather than conflating it with model capacity.

### 2.2 Identical Training Hyperparameters

Both configs use identical probe training:

| Parameter | DINOv2 | SigLIP |
|-----------|--------|--------|
| batch_size | 4 | 4 |
| lr | 0.001 | 0.001 |
| weight_decay | 0.01 | 0.01 |
| max_epochs | 2 | 2 |
| patience | 1 | 1 |
| grad_clip_norm | 1.0 | 1.0 |
| seed | 1337 | 1337 |
| precision | bf16 | bf16 |

Sources: `configs/dinov2.yaml:training`, `configs/siglip.yaml:training`

### 2.3 Same Probe Head Architecture

Both use `MultiLayerLinearHead` with identical configuration:
- `feature_keys: [2, 5, 8, 11]`, `primary_key: 2`
- `fuse_mode: concat`, `use_batchnorm: true`, `dropout: 0.0`

The head aligns all layers to the primary key's spatial grid via bilinear interpolation, then concatenates → BN → 1×1 Conv.

Source: `src/models/linear_head.py:93-110`

### 2.4 Same Layer Selection

Both probe layers [2, 5, 8, 11] — the same 4 equally-spaced layers from a 12-layer encoder. This ensures the same depth profile is sampled.

### 2.5 Same Evaluation Protocol

Both upsample logits to pixel resolution (480×640) before computing mIoU. Both ignore class 0 (background) via `metric_ignore_indices: [0]`.

Source: `src/engine/eval.py:85-91`, `src/engine/trainer.py:797-804`

### 2.6 Same Loss Function and Target

Both use `CrossEntropyLoss(ignore_index=255)` computed at patch-grid resolution. Targets are majority-vote downsampled masks with `min_patch_coverage=0.55`.

Source: `src/engine/trainer.py:560`, `src/data/dataset.py:20-63`

### 2.7 Symmetric Geometry Augmentation

Both published configs have `use_depth: true` and `use_normal: true`. Geometry features (Metric3D depth + surface normals) are concatenated into the probe head symmetrically for both encoders.

Sources: `configs/dinov2.yaml:geometry`, `configs/siglip.yaml:geometry`

### 2.8 No Image Resize

Both receive images at native 480×640 resolution — no resize step. The transform is `ToTensor()` + ImageNet normalization only.

Source: `src/data/transforms.py:16-22`

### 2.9 Fused Feature Dimension

Both encoders output 768-dim per layer. With 4 layers concatenated:
- DINOv2: 768 × 4 = 3,072 fused channels
- SigLIP: 768 × 4 = 3,072 fused channels

The probe head's 1×1 Conv projects from the same input dimensionality. With geometry: both get +1 (depth) + 3 (normal) = 3,076 total.

---

## 3. Where the Comparison Falls Short (Remaining Asymmetries)

### 3.1 Patch Size → Different Spatial Grids (MEDIUM impact)

DINOv2 (patch 14): 480×640 → requires padding → 490×644 → 35×46 = **1,610 tokens**
SigLIP (patch 16): 480×640 → divides evenly → 30×40 = **1,200 tokens**

This creates several downstream effects:

- **Loss granularity**: DINOv2's loss is computed over 1,610 spatial locations; SigLIP over 1,200. DINOv2 gets ~34% more supervision signal per image.
- **Mask downsampling fidelity**: 14×14 pixel patches capture finer affordance boundaries than 16×16 patches. With `min_patch_coverage=0.55`, more 14×14 patches will achieve majority coverage of small affordance regions.
- **Evaluation upsampling**: DINOv2 upsamples from 35×46 → 480×640 (~14× upscale); SigLIP from 30×40 → 480×640 (16× upscale). DINOv2's predictions start closer to pixel resolution.

Configs: `dinov2.yaml` → `pad_to_patch_multiple: true`, `siglip.yaml` → `pad_to_patch_multiple: false`

This is inherent to comparing ViT-14 vs ViT-16 without resizing — there's no clean fix. Zhang et al. chose to keep native patch sizes rather than artificially matching grids, which is a reasonable methodological choice (matching grids would distort one encoder's spatial behavior).

### 3.2 Feature Normalization Asymmetry (MEDIUM impact)

DINOv2 uses `get_intermediate_layers(norm=True)` — applies the model's final LayerNorm to each intermediate layer's output before returning features.

SigLIP uses `output_hidden_states=True` via HuggingFace — returns raw transformer hidden states **without** LayerNorm.

This means:
- DINOv2 features are layer-normalized (zero mean, unit variance per token)
- SigLIP features have raw, unnormalized activation distributions

The probe head includes BatchNorm2d which partially compensates, but BN operates per-channel across spatial locations + batch, not per-token like LayerNorm. The initial feature distributions feeding into BN differ.

Sources: `src/models/dinov2.py:94-98` (`norm=True`), `src/models/siglip2.py:106-112` (`output_hidden_states=True`)

This is likely an implementation artifact from using different extraction APIs (torch.hub `get_intermediate_layers` vs HuggingFace `output_hidden_states`) rather than a deliberate design choice.

### 3.3 Image Preprocessing Path (LOW impact)

DINOv2: Images go through `ToTensor()` + ImageNet normalize → directly to `get_intermediate_layers()`. DINOv2 was trained with ImageNet normalization by convention.

SigLIP: Images go through `ToTensor()` + ImageNet normalize → SigLIP2Backbone internally denormalizes from ImageNet range, then renormalizes with SigLIP-specific mean/std (`siglip2.py:96-104`). This is correct because SigLIP was trained with different normalization statistics.

Both encoders receive correctly normalized inputs for their respective training regimes. The difference is in the code path, not the semantic correctness.

### 3.4 Position Encoding Interpolation (LOW impact)

SigLIP explicitly sets `interpolate_pos_encoding=True` (`siglip2.py:111`) because 480×640 differs from its 384×384 training resolution.

DINOv2's torch.hub `get_intermediate_layers()` handles position encoding interpolation internally for non-native resolutions.

Both models interpolate positional encodings — the difference is in the API surface, not the behavior.

### 3.5 Padding Artifacts (LOW impact)

DINOv2 images are padded from 480×640 to 490×644 with zeros. Padded mask regions get `ignore_index=255`, so they're excluded from loss. But the encoder still processes padded pixels, which could slightly affect attention patterns in border patches.

SigLIP needs no padding (480 and 640 divide evenly by 16).

Source: `src/data/dataset.py:250-271`

---

## 4. The Argument For Fairness

The strongest argument for this being a fair comparison:

1. **Same model capacity** (ViT-B, 12 layers, 768 dim, ~86M params) — the most important control variable
2. **Same probing protocol** (identical head architecture, training hyperparameters, evaluation)
3. **Same data** (no resize, same splits, same coverage threshold, same geometry augmentation)
4. **The question being asked is well-scoped**: "Does self-supervised geometric pretraining (DINO+iBOT) encode more affordance-relevant structure than contrastive image-text pretraining (SigLIP)?"

The patch size difference (14 vs 16) is intrinsic to the canonical ViT-B variants of each model and cannot be eliminated without changing one model's architecture. Accepting this asymmetry is preferable to introducing artificial distortions.

---

## 5. The Argument Against Fairness

1. **Patch size gives DINOv2 a spatial advantage**: 1,610 tokens vs 1,200 → finer spatial resolution, 34% more supervision signal, less information loss in mask downsampling. If DINOv2 outperforms SigLIP, part of the gap may be attributable to spatial resolution rather than feature quality.

2. **Feature normalization is inconsistent**: DINOv2 gets LayerNorm on intermediate features; SigLIP gets raw hidden states. The BN in the probe head partially compensates, but the initial optimization landscape differs. This could disadvantage SigLIP if its raw activations have high variance or outlier magnitudes.

3. **The comparison conflates two variables**: training objective AND patch size. A cleaner comparison would use models with the same patch size (e.g., DINOv2-B/16 or SigLIP-B/14, if they existed). Since Zhang et al. chose the canonical ViT-B variants of each, the patch size confound is embedded.

---

## 6. Summary

| Fairness Dimension | Status | Notes |
|-------------------|--------|-------|
| Model capacity | **FAIR** | Both ViT-B, 12L, 768d, ~86M params |
| Layer selection | **FAIR** | [2, 5, 8, 11] for both |
| Probe head | **FAIR** | Identical MultiLayerLinearHead |
| Training hyperparams | **FAIR** | All identical (lr, wd, epochs, patience, seed) |
| Evaluation protocol | **FAIR** | Both upsample to pixel resolution |
| Loss function | **FAIR** | Same CE on patch grid, same ignore_index |
| Geometry augmentation | **FAIR** | Both depth+normal enabled |
| Fused feature dim | **FAIR** | Both 3,072 (768 × 4) |
| Patch size / spatial grid | **NOT FAIR** | 14 vs 16 → 1,610 vs 1,200 tokens |
| Feature normalization | **NOT FAIR** | LayerNorm (DINOv2) vs raw (SigLIP) |
| Image preprocessing | MINOR | Different normalization paths, both correct |
| Pos encoding interp | MINOR | Both interpolate, different APIs |
| Padding | MINOR | Only DINOv2 padded (490×644) |

**Bottom line**: The comparison is largely fair — capacity-matched models with identical probing protocol. The two notable asymmetries (patch size → spatial resolution, feature normalization) are worth noting but are unlikely to fully explain performance differences. The patch size confound is inherent to comparing ViT-14 vs ViT-16; the normalization inconsistency is likely an implementation artifact that the probe head's BatchNorm partially mitigates.
