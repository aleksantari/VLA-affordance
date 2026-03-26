# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geometric Affordance Probing framework that benchmarks 6 vision encoders on the UMD Part Affordance Dataset. Probes how the SigLIP vision encoder evolves through four training stages (raw → PaliGemma VL → π0 robot → π0.5 insulated robot) and compares against DINOv2 as a geometric ceiling.

**Language:** Python 3.11+ with PyTorch, transformers, scikit-learn

## Environment Setup

```bash
use_conda affordance
pip install -e .
pip install -r requirements.txt
```

Conda env: `affordance` (Python 3.11, PyTorch 2.11.0, CUDA 13.0 for RTX 5090).

## Running the Pipeline

The project uses an 8-step numbered script pipeline. Each script is independently runnable:

```bash
python scripts/01_setup_encoders.py                    # Verify encoders + multi-layer shapes
python scripts/02_extract_features.py                  # Extract & cache multi-layer fused features
python scripts/03_run_linear_probing.py --epochs 50    # Train linear probes (Method 1)
python scripts/04_run_pca_analysis.py                  # PCA subspace analysis (Method 2)
python scripts/05_extract_depth_normal.py              # Cache depth/normal features (DPT)
python scripts/06_run_depth_augmentation.py            # Depth augmentation experiment (Method 3)
python scripts/07_weight_divergence.py                 # Weight change analysis
python scripts/08_generate_report.py                   # Final report
```

Also: `scripts/04b_run_cosine_similarity.py` (deprioritized, kept for stretch goals).

No formal test suite, CI/CD, or linting configuration exists.

## Architecture

### Multi-layer Feature Fusion

Following Zhang et al. / Probing3D protocol, features are extracted from **4 equally-spaced intermediate layers** and concatenated along the feature dimension. This is implemented in `encoders/multilayer.py`.

- SigLIP (27 layers): probe layers [6, 13, 20, 26] → fused dim = 4608
- DINOv2 (12 layers): probe layers [2, 5, 8, 11] → fused dim = 3072

### Unified Feature Extractor (central abstraction)

`encoders/feature_extractor.py` defines `UnifiedFeatureExtractor` with a registry pattern (`ENCODER_REGISTRY`). All encoders output **256 patch tokens** (16x16 grid from 224x224 input, patch size 14).

Key methods:
- `extract(images)` → `(B, 256, feature_dim)` — single layer (backward compat)
- `extract_multilayer(images)` → `(B, 256, fused_feature_dim)` — 4-layer fused (primary)
- `extract_multilayer_spatial(images)` → `(B, fused_feature_dim, 16, 16)` — spatial grid

Each registry entry has: `loader`, `extractor`, `hidden_states_extractor`, `feature_dim`, `fused_feature_dim`, `encoder_type`, `num_layers`.

### Encoders (`encoders/`)

| Module | Source | Single dim | Fused dim |
|--------|--------|-----------|-----------|
| `raw_siglip.py` | google/siglip-so400m-patch14-224 | 1152 | 4608 |
| `paligemma_siglip.py` | google/paligemma-3b-pt-224 | 1152 | 4608 |
| `pi0_siglip.py` | lerobot/pi0_base | 1152 | 4608 |
| `pi05_siglip.py` | lerobot/pi05_base | 1152 | 4608 |
| `dinov2.py` | facebook/dinov2-base | 768 | 3072 |
| `dino_wm.py` | mazpie/dino-wm + transition model | 768 | 3072 |

### Probing Protocol

All probing keeps encoders **frozen** (no gradients). Only the probe head is trained. This isolates what the encoder already encodes.

- **Linear probe** (`probing/linear_probe.py`): BatchNorm2d + 1x1 Conv on fused features, evaluated by mIoU
- **PCA analysis** (`probing/pca_analysis.py`): 3-component subspace projection on fused features, part separation ratio
- **Depth/normal augmentation** (`probing/depth_normal.py`): Concatenate DPT depth+normals with encoder features, measure mIoU delta
- **Weight divergence** (`probing/weight_divergence.py`): Per-layer L2/cosine comparison across SigLIP progression

### Data Flow

```
UMD Dataset (RGB + affordance masks, 7 classes)
  → Encoders (frozen, extract multi-layer fused patch features)
  → Cached features (numpy arrays in results/cached_features/, float16)
  → Probing experiments
  → Results (results/figures/, results/tables/)
```

Features are cached after step 02 so later steps don't require GPU re-inference.

### Adding a New Encoder

1. Create `encoders/my_encoder.py` with `load_*()`, `extract_features()`, and `extract_hidden_states()` functions
2. Register in `ENCODER_REGISTRY` in `encoders/feature_extractor.py` with all required fields
3. All pipeline scripts automatically pick it up

## Configuration

Central config: `configs/probing_config.yaml` — paths, encoder specs (including fused dims), hyperparameters, multilayer settings, depth augmentation settings, affordance categories.

## Key Constraints

- Encoders must always be frozen during probing (no fine-tuning)
- All encoders must output (B, 256, feature_dim) patch tokens for fair comparison
- Multi-layer fused features are the primary input to all probing methods
- UMD+GT dataset has 6 affordances + background (7 classes): grasp, cut, scoop, contain, pound, wrap-grasp
