# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geometric Affordance Probing framework that benchmarks 5 vision encoders on the UMD Part Affordance Dataset. Compares how VLA fine-tuning (pi0, pi0.5) affects affordance understanding vs raw pre-training (SigLIP) and self-supervised baselines (DINOv2, DINO-WM).

**Language:** Python 3.11+ with PyTorch, transformers, scikit-learn

## Environment Setup

```bash
conda activate affordance
pip install -e .
pip install -r requirements.txt
```

Conda env: `affordance` (Python 3.11, PyTorch 2.11.0, CUDA 13.0 for RTX 5090).

## Running the Pipeline

The project uses a 7-step numbered script pipeline. Each script is independently runnable:

```bash
python scripts/01_setup_encoders.py                    # Verify encoders
python scripts/02_extract_features.py                  # Extract & cache features
python scripts/03_run_linear_probing.py --epochs 50    # Train linear probes
python scripts/04_run_pca_analysis.py                  # PCA subspace analysis
python scripts/05_run_cosine_similarity.py             # Patch correspondence
python scripts/06_weight_divergence.py                 # Weight change analysis
python scripts/07_generate_report.py                   # Final report
```

No formal test suite, CI/CD, or linting configuration exists.

## Architecture

### Unified Feature Extractor (central abstraction)

`encoders/feature_extractor.py` defines `UnifiedFeatureExtractor` with a registry pattern (`ENCODER_REGISTRY`). All encoders output **256 patch tokens** (16x16 grid from 224x224 input, patch size 14). The rest of the system is encoder-agnostic.

Each registry entry has: `loader` (one-time init), `extractor` (per-image feature extraction), `feature_dim` (1152 for SigLIP variants, 768 for DINOv2 variants).

### Encoders (`encoders/`)

| Module | Source | Feature Dim |
|--------|--------|-------------|
| `raw_siglip.py` | google/siglip-so400m-patch14-384 | 1152 |
| `pi0_siglip.py` | lerobot/pi0_base | 1152 |
| `pi05_siglip.py` | lerobot/pi05_base | 1152 |
| `dinov2.py` | facebook/dinov2-base | 768 |
| `dino_wm.py` | mazpie/dino-wm + transition model | 768 |

### Probing Protocol

All probing keeps encoders **frozen** (no gradients). Only the probe head is trained. This isolates what the encoder already encodes.

- **Linear probe** (`probing/linear_probe.py`): BatchNorm2d + 1x1 Conv, evaluated by mIoU
- **PCA analysis** (`probing/pca_analysis.py`): 3-component subspace projection, part separation ratio
- **Cosine similarity** (`probing/cosine_similarity.py`): Cross-object patch correspondence, Hit@K
- **Weight divergence** (`probing/weight_divergence.py`): Per-layer L2/cosine comparison (pi0 vs raw SigLIP)

### Data Flow

```
UMD Dataset (RGB + affordance masks, 8 classes)
  → Encoders (frozen, extract patch features)
  → Cached features (numpy arrays in results/cached_features/)
  → Probing experiments
  → Results (results/figures/, results/tables/)
```

Features are cached after step 02 so later steps don't require GPU re-inference.

### Adding a New Encoder

1. Create `encoders/my_encoder.py` with `load_*()` and `extract_*()` functions
2. Register in `ENCODER_REGISTRY` in `encoders/feature_extractor.py`
3. All pipeline scripts automatically pick it up

## Configuration

Central config: `configs/probing_config.yaml` — paths, encoder specs, hyperparameters, affordance categories.

## Key Constraints

- Encoders must always be frozen during probing (no fine-tuning)
- All encoders must output (B, 256, feature_dim) patch tokens for fair comparison
- UMD dataset has 7 affordances + background (8 classes total): grasp, cut, scoop, contain, pound, support, wrap-grasp
