# Affordance Probing Project — Status Report
**Date:** 2026-04-06 (updated)

---

## What's Done

### Project Structure (Complete)

```
affordance/
├── configs/probing_config.yaml
├── data/
│   ├── README.md                    # Dataset download + reproducibility guide
│   ├── download_umd.py
│   └── umd_dataset.py
├── docs/
│   └── siglip_progression.md        # Research: SigLIP through PaliGemma/pi0/pi0.5
├── encoders/
│   ├── multilayer.py                # Multi-layer feature fusion (Zhang et al.)
│   ├── raw_siglip.py                # google/siglip-so400m-patch14-224 (224-native)
│   ├── paligemma_siglip.py          # google/paligemma-3b-pt-224
│   ├── pi0_siglip.py                # lerobot/pi0_base
│   ├── pi05_siglip.py               # lerobot/pi05_base
│   ├── dinov2.py                    # facebook/dinov2-base
│   ├── dino_wm.py                   # DINO-WM (deferred)
│   └── feature_extractor.py         # Unified registry + extraction
├── probing/
│   ├── linear_probe.py              # Method 1: linear probing (mIoU)
│   ├── pca_analysis.py              # Method 2: PCA subspace projection
│   ├── depth_normal.py              # Method 3: depth/normal augmentation
│   ├── cosine_similarity.py         # Deprioritized
│   └── weight_divergence.py
├── evaluation/
│   ├── metrics.py
│   └── visualization.py
├── scripts/
│   ├── 01_setup_encoders.py         # Verify encoders + multi-layer shapes
│   ├── 02_extract_features.py       # Cache multi-layer fused features (float16)
│   ├── 03_run_linear_probing.py     # Train linear probes
│   ├── 04_run_pca_analysis.py       # PCA visualization
│   ├── 04b_run_cosine_similarity.py # Deprioritized
│   ├── 05_extract_depth_normal.py   # Cache DPT depth + normals
│   ├── 06_run_depth_augmentation.py # Depth augmentation experiment
│   ├── 07_weight_divergence.py      # Weight comparison (now includes PaliGemma)
│   └── 08_generate_report.py        # Final report
├── results/{figures,tables,cached_features}/
├── requirements.txt
├── setup.py
├── AXIS1_IMPLEMENTATION_GUIDE.pdf
└── AXIS1_PROBING_METHODS_SUPPLEMENT.md
```

### Conda Environment (Complete)
- **Name:** `affordance`
- **Python:** 3.11
- **PyTorch:** 2.11.0 (cu128 wheels for RTX 5090, CUDA 13.0 driver)
- **Other deps:** transformers (4.53.3 custom fork), lerobot 0.4.4, scikit-learn, numpy, scipy, matplotlib, seaborn, pillow, opencv-python, wandb, tqdm
- **Note:** pi0/pi0.5 require a custom transformers fork (`fix/lerobot_openpi` branch) for the SigLIP check module

### Encoders (5 of 6 ready)

| # | Encoder | Source | Status | Fused dim |
|---|---------|--------|--------|-----------|
| 1 | Raw SigLIP | google/siglip-so400m-patch14-224 | **Ready** | 4608 |
| 2 | PaliGemma SigLIP | google/paligemma-3b-pt-224 | **Ready** (HF access granted) | 4608 |
| 3 | pi0 SigLIP | lerobot/pi0_base | **Ready** | 4608 |
| 4 | pi0.5 SigLIP | lerobot/pi05_base | **Ready** | 4608 |
| 5 | DINOv2 | facebook/dinov2-base | **Ready** | 3072 |
| 6 | DINO-WM | mazpie/dino-wm | **Deferred** | 3072 |

### UMD Dataset (Available)
- **Original UMD Part Affordance Dataset** — 7 affordances + background (8 classes)
- Source: https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/
- Tools: 2.9 GB, Clutter: 0.2 GB
- Labels in `.mat` format (MATLAB), RGB as `.jpg`
- Data loader updated to handle both original UMD and UMD+GT formats

### Protocol Alignment with Zhang et al. / Probe3D (Complete)

Comprehensive comparison against Zhang et al. (arXiv 2602.20501, CVPR 2026) and the Probe3D reference codebase (El Banani et al., CVPR 2024) was performed. All identified discrepancies have been fixed:

1. **mIoU computation** — Now computed over 7 affordance classes (excluding background class 0), matching Zhang et al.'s "seven affordance categories." Both `mIoU` (7-class) and `mIoU_all` (8-class) are reported.

2. **Probe architecture** — Forward pass reordered to match Probe3D: features upsampled 4x (16x16 → 64x64) BEFORE BatchNorm + 1x1 Conv, then resized to target. Uses `align_corners=True` throughout.

3. **Optimizer** — Switched from Adam (constant LR) to AdamW with cosine LR schedule + 10% linear warmup, matching Probe3D protocol. Weight decay = 0.05.

4. **Confusion matrix** — Vectorized via `torch.bincount` (was pixel-by-pixel Python loop — would have been unusably slow on 14K test images at 224x224).

5. **Config consistency** — Fixed `num_classes: 7` → `num_classes: 8` in config, added missing "support" category (class 6), updated optimizer settings to reflect AdamW + cosine schedule.

6. **Documentation** — `AXIS1_PROBING_METHODS_SUPPLEMENT.md` updated with protocol alignment section, refreshed code snippets, updated validation targets, and model scale divergence rationale.

**Files modified:** `probing/linear_probe.py`, `evaluation/metrics.py`, `scripts/03_run_linear_probing.py`, `scripts/06_run_depth_augmentation.py`, `configs/probing_config.yaml`, `AXIS1_PROBING_METHODS_SUPPLEMENT.md`

**Model scale divergence (documented):** Zhang et al. use ViT-B sized models (12 layers, 768-dim). We use SigLIP-SO400M (27 layers, 1152-dim) because that's what PaliGemma/π0/π0.5 actually use. Our DINOv2-B matches theirs. Numbers not directly comparable for SigLIP, but our research question is the SigLIP progression, not replicating their exact values.

### Key Changes History

**2026-04-06 — Protocol alignment**
- Linear probing pipeline fully aligned with Probe3D/Zhang et al. protocol (see above)
- PaliGemma HuggingFace access obtained

**2026-03-26 — Multi-layer fusion + depth augmentation**
1. Multi-layer feature fusion — All encoders extract from 4 equally-spaced layers and concatenate
2. Raw SigLIP switched to 224-native model
3. PaliGemma added as encoder
4. Depth/normal augmentation added (Method 3)
5. Cosine similarity deprioritized
6. Scripts renumbered to 8 steps

---

## Active Issues

### Issue 1: PaliGemma HuggingFace Access (RESOLVED)
HuggingFace gated access to `google/paligemma-3b-pt-224` has been obtained.

### Issue 2: Affordance Label Mapping (RESOLVED)
Original UMD dataset found with all 7 affordances (1-7: grasp, cut, scoop, contain, pound, support, wrap-grasp). Data loader updated to handle `.mat` label files and `category_split.txt` train/test split. Config has 8 classes (7 affordances + background).

### Issue 3: DINO-WM Setup (Deferred)
Need to clone repo, download checkpoint, adapt transition model loading.

---

## Next Steps (In Priority Order)

1. **Run `01_setup_encoders.py`** — verify all encoders produce correct multi-layer shapes
2. **Extract and cache features** via `02_extract_features.py`
3. **Run linear probing** (Method 1) — validate DINOv2 reproduces ~0.662 mIoU (7-class)
4. **Extract depth/normal features** via `05_extract_depth_normal.py`
5. **Run depth augmentation** (Method 3) — validate DINOv2 delta ~0
6. **Run PCA analysis** (Method 2) — generate visualizations
7. **Run weight divergence** — confirm SigLIP weights differ across stages

---

## Axis 2: Interaction Affordance (Flux)

**Status:** Infrastructure complete, not yet run

### What's Done

- `interaction/` module with Flux cross-attention extraction (`flux_attention.py`)
- Verb-spatial binding metrics: KLD, SIM, NSS (`verb_spatial_binding.py`)
- Visualization pipeline: overlays, comparison grids, timestep progression (`visualization.py`)
- AGD20K dataset loader with download helper (`data/agd20k_dataset.py`, `data/download_agd20k.py`)
- Scripts 09-11 (setup, probing, report generation)
- Axis 2 config section in `probing_config.yaml`
- Separate `requirements_axis2.txt` for Colab dependencies

### Compute Requirements

- **Must run on Colab Pro (A100 40GB)** — Flux is ~12B params, needs ~24GB VRAM
- Local RTX 5060 (8GB) is insufficient for Flux inference
- Workflow: push to GitHub → clone on Colab → install → run scripts 09-11

### Next Steps (Axis 2)

1. Download AGD20K dataset to Google Drive
2. Run `09_setup_flux.py` on Colab with schnell — verify pipeline
3. Run `10_run_interaction_probing.py` with schnell on a few categories — sanity check
4. Switch to `FLUX.1-dev` for final results
5. Run `11_generate_axis2_report.py` — produce figures and tables

