# Affordance Probing Project — Status Report
**Date:** 2026-03-26 (updated)

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
│   ├── paligemma_siglip.py          # google/paligemma-3b-pt-224 (NEW)
│   ├── pi0_siglip.py                # lerobot/pi0_base
│   ├── pi05_siglip.py               # lerobot/pi05_base
│   ├── dinov2.py                    # facebook/dinov2-base
│   ├── dino_wm.py                   # DINO-WM (deferred)
│   └── feature_extractor.py         # Unified registry + extraction
├── probing/
│   ├── linear_probe.py              # Method 1: linear probing (mIoU)
│   ├── pca_analysis.py              # Method 2: PCA subspace projection
│   ├── depth_normal.py              # Method 3: depth/normal augmentation (NEW)
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
│   ├── 05_extract_depth_normal.py   # Cache DPT depth + normals (NEW)
│   ├── 06_run_depth_augmentation.py # Depth augmentation experiment (NEW)
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
| 2 | PaliGemma SigLIP | google/paligemma-3b-pt-224 | **Needs HF access** | 4608 |
| 3 | pi0 SigLIP | lerobot/pi0_base | **Ready** | 4608 |
| 4 | pi0.5 SigLIP | lerobot/pi05_base | **Ready** | 4608 |
| 5 | DINOv2 | facebook/dinov2-base | **Ready** | 3072 |
| 6 | DINO-WM | mazpie/dino-wm | **Deferred** | 3072 |

### Key Changes (2026-03-26)

1. **Multi-layer feature fusion** — All encoders now extract from 4 equally-spaced layers and concatenate. Feature dims are 4x larger (4608 for SigLIP, 3072 for DINOv2).
2. **Raw SigLIP switched to 224-native** — Using `google/siglip-so400m-patch14-224` instead of the 384 model. Eliminates position embedding interpolation artifacts.
3. **PaliGemma added as encoder** — Intermediate probing target between raw SigLIP and pi0. PaliGemma retrains SigLIP on 1B multimodal examples (unfrozen). Requires HuggingFace gated access.
4. **Depth/normal augmentation added** — New Method 3 using DPT for monocular depth + finite-difference normals. Measures how much external geometry helps each encoder.
5. **Cosine similarity deprioritized** — Redundant with PCA per supplement. Kept as stretch goal.
6. **Scripts renumbered** — Now 8 steps (was 7).

### UMD Dataset (Downloaded)
- **UMD+GT version** — 6 affordances (grasp, cut, scoop, contain, pound, wrap-grasp) + background
- Source: Google Drive via [AffKpNet repo](https://github.com/ivalab/AffKpNet)

---

## Active Issues

### Issue 1: PaliGemma HuggingFace Access
`google/paligemma-3b-pt-224` is a gated model. Need to request access on HuggingFace and accept the license.

### Issue 2: Affordance Label Mapping Mismatch
The UMD+GT dataset has 6 affordances at indices 1-6 (wrap-grasp at index 6). The original code defined "support" at index 6 and "wrap-grasp" at index 7 (8 classes). Config updated to 7 classes (6 affordances + background), but `data/umd_dataset.py` category mapping needs verification against actual label files.

### Issue 3: DINO-WM Setup (Deferred)
Need to clone repo, download checkpoint, adapt transition model loading.

---

## Next Steps (In Priority Order)

1. **Request PaliGemma HuggingFace access** — needed for the progression experiment
2. **Fix affordance label mapping** — verify actual label values in UMD+GT dataset, update `umd_dataset.py`
3. **Run `01_setup_encoders.py`** — verify all encoders produce correct multi-layer shapes
4. **Extract and cache features** via `02_extract_features.py`
5. **Run linear probing** (Method 1) — validate DINOv2 reproduces ~0.670 mIoU
6. **Extract depth/normal features** via `05_extract_depth_normal.py`
7. **Run depth augmentation** (Method 3) — validate DINOv2 delta ~0
8. **Run PCA analysis** (Method 2) — generate visualizations
9. **Run weight divergence** — confirm SigLIP weights differ across stages
