# Affordance Probing Project — Status Report
**Date:** 2026-03-26 (updated)

---

## What's Done

### Project Structure (Complete)
All 20+ source files written and ready:

```
affordance/
├── configs/probing_config.yaml
├── data/
│   ├── download_umd.py
│   └── umd_dataset.py
├── encoders/
│   ├── raw_siglip.py
│   ├── pi0_siglip.py
│   ├── pi05_siglip.py
│   ├── dinov2.py
│   ├── dino_wm.py
│   └── feature_extractor.py
├── probing/
│   ├── linear_probe.py
│   ├── pca_analysis.py
│   ├── cosine_similarity.py
│   └── weight_divergence.py
├── evaluation/
│   ├── metrics.py
│   └── visualization.py
├── scripts/
│   ├── 01_setup_encoders.py
│   ├── 02_extract_features.py
│   ├── 03_run_linear_probing.py
│   ├── 04_run_pca_analysis.py
│   ├── 05_run_cosine_similarity.py
│   ├── 06_weight_divergence.py
│   └── 07_generate_report.py
├── results/{figures,tables,cached_features}/
├── requirements.txt
├── setup.py
└── AXIS1_IMPLEMENTATION_GUIDE.pdf
```

### Conda Environment (Complete)
- **Name:** `affordance`
- **Python:** 3.11
- **PyTorch:** 2.11.0 (cu128 wheels for RTX 5090, CUDA 13.0 driver)
- **Other deps:** transformers (4.53.3 custom fork), lerobot 0.4.4, scikit-learn, numpy, scipy, matplotlib, seaborn, pillow, opencv-python, wandb, tqdm
- **Note:** pi0/pi0.5 require a custom transformers fork (`fix/lerobot_openpi` branch) for the SigLIP check module

### Models Downloaded (4 of 5)

| Model | Status | Notes |
|-------|--------|-------|
| Raw SigLIP (google/siglip-so400m-patch14-384) | **Downloaded** | 428.2M params |
| DINOv2 (facebook/dinov2-base) | **Downloaded** | 86.6M params |
| pi0 SigLIP (lerobot/pi0_base) | **Ready** | 412.4M vision tower params, loads via lerobot 0.4.4 |
| pi0.5 SigLIP (lerobot/pi05_base) | **Ready** | 412.4M vision tower params, loads via lerobot 0.4.4 |
| DINO-WM | **Not started** | Needs repo clone + checkpoint |

### UMD Dataset (Downloaded)
- **UMD+GT version** — 6 affordances (grasp, cut, scoop, contain, pound, wrap-grasp) + background
- Source: Google Drive via [AffKpNet repo](https://github.com/ivalab/AffKpNet)

---

## Resolved Issues

### Issue 1: lerobot pi0/pi0.5 Checkpoint Loading (RESOLVED 2026-03-26)

**Solution:** Installed lerobot 0.4.4 from PyPI + custom transformers fork (`fix/lerobot_openpi` branch).
```bash
pip install lerobot==0.4.4
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
```
- Import path: `lerobot.policies.pi0.modeling_pi0.PI0Policy` (pi0), `lerobot.policies.pi05.modeling_pi05.PI05Policy` (pi0.5)
- Vision tower path: `policy.model.paligemma_with_expert.paligemma.vision_tower` (SiglipVisionModel, 412.4M params)
- Known harmless warning: missing `embed_tokens.weight` key (language model embedding, not needed for vision)

---

## Active Issues

### Issue 2: DINO-WM Setup Not Started

Need to:
1. Clone https://github.com/mazpie/dino-wm
2. Find and download a trained checkpoint (PushT or similar)
3. Adapt the transition model loading code

### Issue 3: UMD Dataset Not Downloaded

The original UMD hosting (umiacs.umd.edu) returns 404. The dataset is available via Google Drive from the [AffKpNet repo](https://github.com/ivalab/AffKpNet).

**Manual download instructions:**
```bash
use_conda affordance
# Install gdown if needed
pip install gdown

# Download main dataset (7.46 GB)
gdown "1lWJDKyHILxOtMZ5nctxvY86igH5tFQoS" -O data/umd_dataset/UMD_GT.zip

# Download masks (18 MB)
gdown "1bB94rvWacpXF-Uo21bGEH8Ti2egSbU63" -O data/umd_dataset/UMD_GT_MASK.zip

# Download split files
gdown "1FGBrBhdbtEwcVWdJxMTSi1oaaq-RrE1g" -O data/umd_dataset/umd_gt_category_split.txt

# Extract
cd data/umd_dataset && unzip UMD_GT.zip && unzip UMD_GT_MASK.zip
```

**Note:** This is the UMD+GT version with **6 affordances** (grasp, cut, scoop, contain, pound, wrap-grasp). The original UMD has 7 (includes "support"), but that hosting is dead. Config/code may need updating from 7→6 affordances + background.

---

## Next Steps (In Priority Order)

1. **Download UMD dataset** — see manual instructions above
2. **Clone DINO-WM repo** and set up checkpoint
3. **Verify all 5 encoders** produce correct output shapes via `scripts/01_setup_encoders.py`
4. **Extract and cache features** via `scripts/02_extract_features.py`
5. Probing experiments (no training yet per user instruction)
