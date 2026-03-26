# Affordance Probing Project — Status Report
**Date:** 2026-03-25

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
- **Other deps:** transformers, scikit-learn, numpy, scipy, matplotlib, seaborn, pillow, opencv-python, wandb, tqdm

### Models Downloaded (2 of 4)

| Model | Status | Notes |
|-------|--------|-------|
| Raw SigLIP (google/siglip-so400m-patch14-384) | **Downloaded** | 428.2M params |
| DINOv2 (facebook/dinov2-base) | **Downloaded** | 86.6M params |
| pi0 SigLIP (lerobot/pi0_base) | **BLOCKED** | See issues below |
| pi0.5 SigLIP (lerobot/pi05_base) | **BLOCKED** | See issues below |
| DINO-WM | **Not started** | Needs repo clone + checkpoint |

### UMD Dataset
- **Not yet downloaded** — download script written but not executed

---

## Active Issues

### Issue 1: lerobot pi0/pi0.5 Checkpoint Loading Failure (BLOCKING)

**Problem:** `PI0Policy.from_pretrained('lerobot/pi0_base')` fails with:
```
draccus.utils.DecodingError: The fields `paligemma_variant`, `action_expert_variant`,
`dtype`, `num_inference_steps`, ... are not valid for PI0Config
```

**Root Cause:** The `lerobot/pi0_base` checkpoint on HuggingFace was re-uploaded with a v0.4.x config schema, but lerobot 0.3.2 (latest on PyPI) uses an older PI0Config that doesn't have these fields. The API also changed — `lerobot.common.policies.pi0` moved to `lerobot.policies.pi0`.

**What we've tried:**
- lerobot 0.3.2 from pip: config schema mismatch
- draccus version pinning (0.10.0, 0.11.5): doesn't help
- Upgrading lerobot: pip only has 0.3.2

**Possible solutions (not yet attempted):**
1. **Install lerobot from GitHub main branch** (`pip install git+https://github.com/huggingface/lerobot.git`) — should have the v0.4.x PI0Config that matches the checkpoint
2. **Manual weight extraction** — download the safetensors file directly and filter for vision encoder keys:
   ```python
   from safetensors.torch import load_file
   state_dict = load_file("path/to/model.safetensors")
   vision_keys = {k: v for k, v in state_dict.items()
                  if "paligemma" in k and "vision" in k}
   ```
3. **Use `lerobot/pi0_old`** — the old checkpoint format that works with lerobot 0.3.2 (but may have different weights)
4. **Use openpi directly** — clone the official Physical Intelligence openpi repo and use their JAX->PyTorch conversion

**User preference:** Do NOT use local lerobot installation from other projects.

### Issue 2: DINO-WM Setup Not Started

Need to:
1. Clone https://github.com/mazpie/dino-wm
2. Find and download a trained checkpoint (PushT or similar)
3. Adapt the transition model loading code

### Issue 3: UMD Dataset Not Downloaded

The download script is written but the UMD dataset URL structure needs verification. The dataset may have changed hosting.

---

## Next Steps (In Priority Order)

1. **Resolve pi0/pi0.5 loading** — try `pip install git+https://github.com/huggingface/lerobot.git` or manual safetensors extraction
2. **Download UMD dataset** — run `data/download_umd.py`
3. **Clone DINO-WM repo** and set up checkpoint
4. **Verify all 5 encoders** produce correct output shapes via `scripts/01_setup_encoders.py`
5. **Extract and cache features** via `scripts/02_extract_features.py`
6. Probing experiments (no training yet per user instruction)
