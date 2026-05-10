# Axis 2 — AGD20K Cross-Attention Probing

**Quickstart.** Run the pipeline from the repo root:

```bash
# Smoke tests (3 min each, no commitments)
python scripts/09_setup_flux.py --model schnell
python scripts/09b_setup_cosmos.py --system cosmos_predict2_v2w

# Full pilots (each ~45 min for Flux schnell, ~3 hrs for Cosmos V2W)
python scripts/10_run_interaction_probing.py --model schnell --max_per_category 30 --commit_every 100 --save_attention_maps
python scripts/10b_run_cosmos_probing.py --system cosmos_predict2_v2w --max_per_category 30 --commit_every 50 --save_attention_maps

# Analysis (CPU, ~30 sec)
python scripts/12_three_way_comparison.py
python scripts/13_qualitative_panel.py
python scripts/14_complexity_spectrum.py
```

## What this experiment measures

Verb-spatial binding in generative diffusion models on the AGD20K affordance dataset, via cross-attention extraction during denoising. Tests whether language-conditioned image/video diffusion models develop verb-anchored spatial priors comparable to what Zhang et al. (CVPR 2026) documented for FLUX.

## Systems probed

| System | Pipeline | Conditioning | Status |
|---|---|---|---|
| FLUX.1-schnell | `FluxPipeline` (4-step, no CFG) | text only | calibration baseline |
| FLUX.1-dev | `FluxPipeline` (20-step, CFG=3.5) | text only | optional publication-quality baseline |
| Cosmos-Predict2-2B-Video2World | `Cosmos2VideoToWorldPipeline` | text + first-frame image | H2b — base video diffusion |
| ~~Cosmos-Policy-ALOHA-Predict2-2B~~ | not a diffusers pipeline | text + multi-view + proprio | DEFERRED |

## Metrics

- **KLD ↓** — Kullback-Leibler divergence between predicted attention map and GT heatmap
- **SIM ↑** — Histogram intersection of normalized maps
- **NSS ↑** — Normalized Scanpath Saliency at top-20% GT fixations
- **peak_in_gt** — binary: does the predicted attention argmax fall in the GT functional region?

## Files in this folder

| Path | Purpose |
|---|---|
| `README.md` | quickstart + overview (this file) |
| `guide.md` | long-form research/methods documentation |
| `results.md` | result tables and headline findings (populated as data lands) |
| `PROTOCOL.md` | locked H2 protocol (predictions, statistical tests) |
| `configs/` | per-system YAML configs |
| `scripts/` | symlinks/pointers to the actual scripts in repo root |
| `src/` | symlinks/pointers to the actual code in repo root |
| `metadata/` | dataset splits and class definitions |

## Where the code actually lives (until full restructure)

Until we finalize the layout post-results, code lives at repo root:

| Conceptual location | Actual current location |
|---|---|
| `src/extractors/flux_attention.py` | `interaction/flux_attention.py` |
| `src/extractors/cosmos_attention.py` | `interaction/cosmos_attention.py` |
| `src/metrics/verb_spatial_binding.py` | `interaction/verb_spatial_binding.py` |
| `src/metrics/incremental_results.py` | `interaction/incremental_results.py` |
| `src/visualization/visualization.py` | `interaction/visualization.py` |
| `src/data/agd20k_dataset.py` | `data/agd20k_dataset.py` |
| `scripts/setup_flux.py` | `scripts/09_setup_flux.py` |
| `scripts/setup_cosmos.py` | `scripts/09b_setup_cosmos.py` |
| `scripts/run_flux.py` | `scripts/10_run_interaction_probing.py` |
| `scripts/run_cosmos.py` | `scripts/10b_run_cosmos_probing.py` |
| `scripts/comparison.py` | `scripts/12_three_way_comparison.py` |
| `scripts/qualitative.py` | `scripts/13_qualitative_panel.py` |
| `scripts/binary_metric.py` | `scripts/14_complexity_spectrum.py` |
| `scripts/diagnostic.py` | `scripts/00_diagnostic.py` |

Once the current Flux + Cosmos pilots finish on Colab, all of the above will be moved into this folder.

## Compute

Designed for Colab Pro A100 (40 GB). Local RTX 5060 (8 GB) is insufficient for Flux/Cosmos inference.
