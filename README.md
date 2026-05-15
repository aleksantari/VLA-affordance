# Affordance in the Wild

**Probing visual affordance across two robotic visual pipelines:** Vision–Language–Action (VLA) encoders and language-conditioned generative diffusion models.

[![branch](https://img.shields.io/badge/branch-nj--features-blue)](https://github.com/aleksantari/VLA-affordance/tree/nj-features) [![status](https://img.shields.io/badge/Phase%201-complete-success)](report_phase1/phase1_final_report.pdf) [![Phase 2](https://img.shields.io/badge/Phase%202-in%20progress-orange)](#status)

---

## Overview

Modern robot policies route visual information through one of two architectural paradigms that handle language and vision in opposite ways:

| Paradigm | Direction | Example | Implication for affordance |
|---|---|---|---|
| **Vision–Language–Action (VLA)** | image → action tokens | $\pi_0$, $\pi_{0.5}$, OpenVLA | Visual encoder is fine-tuned toward action prediction. May lose fine-grained spatial structure. |
| **Generative world-model policy** | language → generated image/video | Cosmos-Predict2, Flux | Text drives spatial generation through cross-attention. May develop *verb-spatial binding* implicitly. |

This project asks: *do these two pipelines preserve, develop, or destroy affordance perception in different ways, and what does that imply for designing the next generation of robot policies?*

We pursue **two complementary axes**:

- **Axis 1 — Geometric Affordance Probing.** Linear probing of the SigLIP-So400m vision encoder along its VLA fine-tuning trajectory (raw → PaliGemma → $\pi_0$ → $\pi_{0.5}$) on the UMD Part-Affordance Dataset at two evaluation resolutions.
- **Axis 2 — Interaction Affordance Probing.** Extraction of verb-conditioned cross-attention maps from generative diffusion models (FLUX.1 and Cosmos-Predict2) on AGD20K, scored against ground-truth affordance heatmaps.

Both axes share methodology with the protocol of Zhang et al. (CVPR 2026, arXiv:2602.20501).

---

## Status

| Component | Status | Sample size |
|---|---|---|
| Axis 1: Linear probe across 6 encoders × 2 resolutions × 4 seeds | **Complete** ([Alex's repo](https://github.com/aleksantari/Probing_Bridging_Affordance/tree/VLA-affordance)) | 48 cells |
| Axis 2: FLUX.1 (schnell + dev) on AGD20K | **Complete** | n = 1675 |
| Axis 2: Cosmos-Predict2-2B-Video2World | In progress (Colab A100 pilot scheduled) | — |
| Phase 1 report (4 pages, both axes) | **Submitted** ([PDF](report_phase1/phase1_final_report.pdf)) | — |
| Phase 2 report + paired Flux/Cosmos comparison | Pending Cosmos pilot completion | — |

---

## Key findings (Axis 2, FLUX.1, n = 1675)

| Metric | Result | Uniform null | Zhang et al. (published) |
|---|---|---|---|
| KLD ↓ | **1.86** | ~2.7 | 1.49 |
| SIM ↑ | **0.25** | ~0.15 | 0.33 |
| NSS ↑ | **+0.43** | 0.00 | 1.09 |

**Three findings worth flagging:**

1. **Binding is real.** All three metrics depart from the uniform-attention null in the expected direction. FLUX's cross-attention encodes verb-conditioned spatial structure aligned with affordance ground truth, without ever being trained for affordance.

2. **Schnell beats Dev** (counterintuitive). FLUX.1-schnell (4 inference steps, no classifier-free guidance) significantly beats FLUX.1-dev (20 steps, CFG=3.5) on KLD ($p < 10^{-4}$) and SIM ($p < 10^{-4}$); NSS is statistically tied ($p = 0.16$). Five times more inference compute does **not** improve attention-vs-GT alignment under our pipeline. CFG appears to amplify the model's prompt-conditioned interpretation rather than what is in the image. *Methodological implication:* cheap configurations are valid measurement tools for cross-attention probing.

3. **Manipulation verbs bind *weaker* than non-manipulation verbs** (also counterintuitive). Stratifying by verb type, manipulation verbs (hold, push, lift, pour, ...) have NSS = +0.349; postural / contact verbs (lie\_on, talk\_on, boxing, kick, ...) have NSS = +0.509 (Mann-Whitney $p = 3 \times 10^{-9}$). The signal is shaped by ground-truth-region geometry, not by whether a verb names a manipulation primitive.

See [the Phase 1 report](report_phase1/phase1_final_report.pdf) and [`interaction_probing/agd20k_cross_attention/results.md`](interaction_probing/agd20k_cross_attention/results.md) for the full analysis with statistical tests, per-affordance breakdown, and figures.

---

## Repository structure

```
.
├── interaction_probing/agd20k_cross_attention/   ← Axis 2 self-contained experiment
│   ├── README.md                                   quickstart + system table
│   ├── guide.md                                    long-form methods rationale
│   ├── results.md                                  result tables + paper-narrative
│   ├── PROTOCOL.md                                 locked H2 pre-registration
│   ├── configs/                                    per-system YAMLs (flux_schnell, flux_dev, cosmos)
│   ├── scripts/README.md                           pointers to scripts/* in repo root
│   ├── src/README.md                               pointers to interaction/* in repo root
│   └── metadata/
│
├── interaction/                                  ← Axis 2 implementation
│   ├── flux_attention.py                           custom AttentionProcessor (no attention-map-diffusers)
│   ├── cosmos_attention.py                         Cosmos cross-attention recorder
│   ├── verb_spatial_binding.py                     KLD / SIM / NSS / peak-in-GT metrics
│   ├── incremental_results.py                      append-only resumable CSV writer
│   └── visualization.py                            overlays + comparison grids
│
├── data/                                          dataset loaders
│   ├── umd_dataset.py                              UMD Part-Affordance loader (Axis 1)
│   ├── agd20k_dataset.py                           AGD20K loader with parallel-GT search (Axis 2)
│   ├── download_umd.py
│   └── download_agd20k.py                          --from_drive / --from_zip support for Colab
│
├── encoders/                                      Axis 1 encoder zoo
│   ├── raw_siglip.py, paligemma_siglip.py, pi0_siglip.py, pi05_siglip.py
│   ├── dinov2.py, dino_wm.py
│   ├── multilayer.py                               4-layer feature fusion (Zhang et al. protocol)
│   └── feature_extractor.py                        unified extractor registry
│
├── probing/                                       Axis 1 probing methods
│   ├── linear_probe.py                             BatchNorm + 1×1 Conv head
│   ├── pca_analysis.py
│   ├── depth_normal.py                             DPT augmentation
│   ├── weight_divergence.py
│   └── cosine_similarity.py
│
├── evaluation/                                    shared metrics + visualization
│
├── scripts/                                       all CLI entry points
│   ├── 00_diagnostic.py                            5-sec pre-flight (imports + AGD20K detection)
│   ├── 01_setup_encoders.py … 08_generate_report.py   Axis 1 pipeline
│   ├── 09_setup_flux.py / 09b_setup_cosmos.py      smoke tests
│   ├── 10_run_interaction_probing.py               Flux probing (resumable, periodic-push)
│   ├── 10b_run_cosmos_probing.py                   Cosmos probing
│   ├── 11_generate_axis2_report.py                 per-system report
│   ├── 12_three_way_comparison.py                  paired stats + figures
│   ├── 13_qualitative_panel.py                     paper Figure 2 scaffold
│   ├── 14_complexity_spectrum.py                   binary peak-in-GT analysis
│   └── dev_quick_analysis.py / dev_synthetic_results_test.py
│
├── notebooks/
│   ├── axis2_unified_colab.ipynb                   resilient Colab notebook (smoke + pilots + push)
│   └── axis2_cosmos_optimized.ipynb                GPU-bound optimized Cosmos-only run
│
├── configs/probing_config.yaml                    central config (Axis 1 + Axis 2)
├── requirements.txt + requirements_axis2.txt
│
├── results/                                       experimental record
│   ├── tables/
│   │   ├── axis2_per_sample.csv                    1675-row Flux pilot data
│   │   ├── axis2_three_way_summary.csv             per-system + 95% CIs
│   │   └── axis2_hypothesis_tests.json             H2a verdict
│   └── figures/axis2/                              comparison + per-affordance bar charts
│
├── report_phase1/                                 Phase 1 final report
│   ├── phase1_final_report.tex                     LaTeX source
│   ├── phase1_final_report.pdf                     compiled, 4 pages
│   └── fig{1,2,3}*.png                             figures
│
├── axis2_research/                                autoresearch workspace
│   ├── research-state.yaml + findings.md + research-log.md
│   ├── experiments/H2_protocol.md                  locked H2 protocol
│   ├── literature/cosmos_predict2.md
│   └── to_human/                                   HTML progress reports
│
├── docs/                                          design notes + Probe3D protocol alignment
├── CLAUDE.md                                      AI agent project instructions
└── README.md                                      (this file)
```

---

## Quick start

### Environment

```bash
# Conda environment (Python 3.11, PyTorch 2.11.0, CUDA 13.0 for RTX 5090)
conda create -n affordance python=3.11
conda activate affordance
pip install -e .
pip install -r requirements.txt
pip install -r requirements_axis2.txt   # Axis 2 only (Flux + Cosmos)
```

### Axis 1 — Linear probing on UMD (local, RTX 5060 sufficient)

```bash
# Pipeline (8 steps, ~hours total)
python scripts/01_setup_encoders.py             # verify encoder shapes
python scripts/02_extract_features.py           # cache fused features (float16)
python scripts/03_run_linear_probing.py --epochs 50   # train probes, mIoU evaluation
python scripts/04_run_pca_analysis.py           # PCA visualizations
python scripts/05_extract_depth_normal.py       # DPT depth/normals
python scripts/06_run_depth_augmentation.py     # depth-augmentation deltas
python scripts/07_weight_divergence.py          # SigLIP weight comparison across stages
python scripts/08_generate_report.py            # final report aggregation
```

### Axis 2 — Cross-attention probing on AGD20K (requires A100, run on Colab)

```bash
# Pre-flight diagnostic (5 sec, no GPU)
python scripts/00_diagnostic.py

# Flux smoke test (~3 min on A100)
python scripts/09_setup_flux.py --model schnell

# Flux pilot (~45 min on A100; --commit_every pushes incrementally)
python scripts/10_run_interaction_probing.py \
    --model schnell \
    --max_per_category 30 \
    --commit_every 100 \
    --save_attention_maps

# Cosmos smoke test (~3 min on A100)
python scripts/09b_setup_cosmos.py --system cosmos_predict2_v2w

# Cosmos pilot (~3 hr on A100)
python scripts/10b_run_cosmos_probing.py \
    --system cosmos_predict2_v2w \
    --max_per_category 30 \
    --commit_every 50

# Combined analysis (CPU, ~30 sec)
python scripts/12_three_way_comparison.py       # per-system stats + figures
python scripts/13_qualitative_panel.py          # image | GT | per-system attention overlay
python scripts/14_complexity_spectrum.py        # binary peak-in-GT
```

The `notebooks/axis2_unified_colab.ipynb` runs the entire pipeline end-to-end on Colab Pro A100 with resilience features (incremental CSV writes, persistent HuggingFace cache on Drive, periodic git push, optional idle keepalive). See its inline documentation for setup.

---

## Methodology highlights

### Axis 1 — Linear probing protocol (Zhang et al. / Probe3D compatible)

- Encoder is **frozen**. No fine-tuning during probing.
- Four equally-spaced intermediate transformer layers are hooked, fused by bilinear resize + channel concatenation.
- Probe head: BatchNorm2d → 1×1 Conv → 7-class affordance logits. Forward pass: features upsampled 4× *before* normalization (matches Probe3D, El Banani et al. CVPR 2024).
- Optimizer: AdamW, $\text{lr}=10^{-3}$, weight decay $5 \times 10^{-2}$, cosine LR schedule + 10% linear warmup.
- mIoU computed over 7 affordance classes (excluding background class 0), matching Zhang et al.
- Pipeline validation: DINOv2-B at UMD-native resolution reproduces $0.666$ mIoU vs the $0.670$ reported in Zhang et al.
- Two resolutions probed: $224 \times 224$ (VLA operating resolution; $16 \times 16$ patch grid) and UMD-native ${\sim}480 \times 640$ ($35 \times 46$ patch grid).

### Axis 2 — Cross-attention extraction

- Custom `FluxCrossAttnRecorder` and `CosmosCrossAttnRecorder` replace third-party libraries (`attention-map-diffusers` is brittle against modern diffusers signature changes). Implementations mirror diffusers' default `AttentionProcessor` math but explicitly compute the post-softmax attention probabilities, which scaled-dot-product attention does not expose.
- Image-query × text-key cross-attention is sliced per verb token, head-averaged, then aggregated across denoising timesteps and transformer blocks.
- Three saliency metrics computed against AGD20K ground-truth heatmaps:
    - **KLD** (Kullback-Leibler divergence; lower is better)
    - **SIM** (histogram intersection of normalized maps; higher is better)
    - **NSS** (Normalized Scanpath Saliency at top-20% GT pixels; higher is better)
- A complementary binary metric, **peak-in-GT-region**, is computed as a coarser sanity check.
- Statistical tests: Mann-Whitney U for independent-sample comparisons (FLUX-schnell vs FLUX-dev); paired Wilcoxon signed-rank for paired (FLUX, Cosmos) per-sample deltas (planned for Phase 2).
- Hypotheses (KLD ≤ 1.7, SIM ≥ 0.30, NSS ≥ 1.0 for H2a; binding above null for H2b) are **pre-registered** in [`interaction_probing/agd20k_cross_attention/PROTOCOL.md`](interaction_probing/agd20k_cross_attention/PROTOCOL.md) and committed to the repository **before** any experimental data was collected.

### Axis 2 — Resilience features (Colab durability)

- Per-sample CSV is `flush() + os.fsync()`-ed after every row. `--resume` (default ON) skips sample IDs already present in the CSV.
- `--commit_every N` performs `git add + commit + push origin HEAD` every N processed samples — progress lives on GitHub even if the runtime disconnects.
- `./results/` symlinks to Drive, so files survive runtime wipes.
- Persistent HuggingFace cache redirected to Drive (`HF_HOME=/content/drive/.../hf_cache`) so model weights download once, then reuse across sessions.
- `scripts/00_diagnostic.py` validates imports and dataset detection in 5 seconds (no GPU) before pilots burn compute time.

---

## Reproducing Phase 1 results

All numerical results in [`report_phase1/phase1_final_report.pdf`](report_phase1/phase1_final_report.pdf) are reproducible from this branch:

| Result | Reproduce by |
|---|---|
| Axis 1 mIoU table (Fig. 1, 2) | See [`aleksantari/Probing_Bridging_Affordance/geometry_probing/umd_linear_probing/`](https://github.com/aleksantari/Probing_Bridging_Affordance/tree/VLA-affordance/geometry_probing/umd_linear_probing) |
| Axis 2 FLUX overall metrics | `python scripts/dev_quick_analysis.py` against [`results/tables/axis2_per_sample.csv`](results/tables/axis2_per_sample.csv) |
| Axis 2 per-affordance breakdown (Fig. 3) | `python scripts/12_three_way_comparison.py` regenerates [`results/figures/axis2/`](results/figures/axis2/) |
| H2a verdict JSON | `results/tables/axis2_hypothesis_tests.json` |

---

## Phase 1 report

The 4-page Phase 1 report combining both axes is available at [`report_phase1/phase1_final_report.pdf`](report_phase1/phase1_final_report.pdf). Source LaTeX + figures + pre-compiled PDF are zipped at the repo root as `phase1_final_report_overleaf.zip` for collaborators who want to edit on Overleaf.

---

## Authors

- **Aleks Antari** — Axis 1 design and execution. Project framing, vision-tower extraction from PaliGemma / $\pi_0$ / $\pi_{0.5}$ checkpoints, multi-seed linear probing across both evaluation resolutions, PCA feature-space analysis.
- **Nitik Jain** — Axis 2 design and execution. Custom Flux and Cosmos cross-attention recorders (replacing the brittle `attention-map-diffusers` dependency), AGD20K loader with parallel-GT directory search, resilient Colab pipeline (incremental CSV / resume / periodic git push / persistent caches), FLUX-schnell + FLUX-dev full pilot ($n = 1675$), and per-affordance statistical analysis.

This branch (`nj-features`) primarily reflects Nitik's Axis 2 contributions; the canonical Axis 1 code lives in [`aleksantari/Probing_Bridging_Affordance`](https://github.com/aleksantari/Probing_Bridging_Affordance/tree/VLA-affordance) and Axis 1 results are imported into this report.

---

## References

1. Physical Intelligence. *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control.* 2024.
2. Physical Intelligence. *$\pi_{0.5}$: A VLA with Open-World Generalization via Knowledge Insulation.* 2025.
3. Zhai et al. *Sigmoid Loss for Language Image Pre-Training (SigLIP).* ICCV 2023.
4. Beyer et al. *PaliGemma: A Versatile 3B VLM for Transfer.* 2024.
5. Zhang et al. *Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models.* arXiv:2602.20501, CVPR 2026.
6. Myers, Teo, Fermüller, Aloimonos. *Affordance Detection of Tool Parts from Geometric Features (UMD).* ICRA 2015.
7. Black Forest Labs. *FLUX.1: An Open Image Generation Model.* 2024.
8. Luo et al. *Learning Affordance Grounding from Exocentric Images (AGD20K).* CVPR 2022.
9. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
10. Fu et al. *Hidden in Plain Sight: VLMs Overlook Their Visual Representations.* COLM 2025.
11. El Banani et al. *Probing the 3D Awareness of Visual Foundation Models (Probe3D).* CVPR 2024.
12. NVIDIA. *Cosmos World Foundation Model Platform for Physical AI.* arXiv:2501.03575, 2025.

---

## License

Code in this repository is released for academic and research purposes. See individual file headers for any third-party code attributions.

## Citation

If this work is useful in your research, please cite the Phase 1 report:

```bibtex
@techreport{antari_jain_2026_affordance,
  author      = {Antari, Aleks and Jain, Nitik},
  title       = {Probing Visual Affordance Across Two Robotic Visual Pipelines: VLA Encoders and Generative Diffusion Models},
  institution = {Johns Hopkins University, EN.601.495 / 695 Spring 2026},
  year        = {2026},
  type        = {Phase 1 Final Report},
  url         = {https://github.com/aleksantari/VLA-affordance/blob/nj-features/report_phase1/phase1_final_report.pdf}
}
```
