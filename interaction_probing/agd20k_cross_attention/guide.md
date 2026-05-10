# Axis 2 — Methods and Implementation Guide

This document is the long-form companion to `README.md`. It explains *why* each design choice was made, references the relevant prior work, and documents the extraction recipe in enough detail for reproduction.

## 1. Research question

Do language-conditioned diffusion models develop verb-spatial binding — a representational property where verb tokens activate spatially specific regions of an image (or generated frame) that correspond to where those verbs would be performed?

This question was first established by Zhang et al. (arXiv:2602.20501, CVPR 2026) for the text-to-image model FLUX, who showed:
- Verb tokens like "hold", "cut", "pour" activate spatially specific cross-attention maps
- The activated regions correspond to functional parts (handle, blade, rim)
- This binding persists even when image generation fails visually — it is internal/representational, not output-quality dependent

We extend this question to **video diffusion** (Cosmos-Predict2-2B-Video2World) — does a model trained on general video, never on robot demonstrations, also develop verb-spatial binding?

## 2. Hypotheses (locked)

| ID | Claim | Prediction |
|---|---|---|
| H2a | FLUX shows strong verb-spatial binding | KLD ≤ 1.7, SIM ≥ 0.30, NSS ≥ 1.0 on AGD20K overall |
| H2b | Cosmos-Predict2 V2W shows verb-spatial binding | median NSS > 0.5, p < 0.05 vs uniform null |

H2c (Cosmos Policy > Cosmos Predict2) is **deferred** because Cosmos-Policy-ALOHA-Predict2-2B is not a diffusers pipeline; it requires NVIDIA's custom `cosmos-policy` package with action heads + proprioception input.

Predictions are locked in `PROTOCOL.md` (committed before any experiment runs).

## 3. Dataset — AGD20K

Source: Luo et al., "Learning Affordance Grounding from Exocentric Images" (CVPR 2022). Mirror: Reagan1311/LOCATE Drive (verified 2026-05-10).

- 36 affordance categories
- ~1710 egocentric images with per-pixel affordance heatmaps
- Prompt template: `"a person {verb} a {object}"` (e.g. "a person cutting a knife")
- GT heatmaps come from `Seen/testset/GT/<affordance>/<object>/*.png` (not adjacent to images — separate `GT/` tree)

## 4. Cross-attention extraction

### Flux (text-to-image MMDiT)

Flux uses joint attention: image and text tokens are concatenated along the sequence dim, then a single joint attention block computes Q/K/V across the concatenated sequence. The image-query × text-key sub-matrix gives the verb→image-region map.

We register a custom `FluxCrossAttnRecorder` (mirroring diffusers' `FluxAttnProcessor`) on every `FluxAttention` module:
- Compute Q, K, V projections; apply RMSNorm and RoPE per diffusers spec
- Compute attention probs explicitly (matmul + softmax, NOT scaled-dot-product) so we can capture the post-softmax (B, H, S, S) tensor
- Slice `[..., txt_len:txt_len+img_len, :txt_len]` to get image-query × text-key cross-attention
- Head-average + store on CPU in float16 (memory-bounded)
- For `pre_only=True` blocks (no `to_out`), return the joint tensor unprojected — the block handles projection itself

### Cosmos (CosmosTransformer3DModel)

Cosmos uses traditional cross-attention: each transformer block has `attn1` (self-attention on image stream) and `attn2` (cross-attention from image queries to text keys). Cleaner architecture for our purposes.

We register a `CosmosCrossAttnRecorder` on every `attn2` module:
- Project Q from image stream, K/V from encoder hidden states
- Apply norms (`norm_q`, `norm_k`)
- Reshape to (B, heads, len, head_d), compute attention probs
- Head-average + store on CPU in float16
- The Cosmos pipeline's `Cosmos2VideoToWorldPipeline` is bypassed for `safety_checker` (requires the `cosmos_guardrail` package that fails to install cleanly on Colab)

## 5. Aggregation: from layer-wise probs to one heatmap per (image, verb)

For each (image, verb) pair:

1. Run the diffusion pipeline (4-step schnell or 20-step dev for Flux; 12-step for Cosmos V2W)
2. During each denoising step, each transformer block's recorder appends `(layer, attn_probs)` to a shared store
3. For each verb token's position in the T5 tokenization, slice the relevant column from each stored attention matrix
4. Reshape the image-token dim to spatial grid (`latent_h × latent_w` for Flux; `latent_t × latent_h × latent_w → spatial` for Cosmos, averaged over temporal)
5. Average across denoising timesteps and across transformer blocks → one (H, W) heatmap per verb

Each heatmap is min-max normalized to [0, 1] before metric computation.

## 6. Metrics

All defined in `src/metrics/verb_spatial_binding.py`. Resize predicted map to GT resolution before computing.

- **KLD ↓**: `Σ_i gt_i * log((gt_i + ε) / (pred_i + ε))` after each map is normalized to sum=1
- **SIM ↑**: `Σ_i min(pred_i, gt_i)` after normalization
- **NSS ↑**: z-score-normalize the predicted map, then `mean(predicted_z at top-20% GT pixels)`
- **peak_in_gt** ∈ {0, 1}: binary — does `argmax(pred_map)` fall inside the binarised GT?

The first three are distributional metrics (sensitive to map shape and noise). The binary peak metric is a coarser but more interpretable check that complements them.

## 7. Statistical tests

In `scripts/12_three_way_comparison.py`:

- **H2a verdict:** does Flux's overall mean (KLD, SIM, NSS) clear the locked thresholds? Bootstrap 95% CI from per-sample resampling (n=1000).
- **H2b verdict:** is Cosmos's median NSS > 0.5? Bootstrap test against the locked threshold.
- **(Future) H2 paired comparison:** Wilcoxon signed-rank on paired (Flux, Cosmos) per-sample deltas, stratified by affordance category.

## 8. Resilience features (Colab durability)

Implemented in scripts 10 and 10b:

- **Incremental CSV writes** via `IncrementalCSVWriter`: every per-sample result is `flush()` + `os.fsync()`-ed before moving to the next sample
- **`--resume`** (default on): skip sample_ids already present in the CSV
- **`--commit_every N`**: periodic `git add + commit + push origin HEAD` every N processed samples
- **Drive-symlinked results**: `./results` is a symlink to `/content/drive/MyDrive/VLA-affordance-results` so files survive runtime disconnects
- **Persistent HF cache** on Drive: model downloads survive disconnects
- **`scripts/00_diagnostic.py`**: 5-second pre-flight check, validates imports + dataset detection before pilots burn GPU time

## 9. Known limitations

- **Schnell vs dev:** FLUX.1-schnell uses 4 denoising steps with no classifier-free guidance. Maps are noisier than dev's 20-step + CFG=3.5. Numbers are not directly comparable to Zhang et al.'s published Flux-dev numbers; we use schnell as a fast calibration baseline.
- **Cosmos Policy deferred:** see above.
- **Single-stream Flux blocks:** our `FluxCrossAttnRecorder` only records image_q × text_k on double-stream blocks where image and text are still in separate sequences. Single-stream / `pre_only` blocks return the joint tensor unprojected and are skipped for recording (a minority of blocks).
- **Cosmos VAE temporal axis:** we collapse the temporal latent dimension by averaging. Verb-spatial binding is a spatial property; averaging is a defensible choice but loses any temporal signal.
- **No image-conditioning for Flux:** we feed only text. The model generates an image from scratch, and the attention is on that generated image, not on the AGD20K input image. (Switching to FLUX.1-Kontext-dev would give image-conditioned extraction; this is a future direction.)
