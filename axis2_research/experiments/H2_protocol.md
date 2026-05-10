# Axis 2 — Locked Protocol (H2a / H2b / H2c)

**Locked:** 2026-05-10. Any deviation requires explicit log entry in research-log.md.

## Systems Under Test

| ID | Model | HF ID | Pipeline | Resolution | Steps |
|---|---|---|---|---|---|
| `flux_schnell` | FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | `FluxPipeline` | 512×512 | 4 |
| `flux_dev` | FLUX.1-dev (final) | `black-forest-labs/FLUX.1-dev` | `FluxPipeline` | 512×512 | 20 |
| `cosmos_predict2_v2w` | Cosmos-Predict2-2B-Video2World | `nvidia/Cosmos-Predict2-2B-Video2World` | `Cosmos2VideoToWorldPipeline` | 480×704, 1 frame | 12 |
| `cosmos_policy` | Cosmos-Policy-ALOHA-Predict2-2B | `nvidia/Cosmos-Policy-ALOHA-Predict2-2B` | custom (cosmos-policy repo or low-level forward) | 224×224 multi-view | model default |

## Dataset

- **AGD20K egocentric split**, 36 affordance categories
- Per-category cap: 30 samples for `_pilot` runs; 100+ samples for `_final`
- Prompt template: `"a person {verb} a {object}"`

## Hypotheses (locked predictions)

### H2a — Flux replicates Zhang et al.
- **Prediction:** On AGD20K overall, Flux (dev) achieves KLD ≤ 1.7, SIM ≥ 0.30, NSS ≥ 1.0
- **Test:** single-system summary statistics with 95% CI from per-sample bootstrap
- **Outcome:** confirmed if all three thresholds met; refuted otherwise (would imply our pipeline differs from Zhang)

### H2b — Cosmos Predict2 (V2W) shows verb-spatial binding
- **Prediction:** NSS_predict2 > 0.5 (well above blind baseline NSS≈0)
- **Test:** one-sided Wilcoxon signed-rank against per-sample blind baseline (uniform attention)
- **Outcome:** confirmed if median NSS > 0.5 AND p < 0.05 vs uniform; refuted otherwise

### H2c — Cosmos Policy > Cosmos Predict2 in binding
- **Prediction:** NSS_policy > NSS_predict2 by ≥ 10% relative AND/OR SIM_policy > SIM_predict2 AND KLD_policy < KLD_predict2 — assessed jointly
- **Test:** paired Wilcoxon signed-rank on per-sample (NSS_policy − NSS_predict2)
- **Stratified analysis:** manipulation-relevant verbs {hold, pick_up, pour, push, lift} vs others
- **Outcome:** confirmed if Wilcoxon p < 0.05 AND median delta > 0; partially confirmed if effect only in manipulation verbs

## Metric definitions (frozen)

- KLD: lower is better, computed on flattened normalized maps (sum→1)
- SIM: higher is better, histogram intersection on normalized maps
- NSS: higher is better, GT fixations = top 20% of per-image GT heatmap pixels
- All metrics computed at GT resolution (512×512 after upsampling pred map)

## Cross-attention extraction (frozen)

Per-system, the verb-attention map is the average of:
- Across all `attn2` modules (cross-attention blocks only)
- Across all denoising timesteps (NOT a single chosen step)
- Across all attention heads
- Across all token positions matching any verb root for that affordance

The result is reshaped to the spatial latent grid, then upsampled bilinearly to GT resolution and normalized to [0, 1] before metric computation.

## Compute & timing budget

- Pilot (30 samples × 36 categories = 1080 samples): ~2-4 hours per system on A100
- Final (100+ samples × 36 = 3600+ samples): ~6-10 hours per system on A100
- Run on Google Colab Pro A100, results committed back as CSV/JSON to repo

## Sanity checks (pre-experiment)

1. KLD(GT, GT) ≈ 0, SIM(GT, GT) ≈ 1 — already verified in script 09
2. Verb token detection finds non-empty indices for ≥ 95% of (prompt, verb) pairs across systems
3. Generated image quality is recognizable (otherwise attention may be junk)
4. Attention map normalization preserves ordering: cross-attention summed over verb tokens is non-negative and not constant

## What success looks like (figures table)

- **Table 1:** Overall and per-affordance KLD/SIM/NSS for all 4 systems
- **Figure 1:** Bar chart with 95% CIs comparing the three primary systems on NSS
- **Figure 2:** 4-row qualitative panel (image + GT + Flux pred + Cosmos pred + Policy pred) for 8 representative affordances
- **Figure 3:** Stratified comparison — manipulation verbs vs other verbs, Cosmos Policy vs Predict2

## Failure modes & mitigations

- **Cosmos image quality too poor for meaningful attention** → still extract and report; may indicate attention is not bound to recognizable parts (a finding, not a bug)
- **Cosmos Policy proprio unavailable** → pass zeros; verify cross-attention does not collapse
- **VRAM exceeded on A100** → reduce resolution further; document the constraint
- **Cosmos `attn2` not found** → fall back to all attention modules and slice text-vs-spatial manually post-hoc
