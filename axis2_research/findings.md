# Axis 2 Findings — Living Document

**Last updated:** 2026-05-10
**Status:** bootstrap

## Research Question

Do language-conditioned diffusion world models (Cosmos Predict2, Cosmos Policy) develop verb-spatial binding analogous to Flux, and does manipulation fine-tuning amplify this binding?

## Current Understanding

(Pre-experiment) Three claims to test:

1. **Flux baseline reproduces Zhang et al.** — verb tokens activate spatially specific regions tied to functional object parts. Expected KLD ≈ 1.49, SIM ≈ 0.33, NSS ≈ 1.09 on AGD20K.
2. **Cosmos Predict2 (general video diffusion) shows binding** because it shares the same T5-XXL cross-attention conditioning mechanism as Flux.
3. **Cosmos Policy (manipulation-fine-tuned) shows STRONGER binding** than Cosmos Predict2, because manipulation demos contain verbs engaging with parts.

If H2c is supported, this connects to Axis 1's finding that VLAs *destroy* geometric affordance: world model policies and VLAs would have complementary affordance profiles, motivating dual-stream architectures.

## Patterns and Insights

(empty — populated after first outer loop)

## Lessons and Constraints

- Compute split: agent has only RTX 5060 (8 GB) — insufficient for Flux 12B or Cosmos 2B. **All inference must run on user's Colab Pro A100 (40 GB)**. Agent's role is orchestration, code, and analysis only.
- **Cosmos Policy IS publicly available** on HuggingFace as `nvidia/Cosmos-Policy-ALOHA-Predict2-2B`. Fine-tuned from `Cosmos-Predict2-2B-Video2World` (NOT Text2Image). For the cleanest H2b vs H2c comparison both must use the **Video2World** base.
- **Cosmos text encoder is T5-11B** (not T5-XXL as the proposal stated). Functionally similar (both ~11B frozen T5), conceptually identical to Flux's T5-XXL cross-attention.
- **Cosmos cross-attention layers are named `attn2`** in `CosmosTransformer3DModel` (per diffusers source). Easy custom AttentionProcessor registration target.
- Cosmos2VideoToWorld defaults: 704×1280 res, 93 frames, num_inference_steps=35, max_sequence_length=512. At full res spatial tokens are huge — we will downsize to 480×704 (still divisible by patch), 1 conditional frame, and reduce inference steps to ~12 to keep VRAM/wall-clock manageable.
- AGD20K heatmap GT is normalized to a probability distribution (sum=1) by the loader.
- **attention-map-diffusers does NOT support Cosmos**. We must write a custom AttentionProcessor and register it on every `attn2` module.
- Cosmos Policy takes images + proprioception. For AGD20K (no proprio), we pass zero-vectors for proprio — should not bias verb→latent cross-attention since cross-attention only sees text↔latent.
- Latent grid for Flux at 512×512: nominally 32×32 (16x patch downsampling). Reshape carefully — Flux packs 2×2 patches.

## Open Questions

- Is Cosmos Policy publicly downloadable, or only behind NVIDIA NGC auth?
- For Cosmos (video diffusion), what is the natural "single-image" probing protocol? Conditioning on a still frame? Generating a one-frame video?
- How do we make Flux ↔ Cosmos cross-attention scales comparable? (different latent grid sizes, different num text tokens)
- Should we control for image generation quality? If Cosmos generates worse images than Flux, NSS may be lower for trivial reasons unrelated to interaction priors.

## Trajectory Log

(experiment-by-experiment results populated here)
