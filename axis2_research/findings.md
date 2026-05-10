# Axis 2 Findings — Living Document

**Last updated:** 2026-05-10
**Status:** bootstrap

## Research Question

Do language-conditioned diffusion world models (Cosmos Predict2, Cosmos Policy) develop verb-spatial binding analogous to Flux, and does manipulation fine-tuning amplify this binding?

## Axis 1 results — two complementary sources

Axis 1 (geometric affordance probing) draws on two of Nitik's projects.
The story is **stronger as a synthesis** than either source alone.

### Source A — aleksantari/Probing_Bridging_Affordance @ VLA-affordance

Path: `geometry_probing/umd_linear_probing/results.md`.

**Linear probe mIoU on UMD (5 affordance + bg classes):**

| Encoder | mIoU @ res-UMD (480×640) | mIoU @ 224×224 |
|---|---|---|
| DINOv2-L/14 | **0.6660** | 0.3773 |
| DINOv2-B/14 | 0.6635 | 0.3383 |
| SigLIP raw | 0.6284 | 0.3125 |
| SigLIP PG1 (PaliGemma) | 0.6023 | 0.3519 |
| SigLIP π0.5 | 0.5648 | **0.3889** |
| SigLIP π0 | 0.5496 | 0.3719 |

**Headline:** VLA effect is **resolution-dependent and reverses sign**:
- Native UMD res (480×640): VLA fine-tuning degrades by −7.9 pp (raw → π0). DINOv2 remains the ceiling — "best-supported claim of the project."
- 224 (VLA op-res): VLA fine-tuning *improves* monotonically by +7.7 pp. π0.5 (0.389) beats DINOv2-L (0.377).

### Source B — PAVE (Nitik's other class project, github.com/nitik1998/PAVE)

PAVE probes the same SigLIP-from-VLA encoders on UMD with a different question and obtains complementary findings:

**Per-class IoU pattern (PAVE, validation split, 224 inputs):**
- `cut` drops 27 pp on π0 vs standalone SigLIP (0.455 → 0.181). On OpenVLA: −22 pp. Asymmetric loss generalises across two VLA families.
- `contain` is preserved within 2 pp on all three VLAs probed.
- `support` drops 17 pp.
- **Effect is class-selective**, not uniform.

**Layer-wise mechanism (linear CKA between standalone and each VLA, PAVE):**
- π0 keeps the encoder identical through layer 12 (CKA > 0.95), then rotates the *final block* to CKA = 0.023.
- π0.5 spreads divergence across middle layers (final CKA ≈ 0.36).
- OpenVLA reorganises least (final CKA ≈ 0.61).
- Per-class final-layer drift correlates with per-class IoU drop (Spearman ρ = 0.90, p = 0.037 on π0; Pearson r = 0.95, p = 0.012 on OpenVLA). **Encoder-internal geometry predicts downstream probing accuracy.**

**Intervention finding (PAVE):** a 297K-parameter MLP adapter on frozen VLA features recovers 82% of the cut-class gap on π0. The lost signal is *rotated*, not *deleted*.

**Methodological finding (PAVE complexity spectrum):** the headline 27 pp deficit on multi-class IoU collapses to <3 pp on binary part-discrimination formulations (cut-vs-other-foreground, handle-vs-blade) — the protocols downstream manipulation pipelines actually use. **The standard probing metric overstates the deficit relative to the perception sub-problem manipulation solves.**

### Synthesised Axis 1 narrative for this paper

VLA training is **resolution-specialised, class-asymmetric, mechanistically localised, and recoverable specialization — not destruction**:

1. The effect is resolution-dependent (Source A): degrades at native UMD resolution but improves at the VLA's operating resolution.
2. The effect is class-asymmetric (Source B): cut drops 27 pp; contain preserved within 2 pp.
3. The effect is layer-localised (Source B): rotation concentrated in the final transformer block that cross-attends with the action head.
4. The effect is recoverable (Source B): a 297K-param adapter on frozen features closes 82% of the cut gap.
5. The deficit is protocol-dependent (Source B): on binary part-discrimination tasks, the gap collapses to <3 pp.

This is a stronger Axis 1 story than the original proposal predicted. It also composes cleanly with Axis 2: VLAs and world models each specialise the visual representation to their own training distribution, with the specialisation visible at different "frequency bands" of the affordance signal.

### Methodology transfer from PAVE → Axis 2

PAVE's complexity-spectrum insight generalises. Alongside our KLD/SIM/NSS on AGD20K, we should also report a **binary peak-in-GT-region hit-rate**: for each (image, verb) pair, does the verb-attention peak fall inside the GT functional region (binary 1/0)? If the gap between Flux and Cosmos in KLD/SIM/NSS is large but the binary hit rate is similar, that mirrors PAVE's finding and suggests our metrics may also overstate the actual perceptual gap.

One extra analysis cell after the pilots, ~10 min compute. Adds a methodological hardening claim to Axis 2 without changing the main pipeline.

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

- Will Cosmos Policy accept image+text only without proprio, or will the pipeline error? (Plan: pass zeros for proprio; cross-attention shouldn't depend on it. Mitigation: thin wrapper if needed.)
- VRAM headroom: Cosmos V2W 480×704 × 17 frames × 12 steps may push the A100 40GB. Will fall back to 480×480 × 9 frames if OOM.
- Final-quality runs: should Flux switch to dev (20 steps, guidance=3.5) for the publication numbers? Decide after pilot.

## Resolved Questions (2026-05-10)

- **Cosmos Policy availability:** repo exists at `nvidia/Cosmos-Policy-ALOHA-Predict2-2B`, BUT — see deferral below.
- **Cosmos base model:** Cosmos Policy is fine-tuned from `Cosmos-Predict2-2B-Video2World`, NOT Text2Image. H2b/H2c paired comparison must use Video2World base.
- **Cosmos text encoder:** T5-11B (not T5-XXL as proposal stated). max_seq_length=512.
- **Cosmos cross-attention layer naming:** `attn2` modules in `CosmosTransformer3DModel`. Custom `AttentionProcessor` registration is the path forward (attention-map-diffusers does not support Cosmos).
- **Single-image probing for video model:** Cosmos2VideoToWorld accepts a single first-frame image. We use the AGD20K image directly, then average attention over the temporal latent dimension since binding is a spatial property.

## Integration debt found in Colab pilot (2026-05-10)

User's first Colab run hit four real bugs that synthetic-only testing didn't catch. Lessons:

1. **`attention-map-diffusers` is brittle.** The library monkey-patches diffusers internals and breaks every time diffusers ships a signature change. On Colab's diffusers, `FluxSingleTransformerBlock.forward` requires `encoder_hidden_states` as positional but the library calls `block(...)` without it → TypeError. **Fix:** rewrote `interaction/flux_attention.py` to install a custom `AttentionProcessor` on every Flux block's `attn`, mirroring diffusers' own `FluxAttnProcessor` math but with explicit softmax for capture. No external attention library needed.

2. **`cosmos_guardrail` package is broken on Colab.** Cosmos2VideoToWorldPipeline auto-instantiates a `CosmosSafetyChecker` that requires the optional `cosmos_guardrail` PyPI package, which fails to install cleanly. **Fix:** pass `safety_checker=None` to `from_pretrained()` — we're probing attention internals on AGD20K real-human-interaction images, the safety checker is irrelevant for our research.

3. **AGD20K loader didn't recurse into nested unzip dirs.** The LOCATE-mirror zip extracts to `./data/agd20k/AGD20K/Seen/...` (extra nested folder); my hardcoded path list missed it. **Fix:** loader now scans up to one level deep + has a `rglob("egocentric")` fallback. Error message also now prints `data_dir` contents so future failures are diagnosable.

4. **No pre-flight check.** User burned ~2 min on each model download before the bug surfaced. **Fix:** new `scripts/00_diagnostic.py` runs in 5 sec without touching GPU, checks all imports + AGD20K detection. Notebook Cell 2.5 calls it.

Also added: tqdm progress bars on scripts 10 / 10b (user complained about silent black-box runs).

## H2c Deferred (2026-05-10) — Cosmos Policy is not a diffusers pipeline

**Surfaced when:** user ran `09b_setup_cosmos.py --system cosmos_policy` on Colab — got `404 Not Found` on `https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B/resolve/main/model_index.json`.

**Root cause:** the HF repo lacks `model_index.json`, which `Cosmos2VideoToWorldPipeline.from_pretrained()` requires. The model is not a standard diffusers pipeline — it's a custom NVIDIA robot-policy package with action heads, proprioception input, and bespoke loading code (NVlabs/cosmos-policy repo). Architecturally identical to V2W but the loading mechanism is incompatible.

**Decision:** defer H2c. Run pilot with Flux (H2a) + Cosmos-Predict2-2B-Video2World (H2b) only. The two-system comparison still answers a real research question: *does base video diffusion develop verb-spatial binding via T5-XXL cross-attention?* (H2b). H2c remains an open question for follow-up work — would require a thin adapter that loads policy weights into a Video2World skeleton.

**Code state:** `cosmos_policy` entries in `09b_setup_cosmos.py` and `10b_run_cosmos_probing.py` now have `"deferred": True` and exit immediately with a clear notice (no GPU consumed). Notebook Cell 5 marked DEFERRED.

## Trajectory Log

(experiment-by-experiment results populated here)
