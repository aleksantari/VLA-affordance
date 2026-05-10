# Axis 2 Research Log — Decision Timeline

## 2026-05-10 — Bootstrap

**Context:** User invoked autoresearch on Axis 2. Axis 1 (geometric affordance probing across SigLIP progression) is complete; Axis 2 (interaction affordance via cross-attention in generative world models) is the active focus.

**Audit of existing infrastructure:**
- Flux extractor (interaction/flux_attention.py): complete, hooks via attention-map-diffusers
- Verb-spatial binding metrics (KLD/SIM/NSS): complete and unit-checked in script 09
- AGD20K loader with 36 affordance categories: complete, handles common dir layouts
- Visualization module: complete (overlay, comparison grid, timestep progression)
- Colab notebook (axis2_flux_colab.ipynb): exists for Flux only
- Cosmos extractors: NOT BUILT — this is the major gap

**Decision:** Don't re-implement Flux pipeline; it's solid. Focus the work on:
1. Replicate Zhang et al.'s Flux numbers on AGD20K (lock baseline) — H2a
2. Build Cosmos Predict2 extractor with the same VerbAttentionResult API — H2b
3. Build Cosmos Policy extractor (likely identical to Predict2 with different checkpoint) — H2c
4. Three-way comparison + statistical test
5. Tie back to Axis 1 in the connecting thesis

**Compute plan:** Agent does code + analysis locally. User runs all inference on Colab Pro (A100 40GB). Communication via committed CSV/JSON result files.

**Next inner loop:** literature pass on Cosmos Predict2/Policy (find HF model IDs, attention architecture), then start building cosmos_attention.py.

## 2026-05-10 (later) — Cosmos infrastructure complete

**Literature pass findings:**
- `nvidia/Cosmos-Predict2-2B-Video2World` and `nvidia/Cosmos-Policy-ALOHA-Predict2-2B` both publicly available on HuggingFace.
- Cosmos Policy is fine-tuned from Video2World (NOT Text2Image) — confirmed via model card. Decision: use Video2World as the H2b base for clean H2b/H2c paired comparison.
- Cosmos uses T5-11B (not T5-XXL), max_seq_length=512, default 35 inference steps.
- Cross-attention layers in `CosmosTransformer3DModel` are `attn2` (per diffusers source) — same convention as SDXL/SD3.
- attention-map-diffusers does NOT support Cosmos → custom `AttentionProcessor` written.

**Built:**
- `interaction/cosmos_attention.py`: `CosmosVerbAttentionExtractor`, `CosmosCrossAttnRecorder`. Mirrors Flux extractor API.
- `scripts/10b_run_cosmos_probing.py`: Cosmos counterpart to script 10. Output CSV layout matches.
- `scripts/12_three_way_comparison.py`: paired Wilcoxon for H2c, bootstrap CIs for H2a/H2b, stratified analysis on manipulation verbs, comparison figures.
- `notebooks/axis2_unified_colab.ipynb`: end-to-end Colab notebook for all 3 systems.
- `requirements_axis2.txt`: updated for Cosmos (scipy added, attention-map-diffusers no longer required for Cosmos).
- `axis2_research/to_human/progress_2026-05-10.html`: first progress report.

**Decisions logged:**
- Pilot scale: 30 samples × 36 categories = 1080 per system. Sufficient for paired Wilcoxon. Final scale (100+) deferred to post-pilot.
- Flux baseline runs schnell (4 steps) for pilot, dev (20 steps) reserved for publication-quality final.
- Cosmos resolution: 480×704, 17 frames, 12 inference steps (below default 35) to keep wall-clock manageable while preserving cross-attention pattern fidelity.

**Next inner loop:** await Colab pilot results. While waiting, think about:
- Visual story for the paper (the connecting thesis with Axis 1)
- Whether Cosmos Policy needs proprio padding (zero-vector tested first)
- Whether the prompt template `"a person {verb} a {object}"` is optimal for Cosmos (it's already proven for Flux/Zhang et al.)
