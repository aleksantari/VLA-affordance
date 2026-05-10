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
