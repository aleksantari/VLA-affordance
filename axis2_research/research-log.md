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

## 2026-05-10 (loop tick) — Resilience hardening + Cosmos extractor bugfix

**Context check:** No Colab results yet (origin/nj-features still at 4cbf7f9 before push). I had 3 unpushed local commits — pushed them so the Colab notebook would clone the latest code with the Cosmos extractor.

**Audit found 2 real bugs in `interaction/cosmos_attention.py`:**

1. `attn.norm_q(hidden_states)` — wrong: diffusers' CosmosAttnProcessor2_0 normalizes the *projected query* (after `to_q`), not the input. My version was double-projecting the unnormalized hidden_states.
2. `attn.norm_k(key)` was applied AFTER head reshape, but should be applied BEFORE. Diffusers normalizes pre-reshape on the (B, len, heads*head_d) tensor.

Both bugs would have produced incorrect outputs at runtime — likely NaN or garbage attention probs that bias all downstream metrics. Caught by reading the diffusers transformer_cosmos.py source carefully (key insight: their CosmosAttnProcessor2_0 does `query=to_q; key=to_k; query=norm_q(query); key=norm_k(key)`).

**Built a fail-fast smoke test:** `scripts/09b_setup_cosmos.py`.
- Loads pipeline → registers recorders → counts attn2 modules (must be > 0) → runs 1 inference at 320×320 + 5 frames + 4 steps (~3 min on A100) → checks attention map is non-constant and within [0, 1] → exits with code 2 on any failure.
- Without this, a buggy extractor would have wasted hours of A100 time before producing visibly wrong CSVs.

**Notebook updated:**
- Stages 4a/5a now run 09b smoke test (~3 min) before any pilot
- Stages 4b/5b run a 3×3 mini-pilot (~5 min) to validate AGD20K integration
- Stages 4c/5c are the long pilots
- Cleaner structure with proper VRAM cleanup cells

**Committed and pushed:** cb19c07 → origin/nj-features.

**Next:** Build the per-affordance qualitative figure scaffold (Figure 2 from the protocol) so when results arrive, figure generation is one command. Also prepare a minimal axis1+axis2 connecting figure scaffold.

## 2026-05-10 (later) — Final pre-flight: Colab MCP + dataset wiring

- **Colab MCP installed.** `claude mcp add colab-proxy-mcp uvx git+https://github.com/googlecolab/colab-mcp -s user` succeeded; `claude mcp list` reports ✓ Connected. Tools NOT available until user restarts Claude Code (MCP servers initialize at session start).
- **Found working AGD20K download.** Original Cross-View-AG Drive ID was stale; LOCATE repo (Reagan1311/LOCATE) hosts a verified mirror at `1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk`. Patched `data/download_agd20k.py` (commit c8ffae8).
- **User uploaded AGD20K.zip to `/content/drive/MyDrive/LBV Project/AGD20K.zip` on Drive.** Updated notebook Cell 2 to look for the zip there as a second fallback (after preferred unzipped path). When found, unzips into `./data/agd20k` AND persists copy to `/content/drive/MyDrive/datasets/AGD20K` so subsequent sessions take the fast symlink path. New `--from_zip` flag in `download_agd20k.py` (commit 5719a3e).
- **Regression check.** Re-ran `scripts/dev_synthetic_results_test.py` after all the data-pipeline edits. Still PASSES — H2a/H2b/H2c statistics fire correctly on synthetic data; `pass_nss=True` at p=7e-43; cleanup correctly removes synthetic results when none existed before.
- **Created qualitative panel scaffold.** `scripts/13_qualitative_panel.py` (commit 465365e) loads cached `.npy` per-sample attention maps from each system and produces image | GT | Flux | Cosmos V2W | Cosmos Policy comparison grid. Will run once pilot CSVs land.
- **Hardened the Cosmos extractor.** Audited `interaction/cosmos_attention.py` against diffusers' `CosmosAttnProcessor2_0` source; fixed two bugs (`norm_q` order, `norm_k` head-reshape order) that would have produced garbage attention probs at runtime. Added `scripts/09b_setup_cosmos.py` — ~3 min smoke test that verifies the extractor mechanics before committing to a 3-hour pilot (commit 1737c8c).
- **Hardened Colab notebook.** Resilience against session disconnect: incremental CSV writes with fsync, `--resume` flag (default on) skips done sample IDs, `--commit_every` auto-pushes mid-run, persistent HF cache on Drive, results symlinked to Drive, explicit VRAM cleanup between systems, optional JS keepalive (commit cb19c07).

**Where we are now:**
- All Axis 2 infrastructure pushed to origin/nj-features (HEAD = 5719a3e).
- 5 synthetic-test-validated commits this session: cb19c07, 1737c8c, 465365e, 93aa84d, c8ffae8, 5719a3e.
- Awaiting user to /exit and reopen Claude Code so the colab-proxy-mcp tools become available; then user sets HF_TOKEN + GH_PAT in Colab Secrets and tells me to start the pilot.

**Next inner loop (post-restart):** call `open_colab_browser_connection`, drive cells 0–6 in order, ingest the 3 per-sample CSVs that get pushed back, run script 12 + script 13, write the next progress report.
