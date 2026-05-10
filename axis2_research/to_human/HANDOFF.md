# Axis 2 — Pre-flight Handoff

**Last updated:** 2026-05-10 (right before user restarts Claude Code for Colab MCP)
**Branch:** `nj-features` @ `5719a3e`

## TL;DR for the next agent (or restarted me)

All Axis 2 infrastructure is built, pushed, and validated against synthetic data. Awaiting Colab pilot to land real per-sample CSVs.

## What's ready

- **Three-way probing pipeline** (Flux + Cosmos-Predict2-V2W + Cosmos-Policy) with locked H2 protocol in [axis2_research/experiments/H2_protocol.md](../experiments/H2_protocol.md).
- **Resilient Colab workflow** (incremental CSV with fsync, `--resume`, `--commit_every`, Drive-symlinked results, persistent HF cache, fail-fast smoke tests).
- **Statistical analysis** (script 12) validated end-to-end on synthetic data — `dev_synthetic_results_test.py` produces correct H2a/H2b/H2c verdicts.
- **Paper figure scaffolds:** script 12 (3-way bars + per-affordance), script 13 (qualitative panel).

## What's needed from the user (one-time)

1. `/exit` and reopen Claude Code → so `colab-proxy-mcp` tools load.
2. Have HF_TOKEN + GH_PAT ready to paste into Colab Secrets when the new tab opens.
3. Confirm the AGD20K.zip path on Drive: `/content/drive/MyDrive/LBV Project/AGD20K.zip` ✓ (user confirmed 2026-05-10).

## What I'll do post-restart

1. Verify `open_colab_browser_connection` tool is exposed.
2. Ask user to confirm runtime is A100 and secrets are set.
3. Call `open_colab_browser_connection` → user clicks Allow on browser prompt.
4. Dictate cells 0 → 6 of [notebooks/axis2_unified_colab.ipynb](../../notebooks/axis2_unified_colab.ipynb) into the connected notebook.
5. Run **stage 4a smoke (~3 min) FIRST** — before any 3-hour pilot. If it fails, I patch + push, user pulls + retries.
6. Walk-away phase: stage 4c + 5c long pilots (~6 hrs combined). User keeps Colab tab open.
7. Once CSVs are on `nj-features`: run script 12 locally (no GPU), generate figures, write progress report.

## Where to find things

| What | Path |
|---|---|
| Locked H2 protocol | [axis2_research/experiments/H2_protocol.md](../experiments/H2_protocol.md) |
| Living narrative | [axis2_research/findings.md](../findings.md) |
| Decision timeline | [axis2_research/research-log.md](../research-log.md) |
| Colab runbook (manual + MCP paths) | [axis2_research/to_human/COLAB_RUNBOOK.md](COLAB_RUNBOOK.md) |
| First progress report | [axis2_research/to_human/progress_2026-05-10.html](progress_2026-05-10.html) |
| Flux extractor | [interaction/flux_attention.py](../../interaction/flux_attention.py) |
| Cosmos extractor (audited) | [interaction/cosmos_attention.py](../../interaction/cosmos_attention.py) |
| Incremental CSV helper | [interaction/incremental_results.py](../../interaction/incremental_results.py) |
| Smoke tests | [scripts/09_setup_flux.py](../../scripts/09_setup_flux.py), [scripts/09b_setup_cosmos.py](../../scripts/09b_setup_cosmos.py) |
| Probing scripts | [scripts/10_run_interaction_probing.py](../../scripts/10_run_interaction_probing.py), [scripts/10b_run_cosmos_probing.py](../../scripts/10b_run_cosmos_probing.py) |
| Three-way comparison + stats | [scripts/12_three_way_comparison.py](../../scripts/12_three_way_comparison.py) |
| Qualitative panel | [scripts/13_qualitative_panel.py](../../scripts/13_qualitative_panel.py) |
| Synthetic test (regression check) | [scripts/dev_synthetic_results_test.py](../../scripts/dev_synthetic_results_test.py) |
| Unified Colab notebook | [notebooks/axis2_unified_colab.ipynb](../../notebooks/axis2_unified_colab.ipynb) |

## Open risks (in descending order of likelihood)

1. **Cosmos pipeline may emit `image_rotary_emb` to attn2 despite the diffusers source not showing it.** Smoke test 09b will catch this in 3 min — if it fails with "unexpected keyword argument", I patch the recorder signature.
2. **Cosmos Policy may reject single-image input** (it normally takes 3 multi-view + 14-dim proprio). Mitigation: thin wrapper that pads with zero proprio + duplicates AGD20K image as 3 views.
3. **Colab session caps** (12h Pro / 24h Pro+). Pilot is ~7 hrs; should fit one Pro+ session, two Pro sessions. Resume is implemented.
4. **VRAM OOM at 480×704 × 17 frames × 12 steps on Cosmos V2W.** Mitigation: drop to 480×480 × 9 frames or use --cpu_offload (slow but fits).
5. **Drive zip path might be slightly different** (`LBV Project` vs `LBV-Project`). Cell 2 prints which fallback hit; if none does, user fixes path and re-runs cell.

## Last 6 commits on nj-features

```
5719a3e feat(data): support AGD20K via local Drive zip; idempotent persistence
c8ffae8 fix(data): replace stale AGD20K Drive ID with verified LOCATE-mirror link
93aa84d test(axis2): synthetic-results integration test + MCP-driven runbook
465365e feat(axis2): qualitative panel generator (paper Figure 2 scaffold)
1737c8c fix(cosmos_attention): correct norm_q/norm_k order; add fail-fast smoke test
cb19c07 feat(axis2): resilient Colab pipeline — incremental CSV + resume + persistent caches
```
