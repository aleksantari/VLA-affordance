# Axis 2 — Colab Runbook

End-to-end workflow for running the three-system probing on Colab Pro / Pro+.
Designed to be **interruptible**: any session crash leaves CSV results on
disk that the next run can resume from.

## One-time setup (do these once before first run)

1. **Get model access on HuggingFace** (browser):
   - [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) — usually open
   - [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) — gated, click Agree
   - [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) — may be gated
   - [Cosmos-Policy-ALOHA-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B) — may be gated
   - Make a token at <https://huggingface.co/settings/tokens> (Read access is enough)

2. **Add Colab Secrets** (notebook left sidebar 🔑):
   - `HF_TOKEN` — your HuggingFace token (mandatory for FLUX.1-dev and Cosmos)
   - `GH_PAT` — GitHub personal access token with `repo` scope (so the notebook can push results back; without it, results live on Drive only)

3. **Stage AGD20K on Drive** (~5 GB):
   - Download AGD20K from <https://github.com/lhc1224/Cross-View-AG> (their README has the Drive link).
   - Unzip and put it at `/content/drive/MyDrive/datasets/AGD20K`
   - Verify the egocentric directory is visible: e.g. `…/AGD20K/Seen/testset/egocentric/cut/...`

That's it for one-time setup. Going forward, opening the notebook just works.

## Running the pilot (~7 hours total)

Open `notebooks/axis2_unified_colab.ipynb` and run **cells 0 → 6 in order**.

| Cell | What it does | Time |
|---|---|---|
| 0 | Clone + checkout `nj-features` + pip install | ~2 min |
| 1 | Mount Drive, redirect HF cache to Drive, HF login | ~30 s |
| 2 | Symlink AGD20K, mirror `./results` to Drive | ~10 s |
| 3 | Flux smoke test + pilot (1080 samples) | ~45 min |
| 3.5 | Free Flux from VRAM | ~5 s |
| 4 | Cosmos V2W smoke test + pilot | ~3 hrs |
| 4.5 | Free Cosmos V2W from VRAM | ~5 s |
| 5 | Cosmos Policy smoke test + pilot | ~3 hrs |
| 6 | Three-way comparison + git push results | ~5 min |

**Cell 7 is optional** — JS keepalive that pokes the Colab tab every 30s
to suppress the 90-minute idle disconnect. Useful if your Colab tab will
be in the background for a long time. Doesn't extend the absolute session
cap (12h Pro / 24h Pro+).

## If the session dies

1. Open the notebook fresh.
2. Run cells 0 → 2 (fast, idempotent — re-cloning, re-mounting Drive,
   etc., are all no-ops if state is intact).
3. Re-run whichever long cell was running when the session died. The
   probing scripts default to `--resume`: they read the existing CSV,
   skip already-completed sample IDs, and continue.
4. The HF model cache lives on Drive (cell 1), so models reload from
   Drive in ~1 minute instead of re-downloading from HuggingFace
   (~5–7 min cold).

## Where the results live

Three places, in order of durability:

| Location | What | When |
|---|---|---|
| `/content/drive/MyDrive/VLA-affordance-results/` | All `./results/` content | Continuously (symlinked) |
| GitHub `nj-features` branch | Per-sample CSVs + final figures | Every 50–100 samples (auto-commit) + at end |
| `/content/VLA-affordance/results/` | Active working dir | Until session ends |

Even with no GH_PAT, results live on Drive. With GH_PAT, results also
flow to GitHub so the agent can pick them up locally and run analysis.

## Key files the agent watches for

After the pilot finishes, these files should exist:

- `results/tables/axis2_per_sample.csv` — Flux per-sample results
- `results/tables/axis2_cosmos_predict2_v2w_per_sample.csv`
- `results/tables/axis2_cosmos_policy_per_sample.csv`
- `results/tables/axis2_three_way_summary.csv` — all systems combined
- `results/tables/axis2_hypothesis_tests.json` — H2a/H2b/H2c verdicts
- `results/figures/axis2/three_way_*.png` — comparison plots

When these land on the `nj-features` branch, the agent's loop will
ingest them and write the next progress report.

## Final-quality runs (after pilot validates the protocol)

The pilot uses `--max_per_category 30` (1080 samples per system).
For the publication-quality run, increase to ~100/category and switch
Flux from `schnell` to `dev`:

```bash
# Flux dev with 20 inference steps (no max_per_category = full dataset)
python scripts/10_run_interaction_probing.py --model dev --commit_every 100 --save_attention_maps

# Cosmos V2W with longer denoising
python scripts/10b_run_cosmos_probing.py --system cosmos_predict2_v2w \
    --max_per_category 100 --num_inference_steps 20 --commit_every 100

# Cosmos Policy paired (must use same max_per_category for paired Wilcoxon to work)
python scripts/10b_run_cosmos_probing.py --system cosmos_policy \
    --max_per_category 100 --num_inference_steps 20 --commit_every 100
```

Final-quality runs total ~20 hours, **almost certainly span multiple
sessions**. Same resume mechanism applies — just keep re-running cells
4 and 5 across sessions.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM loading Cosmos V2W | Drop to `--num_inference_steps 8` and `--num_frames 9`; or pass `--cpu_offload` (slow but works) |
| Cosmos Policy errors about proprio | The current cosmos_attention.py extractor passes the AGD20K image as the conditioning frame and ignores proprio. If the pipeline still complains, add a thin wrapper that supplies zero proprio. Tell the agent and it'll patch. |
| Flux can't find verb tokens | Check `prompt` template in `data/agd20k_dataset.py`. Some affordance categories like "look_out" have multi-word gerunds ("looking out of") that span multiple tokens — extractor handles this, but worth eyeballing. |
| `git push` fails | Confirm `GH_PAT` Colab secret is set and has `repo` scope. Without it, results stay on Drive. |
| Resume doesn't actually skip rows | Verify the CSV exists at `results/tables/axis2_<system>_per_sample.csv` and has more than just the header. The script prints "Resume: N samples already done" at the start. |
