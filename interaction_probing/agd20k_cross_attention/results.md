# Axis 2 — Results

**Status:** in progress. Results land as Colab pilots finish.

This document grows as data arrives. Each section is dated and references the commit that produced the underlying CSV.

## Table 1 — Per-system overall metrics on AGD20K

| System | n | KLD ↓ | SIM ↑ | NSS ↑ | peak_in_gt ↑ | Status |
|---|---|---|---|---|---|---|
| FLUX.1-schnell (pilot, 30/cat) | 849 | 1.77 ± 0.77 | 0.27 ± 0.14 | 0.45 ± 0.56 | — | **complete** (locally reported by user; CSV not yet on origin) |
| FLUX.1-schnell (full eval, all categories) | ~1710 expected | — | — | — | — | running on Colab |
| Cosmos-Predict2-2B-Video2World | — | — | — | — | — | pending Cell 4 |

## H2a verdict — does Flux replicate Zhang et al.?

Pilot (n=849, FLUX.1-schnell):

| Metric | Result | Locked threshold | Verdict |
|---|---|---|---|
| KLD ↓ | 1.77 | ≤ 1.7 | **misses by 0.07** |
| SIM ↑ | 0.27 | ≥ 0.30 | **misses by 0.03** |
| NSS ↑ | 0.45 | ≥ 1.0 | **misses by 0.55** |

H2a as locked is **refuted on schnell**. NSS gap is the largest. Most likely cause: schnell's 4-step denoising produces noisier attention than Zhang's likely FLUX.1-dev (20-step + CFG=3.5).

**Above-null sanity:** A uniform-attention baseline would give KLD ≈ 2.5–3, SIM ≈ 0.15, NSS ≈ 0. Our numbers (KLD=1.77, SIM=0.27, NSS=0.45) sit clearly above null on all three metrics — verb-spatial binding is present, just weaker than Zhang's dev-model numbers.

**Implication for paper:** report FLUX.1-schnell as a calibration baseline, NOT as the Zhang-replication number. If publication-quality replication is needed, run FLUX.1-dev later (one ~2h Colab pilot).

## Per-affordance breakdown

(populated when full-eval CSV pushes to origin)

## H2b verdict — does Cosmos-Predict2 V2W show binding?

(populated when Cell 4 pilot finishes)

## Statistical comparison

(populated when both systems are done — paired Wilcoxon on (Flux, Cosmos) per-sample deltas, stratified by manipulation vs other verbs)
