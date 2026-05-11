# Axis 2 — Results

**Last updated:** 2026-05-11
**Branch:** `nj-features`

## Table 1 — Overall metrics on AGD20K

Source: user's Colab run, manually reported. Full CSV exists on Drive at `/MyDrive/VLA-affordance-results/tables/axis2_per_sample.csv` but not yet fully persisted to repo (tooling chunking limits).

| System | Model | n | KLD ↓ | SIM ↑ | NSS ↑ |
|---|---|---|---|---|---|
| FLUX.1-schnell (pilot, 30/cat) | schnell, 4 steps, no CFG | 849 | 1.77 ± 0.77 | 0.27 ± 0.14 | 0.45 ± 0.56 |
| **FLUX.1-dev (full eval)** | **dev, 20 steps, CFG=3.5** | **1675** | **1.86 ± 0.73** | **0.25 ± 0.13** | **0.43 ± 0.55** |
| Uniform null (reference) | — | — | ~2.5–3 | ~0.15 | 0 |
| Zhang et al. (CVPR 2026, FLUX) | published | full | 1.49 | 0.33 | 1.09 |

## Headline findings

### 1. Binding is real
All three metrics sit clearly above the uniform-attention null:
- KLD = 1.86 < ~2.7 (well below null)
- SIM = 0.25 > 0.15 (67% above null)
- NSS = 0.43 > 0 (clearly positive concentration)

### 2. schnell ≈ dev
| Metric | schnell (n=849) | dev (n=1675) | Δ |
|---|---|---|---|
| KLD | 1.77 | 1.86 | +0.09 |
| SIM | 0.27 | 0.25 | −0.02 |
| NSS | 0.45 | 0.43 | −0.02 |

5× more inference compute did not meaningfully shift the binding numbers. Schnell is a valid fast proxy. The gap to Zhang's numbers is methodology-bound, not model-bound.

### 3. Locked H2a thresholds missed
Locked predictions: KLD≤1.7, SIM≥0.30, NSS≥1.0. All three failed on both schnell and dev. The strict-replication version of H2a is refuted; the qualitative version (binding above null) is confirmed.

## Per-affordance breakdown (partial — 5 of 36 categories)

The following 5 categories were successfully loaded into `results/tables/axis2_per_sample.csv` (n=109 of 1675):

| Affordance | n | NSS mean | Reading |
|---|---|---|---|
| boxing | 23 | **+0.82** | strong binding |
| beat | 12 | +0.48 | moderate binding |
| carry | 30 | +0.28 | weak positive |
| catch | 30 | +0.18 | weak positive |
| brush_with | 14 | **−0.09** | no binding / slight anti-binding |

This partial sample already shows a pattern that will likely hold across all 36 categories: **highly action-specific verbs ("boxing", "beating") bind to consistent body/contact regions; ambiguous or distal verbs ("brush_with") fail.** Worth a full breakdown once the full CSV is available.

## Implications for the paper

The paper's claim is *relative*. Even with absolute numbers below Zhang's, the Flux-vs-Cosmos comparison stays defensible because both systems will be measured by the identical pipeline.

**Updates to the paper narrative from these results:**

1. **Drop** "we replicate Zhang's Flux numbers." We don't.
2. **Add** "Flux shows verb-spatial binding above null at the level our pipeline can detect. The methodology has known limitations (single-stream blocks unrecorded; uniform block weighting; T5 token-index extraction; spatial reshape) that we document and defer to future work."
3. **Center** the Flux-vs-Cosmos comparison as the novel contribution.
4. **Per-category** results (partial above) suggest binding is heterogeneous across affordances — manipulation-relevant verbs likely show stronger binding than ambient ones. Final breakdown when CSV is persisted.

## H2b — Cosmos verdict

(pending — tonight's run via `notebooks/axis2_cosmos_optimized.ipynb`)

## Paired comparison

(pending both systems complete)

## Note on data persistence

The full 1675-sample CSV exists on the user's Drive but couldn't be persisted to the repo through this conversation's tooling (the Write tool's per-call output size made it impractical to append all 36 categories' worth of rows). Two options to fix:

**Option A (preferred):** save the file from Drive directly to the repo and push:
```bash
# In a fresh Colab CPU notebook
from google.colab import drive
drive.mount('/content/drive')
!git clone https://YOUR_GH_PAT@github.com/aleksantari/VLA-affordance.git /content/repo
!cp '/content/drive/MyDrive/VLA-affordance-results/tables/axis2_per_sample.csv' /content/repo/results/tables/
%cd /content/repo
!git checkout nj-features
!git add results/tables/axis2_per_sample.csv
!git commit -m 'results(flux): persist full Colab CSV (1675 samples)'
!git push origin nj-features
```

**Option B (quick summary):** run this one-liner in a Colab CPU notebook to produce a small per-affordance table I can analyze directly:
```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/VLA-affordance-results/tables/axis2_per_sample.csv')
print(df.groupby(['system','affordance'])[['kld','sim','nss']].agg(['mean','std','count']).round(3))
```

Paste the output and the full per-category analysis follows immediately.
