# Axis 2 — Results

**Last updated:** 2026-05-11
**Branch:** `nj-features`
**Source CSV:** `results/tables/axis2_per_sample.csv` (n=1675, persisted in commit)

## Table 1 — Per-system overall metrics on AGD20K

| System | Model config | n | KLD ↓ | SIM ↑ | NSS ↑ |
|---|---|---|---|---|---|
| FLUX.1-schnell | 4 steps, no CFG | 849 | **1.767 ± 0.77** | **0.266 ± 0.14** | **+0.449 ± 0.56** |
| FLUX.1-dev | 20 steps, CFG=3.5 | 826 | 1.965 ± 0.67 | 0.224 ± 0.12 | +0.417 ± 0.53 |
| Combined | — | **1675** | 1.865 ± 0.73 | 0.246 ± 0.13 | +0.433 ± 0.55 |
| Uniform null (reference) | — | — | ~2.7 | ~0.15 | 0 |
| Zhang et al. (CVPR 2026, Flux) | published | full | 1.49 | 0.33 | 1.09 |

## Headline findings

### 1. Binding is real
All three metrics sit clearly above null:
- **KLD = 1.86** is 31% below the uniform-null baseline (~2.7)
- **SIM = 0.25** is 67% above null (~0.15)
- **NSS = 0.43** is 0.43 σ of attention concentration at GT — clearly positive

### 2. **Schnell beats Dev** (statistically significant)

This is the most unexpected finding.

| Metric | Schnell mean (n=849) | Dev mean (n=826) | Mann-Whitney p | Verdict |
|---|---|---|---|---|
| KLD ↓ | 1.767 | 1.965 | **p < 0.0001** | schnell *significantly* better |
| SIM ↑ | 0.266 | 0.224 | **p < 0.0001** | schnell *significantly* better |
| NSS ↑ | +0.449 | +0.417 | p = 0.16 | indistinguishable |

5× more inference compute (dev's 20 steps + CFG=3.5) **reduces** attention map fidelity vs GT on two of three metrics. Why this might happen:

- **CFG sharpens attention** but it sharpens on the *model's* interpretation of the prompt, not the AGD20K GT regions. CFG=3.5 amplifies model bias.
- **More denoising steps** produces sharper but potentially less GT-aligned final attention. Early in denoising, attention is diffuse and partially right; later steps focus on what the model *thinks* the scene should look like.
- **Schnell's 4-step trajectory is closer to "early" attention** of dev — apparently a better match to GT for this particular evaluation.

Implication: **fast configs are not just adequate for Cosmos comparison — they may be *preferred*.** This validates the design of `notebooks/axis2_cosmos_optimized.ipynb` (8 steps, no CFG).

### 3. Manipulation verbs bind *weaker* than other verbs (also unexpected)

Mann-Whitney comparison between two groups:
- **Manipulation verbs** (hold, pick_up, pour, push, lift, carry, drag, pack, stir, kick, throw, cut, cut_with, peel, stick, wash): n=794, NSS = +0.349
- **Other verbs**: n=881, NSS = +0.509
- **p = 3 × 10⁻⁹** (highly significant)

The original H2c hypothesis assumed manipulation-relevant verbs would show *stronger* binding (because Cosmos Policy is trained on manipulation data, the gap would amplify there). For Flux, the *opposite* is true: more abstract verbs (lying on, talking on, sitting on) bind more sharply to GT than physical-manipulation verbs.

Likely explanation: AGD20K's "manipulation" verbs often have *distributed* GT regions (e.g. wash covers the whole sink + tool; push covers the entire pushed object). Postural/contact verbs have *concentrated* GT regions (lie_on covers a single surface). Concentrated GT is easier for any attention to match.

### 4. Top and bottom binding affordances

| Rank | Affordance | n | NSS ↑ | Comment |
|---|---|---|---|---|
| 1 | lie_on | 68 | +0.828 | body→surface, very concentrated GT |
| 2 | talk_on | 9 | +0.824 | phone+face |
| 3 | boxing | 23 | +0.817 | hands+target |
| 4 | kick | 50 | +0.772 | feet+ball |
| 5 | peel | 24 | +0.771 | hands+fruit |
| 6 | text_on | 9 | +0.723 | |
| 7 | cut | 39 | +0.694 | |
| 8 | type_on | 29 | +0.683 | |
| 9 | stir | 8 | +0.672 | |
| 10 | write | 20 | +0.633 | |
| ⋮ | | | | |
| 32 | push | 26 | +0.114 | distributed action |
| 33 | wash | 28 | +0.061 | distributed action |
| 34 | stick | 48 | +0.018 | ambiguous verb |
| 35 | swing | 42 | −0.011 | dynamic, no fixed locus |
| 36 | brush_with | 14 | **−0.092** | only category with negative binding |

### 5. Locked H2a thresholds remain refuted

Locked predictions: KLD ≤ 1.7, SIM ≥ 0.30, NSS ≥ 1.0. All three failed on both schnell and dev. The qualitative version of H2a (binding above null) is confirmed; the literal version (matching Zhang's published numbers) is not.

## Implications for the paper

1. **The methodology has a known gap vs Zhang.** Document honestly. Likely causes: we only record on double-stream blocks; uniform block weighting; T5 token-index extraction; spatial reshape (Flux's 2×2 patch packing). All diagnosable on cached attention maps in future work.

2. **Schnell is the right tool for the comparison.** Schnell beat dev on KLD/SIM under our pipeline. Use schnell-equivalent fast configs for Cosmos.

3. **The manipulation-vs-other finding is a real, surprising contribution.** Original H2 reasoning assumed manipulation verbs would dominate. Reality is the opposite for Flux. This will matter for interpreting Cosmos: if Cosmos shows the same manipulation-weakness pattern, that's evidence the model is doing the *same kind* of binding; if Cosmos shows the manipulation-strength pattern, that's a real specialization difference.

4. **Per-affordance heterogeneity is large.** Best category (lie_on, +0.83) vs worst (brush_with, −0.09) span ~1.5 σ. Future analyses should stratify by affordance type (postural/contact vs distributed manipulation) rather than reporting only overall means.

## Statistical artifacts produced

- `results/tables/axis2_per_sample.csv` — full 1675-row data
- `results/tables/axis2_three_way_summary.csv` — per-system overall + 95% bootstrap CIs
- `results/tables/axis2_hypothesis_tests.json` — H2a verdict JSON
- `results/figures/axis2/three_way_overall.png` — bar chart with CIs
- `results/figures/axis2/three_way_per_affordance_{kld,sim,nss}.png` — per-affordance bar charts

## H2b — Cosmos verdict

(pending — tonight's run via `notebooks/axis2_cosmos_optimized.ipynb`)

## Paired Flux-vs-Cosmos comparison

(pending both systems complete)
