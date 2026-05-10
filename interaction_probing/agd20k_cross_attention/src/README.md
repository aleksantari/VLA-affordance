# Source code (pointers)

Until the post-pilot migration, source modules live at repo root. This folder maps the conceptual module names to the actual files.

| Conceptual location | Actual file | Purpose |
|---|---|---|
| `src/extractors/flux_attention.py` | `interaction/flux_attention.py` | `FluxVerbAttentionExtractor` + `FluxCrossAttnRecorder` |
| `src/extractors/cosmos_attention.py` | `interaction/cosmos_attention.py` | `CosmosVerbAttentionExtractor` + `CosmosCrossAttnRecorder` |
| `src/metrics/verb_spatial_binding.py` | `interaction/verb_spatial_binding.py` | KLD, SIM, NSS, peak_in_gt_region |
| `src/metrics/incremental_results.py` | `interaction/incremental_results.py` | Append-only resumable CSV writer |
| `src/visualization/visualization.py` | `interaction/visualization.py` | Attention overlays, comparison grids |
| `src/data/agd20k_dataset.py` | `data/agd20k_dataset.py` | AGD20K loader with parallel GT/ search |
