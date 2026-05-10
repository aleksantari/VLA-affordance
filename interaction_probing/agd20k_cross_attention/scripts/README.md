# Scripts (pointers)

Until the post-pilot code migration, the actual scripts live at repo root. This folder maps the conceptual entry-point names to the actual files.

| Conceptual name | Actual file | Purpose |
|---|---|---|
| `diagnostic.py` | `scripts/00_diagnostic.py` | 5-second pre-flight: imports + dataset detection |
| `setup_flux.py` | `scripts/09_setup_flux.py` | Flux smoke test (~3 min) |
| `setup_cosmos.py` | `scripts/09b_setup_cosmos.py` | Cosmos smoke test (~3 min) |
| `run_flux.py` | `scripts/10_run_interaction_probing.py` | Flux full probing |
| `run_cosmos.py` | `scripts/10b_run_cosmos_probing.py` | Cosmos full probing |
| `single_system_report.py` | `scripts/11_generate_axis2_report.py` | Per-system report |
| `comparison.py` | `scripts/12_three_way_comparison.py` | Two/three-system stats + figures |
| `qualitative.py` | `scripts/13_qualitative_panel.py` | Paper Figure 2 (qualitative grid) |
| `binary_metric.py` | `scripts/14_complexity_spectrum.py` | peak-in-GT binary metric on cached attention |
