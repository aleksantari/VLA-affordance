# Configs

One YAML per (system × variant) combination. Decouples experimental parameters from code.

| Config | Purpose | Wall-clock |
|---|---|---|
| `flux_schnell.yaml` | H2a calibration baseline | ~45 min pilot |
| `flux_dev.yaml` | H2a publication-quality replication | ~2-3 hrs full |
| `cosmos_predict2_v2w.yaml` | H2b — base video diffusion | ~3 hrs pilot |

**Status (2026-05-10):** these YAMLs document the canonical parameters for each system. The current Python scripts (`scripts/09*`, `scripts/10*`, etc.) still take parameters as CLI flags rather than loading from YAML; the YAML files are the source of truth for what those CLI invocations should do.

When the code is refactored into this folder, the scripts will load directly from these YAMLs and the CLI will become `python scripts/run.py --config configs/flux_schnell.yaml`.
