"""
Script 00: Pre-flight diagnostic — runs in seconds, no GPU.

Verifies the things that have failed in past Colab runs:
  1. Python deps importable (diffusers, torch, scipy, transformers)
  2. interaction.flux_attention and interaction.cosmos_attention import cleanly
  3. Cosmos pipeline classes exist in this diffusers version
  4. AGD20K dataset is discoverable at ./data/agd20k
  5. Cell 2 environment vars are set correctly

Run this on Colab BEFORE Cell 3 / Cell 4 to catch broken installs / missing
data / wrong paths in 5 seconds instead of after model download (~3 min)
or pilot start (~3 hrs).

Exit code is the number of failures.

Usage:
    python scripts/00_diagnostic.py
"""

import importlib
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
fails = []


def check(name, fn):
    print(f"  Checking {name}...", end=" ", flush=True)
    try:
        msg = fn()
        if msg is None:
            msg = "ok"
        print(f"{PASS} {msg}")
        return True
    except Exception as e:
        print(f"{FAIL} {type(e).__name__}: {e}")
        fails.append((name, e, traceback.format_exc()))
        return False


def main():
    print("=" * 60)
    print("AXIS 2 — Pre-flight Diagnostic")
    print("=" * 60)

    # ── 1. Core imports ──
    print("\n[1] Python dependencies")

    def _import(modname, attr=None):
        m = importlib.import_module(modname)
        if attr is not None:
            getattr(m, attr)
        return getattr(m, "__version__", "(no version)")

    check("torch", lambda: f"v{_import('torch')}")
    check("diffusers", lambda: f"v{_import('diffusers')}")
    check("transformers", lambda: f"v{_import('transformers')}")
    check("scipy", lambda: f"v{_import('scipy')}")
    check("numpy", lambda: f"v{_import('numpy')}")
    check("PIL", lambda: f"v{_import('PIL')}")

    # ── 2. Diffusers pipeline classes ──
    print("\n[2] Diffusers pipeline classes (Cosmos may be missing on old versions)")

    check("FluxPipeline", lambda: _import("diffusers", "FluxPipeline"))
    check(
        "Cosmos2VideoToWorldPipeline",
        lambda: _import("diffusers", "Cosmos2VideoToWorldPipeline"),
    )
    check(
        "Cosmos2TextToImagePipeline",
        lambda: _import("diffusers", "Cosmos2TextToImagePipeline"),
    )

    # ── 3. Our own modules ──
    print("\n[3] Project modules")

    check(
        "interaction.flux_attention",
        lambda: _import("interaction.flux_attention", "FluxVerbAttentionExtractor"),
    )
    check(
        "interaction.cosmos_attention",
        lambda: _import("interaction.cosmos_attention", "CosmosVerbAttentionExtractor"),
    )
    check(
        "interaction.verb_spatial_binding",
        lambda: _import("interaction.verb_spatial_binding", "evaluate_single"),
    )
    check(
        "interaction.incremental_results",
        lambda: _import("interaction.incremental_results", "IncrementalCSVWriter"),
    )
    check(
        "data.agd20k_dataset",
        lambda: _import("data.agd20k_dataset", "AGD20KDataset"),
    )

    # ── 4. AGD20K availability ──
    print("\n[4] AGD20K dataset")

    def _check_agd20k():
        from data.agd20k_dataset import AGD20KDataset

        data_dir = Path("./data/agd20k")
        if not data_dir.exists():
            raise FileNotFoundError(f"{data_dir} does not exist — run Cell 2")

        try:
            top = sorted(p.name for p in data_dir.iterdir())[:10]
        except Exception:
            top = ["(could not list)"]
        print(f"\n      contents: {top}")

        ds = AGD20KDataset(data_dir=str(data_dir), image_size=224)
        if len(ds) == 0:
            raise RuntimeError("loader returned 0 samples")
        # Also verify GT heatmap finder works for at least one sample
        n_with_gt = sum(1 for s in ds.samples if s.get("heatmap_path"))
        if n_with_gt == 0:
            raise RuntimeError(
                f"loader found {len(ds)} images but ZERO heatmaps — probing would skip everything"
            )
        n_cats = len(ds.get_affordance_categories())
        return f"{len(ds)} samples ({n_with_gt} with GT) across {n_cats} affordances"

    check("AGD20KDataset loads", _check_agd20k)

    # ── 5. Environment ──
    print("\n[5] Environment / paths")

    def _check_torch_cuda():
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{name} ({vram:.1f} GB)"
        else:
            return "(no GPU — diagnostic OK without one, but pilot needs A100)"

    check("torch.cuda", _check_torch_cuda)
    check("HF_HOME", lambda: os.environ.get("HF_HOME") or "(unset — Cell 1?)")
    check(
        "results dir",
        lambda: "exists" if Path("./results").exists() else "(missing — Cell 2?)",
    )

    # ── Summary ──
    print()
    print("=" * 60)
    if not fails:
        print(f"{PASS} ALL CHECKS PASSED — safe to proceed to Cell 3.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"{FAIL} {len(fails)} check(s) failed — fix before running pilots.")
        print("=" * 60)
        for name, exc, tb in fails:
            print(f"\n[{name}] {type(exc).__name__}: {exc}")
        sys.exit(len(fails))


if __name__ == "__main__":
    main()
