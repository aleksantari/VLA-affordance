"""
Script 13: Generate the Axis 2 qualitative panel (paper Figure 2).

For a curated set of representative affordances, produces a grid showing:
  Original image | GT heatmap | Flux pred | Cosmos V2W pred | Cosmos Policy pred

Loads cached attention maps from `results/cached_features/<system>_attention/*.npy`
(produced by scripts 10/10b with --save_attention_maps).

Usage:
    python scripts/13_qualitative_panel.py
    python scripts/13_qualitative_panel.py --affordances cut hold pour pick_up lift drink_with
    python scripts/13_qualitative_panel.py --samples_per_aff 1 --output paper_figure2.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# Reasonable defaults: 6 affordances spanning manipulation diversity
DEFAULT_AFFORDANCES = ["cut", "hold", "pour", "pick_up", "lift", "drink_with"]

SYSTEM_LABELS = {
    "flux_attention": "Flux",
    "cosmos_predict2_v2w_attention": "Cosmos-Predict2 (V2W)",
    "cosmos_policy_attention": "Cosmos-Policy (FT)",
}


def overlay_heatmap(img: np.ndarray, hm: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a heatmap onto an image. img: (H, W, 3) uint8; hm: (Hh, Wh) float."""
    if hm.shape != img.shape[:2]:
        hm = np.array(
            Image.fromarray(hm.astype(np.float32), mode="F").resize(
                (img.shape[1], img.shape[0]), Image.BILINEAR
            )
        )
    if hm.max() > hm.min():
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min())
    else:
        hm_norm = np.zeros_like(hm)
    cmap = plt.get_cmap("jet")
    rgba = cmap(hm_norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return ((1 - alpha) * img + alpha * rgb).astype(np.uint8)


def find_sample_for_affordance(
    cache_dirs: dict,
    affordance: str,
    n: int,
) -> list:
    """
    Return up to n sample IDs that have cached attention from ALL system caches.
    """
    sets = []
    for sys_id, cdir in cache_dirs.items():
        if not cdir.exists():
            sets.append(set())
            continue
        ids = {p.stem for p in cdir.glob(f"{affordance}_*.npy")}
        sets.append(ids)
    if not sets or not all(sets):
        # Fall back to whichever system has any
        return sorted(set.union(*sets))[:n] if sets else []
    common = sorted(set.intersection(*sets))
    return common[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--data_dir", type=str, default="./data/agd20k")
    p.add_argument(
        "--affordances", type=str, nargs="+", default=DEFAULT_AFFORDANCES,
        help="Affordances to include as rows",
    )
    p.add_argument("--samples_per_aff", type=int, default=1)
    p.add_argument("--output", type=str, default="qualitative_panel.png")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    cache_root = results_dir / "cached_features"

    cache_dirs = {
        "flux_attention": cache_root / "flux_attention",
        "cosmos_predict2_v2w_attention": cache_root / "cosmos_predict2_v2w_attention",
        "cosmos_policy_attention": cache_root / "cosmos_policy_attention",
    }
    print("Cache dirs:")
    for k, v in cache_dirs.items():
        n = len(list(v.glob("*.npy"))) if v.exists() else 0
        print(f"  {k}: {n} maps in {v}")

    from data.agd20k_dataset import AGD20KDataset
    dataset = AGD20KDataset(data_dir=args.data_dir, image_size=512)

    # For each affordance, collect rows of (sample_id, image, GT, *system_preds)
    rows = []
    for aff in args.affordances:
        sample_ids = find_sample_for_affordance(cache_dirs, aff, args.samples_per_aff)
        if not sample_ids:
            print(f"  ⚠ No cached attention for affordance '{aff}', skipping")
            continue
        for sid in sample_ids:
            # Locate the AGD20K sample by ID (sample_ids look like 'cut_3')
            # Need to map "<aff>_<i>" -> dataset index whose affordance matches and is the i-th
            try:
                _, idx_str = sid.rsplit("_", 1)
                idx = int(idx_str)
            except ValueError:
                continue
            indices_for_aff = dataset.get_samples_by_affordance(aff)
            if idx >= len(indices_for_aff):
                continue
            sample = dataset[indices_for_aff[idx]]
            img = np.array(sample["image"])
            gt = sample["gt_heatmap"]
            if gt is None:
                continue

            preds = {}
            for sys_key, cdir in cache_dirs.items():
                p_path = cdir / f"{sid}.npy"
                preds[sys_key] = np.load(p_path) if p_path.exists() else None

            rows.append({
                "sid": sid, "aff": aff, "img": img, "gt": gt, "preds": preds,
                "prompt": sample["prompt"],
            })

    if not rows:
        print("ERROR: no rows assembled — likely no cached attention maps yet.")
        print("       Run scripts/10*.py with --save_attention_maps first.")
        sys.exit(1)

    n_rows = len(rows)
    col_titles = ["Image", "GT", *[SYSTEM_LABELS[k] for k in cache_dirs]]
    n_cols = len(col_titles)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axs = axs[None, :]

    for r, row in enumerate(rows):
        # Image
        axs[r, 0].imshow(row["img"])
        axs[r, 0].set_ylabel(f"{row['aff']}\n{row['sid']}", fontsize=8)
        # GT overlay
        axs[r, 1].imshow(overlay_heatmap(row["img"], row["gt"]))
        # System preds
        for ci, (sys_key, _) in enumerate(cache_dirs.items()):
            ax = axs[r, 2 + ci]
            pred = row["preds"][sys_key]
            if pred is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
                ax.imshow(row["img"], alpha=0.3)
            else:
                ax.imshow(overlay_heatmap(row["img"], pred))

        for ax in axs[r]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Column titles
    for ci, t in enumerate(col_titles):
        axs[0, ci].set_title(t, fontsize=10)

    plt.tight_layout()
    out_path = Path(args.results_dir) / "figures" / "axis2" / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


if __name__ == "__main__":
    main()
