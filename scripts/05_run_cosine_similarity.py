"""Step 5: Part correspondence experiments via cosine similarity."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path
from PIL import Image

from encoders.feature_extractor import UnifiedFeatureExtractor
from probing.cosine_similarity import cosine_similarity_map, compute_correspondence_accuracy
from evaluation.visualization import plot_similarity_heatmap


ENCODER_NAMES = ["raw_siglip", "pi0_siglip", "pi05_siglip", "dinov2", "dino_wm"]


def run_cosine_similarity(query_image, query_patch_idx, target_image,
                          target_part_mask=None, device="cuda",
                          encoders=None, figures_dir="./results/figures"):
    """Run cosine similarity analysis for all encoders.

    Args:
        query_image: PIL image with query object
        query_patch_idx: (row, col) of query patch in 16x16 grid
        target_image: PIL image with target object
        target_part_mask: optional (16, 16) binary mask of corresponding part
        device: torch device
        encoders: list of encoder names
        figures_dir: directory for saving figures
    """
    if encoders is None:
        encoders = ENCODER_NAMES

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for encoder_name in encoders:
        print(f"\nCosine Similarity: {encoder_name}")
        try:
            extractor = UnifiedFeatureExtractor(encoder_name, device=device)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        sim_map = cosine_similarity_map(
            extractor, query_image, query_patch_idx, target_image
        )

        plot_similarity_heatmap(
            sim_map,
            query_patch_idx=query_patch_idx,
            title=f"{encoder_name} - Cosine Similarity",
            save_path=figures_dir / f"cosim_{encoder_name}.png"
        )

        result = {"similarity_map_saved": True}

        if target_part_mask is not None:
            accuracy = compute_correspondence_accuracy(
                extractor, query_image, query_patch_idx,
                target_image, target_part_mask
            )
            result.update({
                "hit_at_1": accuracy["hit_at_1"],
                "hit_at_k": accuracy["hit_at_k"],
                "similarity_ratio": accuracy["similarity_ratio"],
            })
            print(f"  Hit@1: {accuracy['hit_at_1']}")
            print(f"  Hit@K: {accuracy['hit_at_k']:.3f}")
            print(f"  Similarity ratio: {accuracy['similarity_ratio']:.3f}")

        all_results[encoder_name] = result

        del extractor
        torch.cuda.empty_cache()

    # Save results
    results_dir = Path("./results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Filter non-serializable values
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, type(None))}
    with open(results_dir / "cosine_similarity_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_image", required=True)
    parser.add_argument("--query_row", type=int, required=True)
    parser.add_argument("--query_col", type=int, required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    parser.add_argument("--figures_dir", default="./results/figures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    query_img = Image.open(args.query_image).convert("RGB")
    target_img = Image.open(args.target_image).convert("RGB")

    run_cosine_similarity(
        query_img, (args.query_row, args.query_col), target_img,
        device=device, encoders=args.encoders, figures_dir=args.figures_dir,
    )
