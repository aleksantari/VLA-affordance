"""Step 4: PCA subspace projection experiments for all encoders."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path
from PIL import Image

from encoders.feature_extractor import UnifiedFeatureExtractor
from probing.pca_analysis import pca_subspace_analysis, quantify_part_separation
from evaluation.visualization import plot_pca_colormap, plot_pca_comparison


ENCODER_NAMES = ["raw_siglip", "pi0_siglip", "pi05_siglip", "dinov2", "dino_wm"]


def run_pca_analysis(reference_image, test_images, device="cuda",
                     encoders=None, n_components=3, figures_dir="./results/figures"):
    """Run PCA analysis for all encoders.

    Args:
        reference_image: PIL image of reference object (e.g., a mug)
        test_images: list of PIL images for projection
        device: torch device
        encoders: list of encoder names (default: all)
        n_components: PCA components
        figures_dir: directory for saving figures
    """
    if encoders is None:
        encoders = ENCODER_NAMES

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_projections = {}
    all_separation_ratios = {}

    for encoder_name in encoders:
        print(f"\nPCA Analysis: {encoder_name}")
        try:
            extractor = UnifiedFeatureExtractor(encoder_name, device=device)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        results, pca, ref_projected = pca_subspace_analysis(
            extractor, reference_image, test_images, n_components=n_components
        )

        # Save reference projection
        all_projections[encoder_name] = ref_projected

        # Plot individual PCA colormap
        plot_pca_colormap(
            ref_projected,
            title=f"{encoder_name} - PCA Projection",
            save_path=figures_dir / f"pca_{encoder_name}_reference.png"
        )

        # Plot test image projections
        for i, proj in enumerate(results):
            plot_pca_colormap(
                proj,
                title=f"{encoder_name} - Test Image {i+1}",
                save_path=figures_dir / f"pca_{encoder_name}_test_{i}.png"
            )

        print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")

        del extractor
        torch.cuda.empty_cache()

    # Comparison plot
    if all_projections:
        plot_pca_comparison(
            all_projections,
            save_path=figures_dir / "pca_comparison.png"
        )

    return all_projections, all_separation_ratios


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", required=True, help="Path to reference image")
    parser.add_argument("--test_images", nargs="+", required=True, help="Paths to test images")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    parser.add_argument("--figures_dir", default="./results/figures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_img = Image.open(args.reference_image).convert("RGB")
    test_imgs = [Image.open(p).convert("RGB") for p in args.test_images]

    run_pca_analysis(ref_img, test_imgs, device=device, encoders=args.encoders,
                     n_components=args.n_components, figures_dir=args.figures_dir)
