"""Step 5: Extract and cache depth/normal features for all UMD images.

Uses DPT (Dense Prediction Transformer) for monocular depth estimation,
then derives surface normals via finite differences. These features are
encoder-independent and only need to be extracted once.

Output: (N, 4, 16, 16) per split — depth (1ch) + normals (3ch) at patch grid resolution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.umd_dataset import UMDAffordanceDataset
from probing.depth_normal import load_depth_model, extract_depth_normal_features


def extract_and_cache_depth_normal(dataset, cache_dir, split, device="cuda", batch_size=1):
    """Extract depth/normal features for all images in a dataset split.

    Processes one image at a time (DPT is memory-intensive).

    Saves:
        - depth_normal.npy: (N, 4, 16, 16) float32 array
    """
    cache_path = Path(cache_dir) / "depth_normal" / split
    cache_path.mkdir(parents=True, exist_ok=True)

    features_path = cache_path / "depth_normal.npy"

    if features_path.exists():
        print(f"  Cache exists for {split}, skipping.")
        return

    print(f"\nExtracting depth/normal features for {split} split...")
    depth_model, depth_processor = load_depth_model(device=device)

    all_features = []

    for idx in tqdm(range(len(dataset)), desc=f"  depth/normal ({split})"):
        image, _ = dataset[idx]  # PIL image, mask

        features = extract_depth_normal_features(
            image, depth_model, depth_processor, device=device
        )  # (1, 4, 16, 16)

        all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)  # (N, 4, 16, 16)

    np.save(str(features_path), all_features)
    print(f"  Saved {split}: depth_normal {all_features.shape}")

    del depth_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/umd_dataset")
    parser.add_argument("--cache_dir", default="./results/cached_features")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for split in ["train", "test"]:
        dataset = UMDAffordanceDataset(args.data_dir, split=split)
        extract_and_cache_depth_normal(dataset, args.cache_dir, split, device=device)

    print("\nDepth/normal extraction complete!")
