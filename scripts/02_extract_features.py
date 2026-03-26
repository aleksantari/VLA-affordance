"""Step 2: Extract & cache features from all encoders on UMD dataset.

Pre-extracts patch features for all encoder systems and caches them
to disk for fast iteration during probing experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoders.feature_extractor import UnifiedFeatureExtractor
from data.umd_dataset import UMDAffordanceDataset


ENCODER_NAMES = ["raw_siglip", "pi0_siglip", "pi05_siglip", "dinov2", "dino_wm"]


def extract_and_cache(encoder_name, dataset, cache_dir, device="cuda", batch_size=16):
    """Extract features for an encoder and save to disk.

    Saves:
        - features.npy: (N, 256, feature_dim) array
        - masks.npy: (N, 224, 224) array of integer labels
    """
    cache_path = Path(cache_dir) / encoder_name
    cache_path.mkdir(parents=True, exist_ok=True)

    features_path = cache_path / "features.npy"
    masks_path = cache_path / "masks.npy"

    if features_path.exists() and masks_path.exists():
        print(f"  Cache exists for {encoder_name}, skipping.")
        return

    print(f"\nExtracting features for: {encoder_name}")
    try:
        extractor = UnifiedFeatureExtractor(encoder_name, device=device)
    except Exception as e:
        print(f"  ERROR loading {encoder_name}: {e}")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=dataset.collate_fn,
    )

    all_features = []
    all_masks = []

    for images, masks in tqdm(dataloader, desc=f"  {encoder_name}"):
        with torch.no_grad():
            features = extractor.extract(images)  # (B, 256, C)
            all_features.append(features.cpu().numpy())
            all_masks.append(masks.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    np.save(str(features_path), all_features)
    np.save(str(masks_path), all_masks)

    print(f"  Saved {encoder_name}: features {all_features.shape}, masks {all_masks.shape}")

    del extractor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/umd_dataset")
    parser.add_argument("--cache_dir", default="./results/cached_features")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoders", nargs="+", default=ENCODER_NAMES)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = UMDAffordanceDataset(args.data_dir, split=args.split)

    for encoder_name in args.encoders:
        extract_and_cache(encoder_name, dataset, args.cache_dir,
                          device=device, batch_size=args.batch_size)

    print("\nFeature extraction complete!")
