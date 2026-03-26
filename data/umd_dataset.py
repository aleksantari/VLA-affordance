"""PyTorch Dataset class for UMD Part Affordance Dataset."""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# UMD affordance label mapping
UMD_CATEGORIES = {
    0: "background",
    1: "grasp",
    2: "cut",
    3: "scoop",
    4: "contain",
    5: "pound",
    6: "support",
    7: "wrap-grasp",
}

NUM_CLASSES = 8  # 7 affordances + background


class UMDAffordanceDataset(Dataset):
    """UMD Part Affordance Dataset for linear probing.

    Each sample returns:
        - image: PIL Image (for encoder-specific preprocessing)
        - mask: (H, W) tensor with integer class labels 0-7
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        # Discover image-mask pairs
        self.samples = self._find_samples()
        print(f"UMD {split}: found {len(self.samples)} samples")

        # Mask transform (resize to match patch grid for supervision)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def _find_samples(self):
        """Find all (image, mask) pairs in the dataset."""
        samples = []

        # Try common UMD dataset structures
        # Structure 1: split/category/objectN/image_crop.png + image_label.png
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            # Structure 2: flat with train/test in filename
            split_dir = self.root_dir

        for rgb_path in sorted(split_dir.rglob("*_rgb.png")):
            mask_path = rgb_path.parent / rgb_path.name.replace("_rgb.png", "_label.png")
            if mask_path.exists():
                samples.append((str(rgb_path), str(mask_path)))

        # Also try _crop.png / _mask.png pattern
        if not samples:
            for rgb_path in sorted(split_dir.rglob("*_crop.png")):
                mask_path = rgb_path.parent / rgb_path.name.replace("_crop.png", "_label.png")
                if mask_path.exists():
                    samples.append((str(rgb_path), str(mask_path)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, mask_path = self.samples[idx]

        # Load RGB image as PIL
        image = Image.open(rgb_path).convert("RGB")

        # Load affordance mask
        mask = Image.open(mask_path)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        # Clamp to valid range
        mask = mask.clamp(0, NUM_CLASSES - 1)

        return image, mask

    @staticmethod
    def collate_fn(batch):
        """Custom collate that keeps images as list (for encoder preprocessing)."""
        images, masks = zip(*batch)
        masks = torch.stack(masks)
        return list(images), masks
