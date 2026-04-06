"""PyTorch Dataset class for UMD Part Affordance Dataset.

Supports the original UMD dataset (Myers et al., ICRA 2015) with:
- RGB images as *_rgb.jpg (480x640)
- Labels as *_label.mat (MATLAB, variable 'gt_label', values 0-7)
- Train/test split via category_split.txt
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# UMD affordance label mapping (7 affordances + background)
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


def _load_split_file(root_dir):
    """Load category_split.txt to determine train/test assignment.

    Returns:
        dict mapping object directory name -> 1 (train) or 2 (test)
    """
    split_assignments = {}

    # Try multiple possible locations for the split file
    candidates = [
        Path(root_dir) / "category_split.txt",
        Path(root_dir) / "umd_gt_category_split.txt",
        Path(root_dir).parent / "category_split.txt",
    ]

    split_path = None
    for candidate in candidates:
        if candidate.exists():
            split_path = candidate
            break

    if split_path is None:
        return None

    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                obj_name = parts[0]
                split_id = int(parts[1])
                split_assignments[obj_name] = split_id

    return split_assignments


def _load_mat_label(mat_path):
    """Load affordance label from a .mat file.

    Returns:
        numpy array (H, W) with integer values 0-7
    """
    from scipy.io import loadmat
    data = loadmat(str(mat_path))
    label = data["gt_label"].astype(np.int64)
    return label


class UMDAffordanceDataset(Dataset):
    """UMD Part Affordance Dataset for linear probing.

    Supports two dataset formats:
    1. Original UMD: *_rgb.jpg + *_label.mat (under tools/ subdirectory)
    2. UMD+GT derivative: *_rgb.png + *_labelid.png (flat structure)

    Each sample returns:
        - image: PIL Image (for encoder-specific preprocessing)
        - mask: (H, W) tensor with integer class labels 0-7
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        # Load train/test split assignments
        self.split_map = _load_split_file(root_dir)
        self.split_id = 1 if split == "train" else 2

        # Discover image-mask pairs
        self.samples = self._find_samples()
        print(f"UMD {split}: found {len(self.samples)} samples")

        # Mask transform (resize to match image_size for supervision)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def _get_search_dirs(self):
        """Get directories to search for samples."""
        dirs = []
        # Original UMD: tools/ subdirectory
        tools_dir = self.root_dir / "tools"
        if tools_dir.exists():
            dirs.append(tools_dir)
        # Flat structure (UMD+GT or already extracted)
        dirs.append(self.root_dir)
        return dirs

    def _obj_in_split(self, obj_name):
        """Check if an object directory belongs to the current split."""
        if self.split_map is None:
            return True  # No split file, include everything

        # Try exact match and common variants
        for name in [obj_name, obj_name.lower()]:
            if name in self.split_map:
                return self.split_map[name] == self.split_id
        return True  # Unknown object, include by default

    def _find_samples(self):
        """Find all (image, mask) pairs in the dataset."""
        samples = []

        for search_dir in self._get_search_dirs():
            # Strategy 1: Original UMD format (*_rgb.jpg + *_label.mat)
            for rgb_path in sorted(search_dir.rglob("*_rgb.jpg")):
                obj_name = rgb_path.parent.name
                if not self._obj_in_split(obj_name):
                    continue

                mat_path = rgb_path.parent / rgb_path.name.replace("_rgb.jpg", "_label.mat")
                if mat_path.exists():
                    samples.append((str(rgb_path), str(mat_path), "mat"))

            if samples:
                break  # Found samples, don't search other dirs

            # Strategy 2: UMD+GT format (*_rgb.png + *_labelid.png)
            for rgb_path in sorted(search_dir.rglob("*_rgb.png")):
                obj_name = rgb_path.parent.name
                if not self._obj_in_split(obj_name):
                    continue

                label_path = rgb_path.parent / rgb_path.name.replace("_rgb.png", "_labelid.png")
                if label_path.exists():
                    samples.append((str(rgb_path), str(label_path), "png"))

            if samples:
                break

            # Strategy 3: *_crop.png + *_label.png (alternative naming)
            for rgb_path in sorted(search_dir.rglob("*_crop.png")):
                obj_name = rgb_path.parent.name
                if not self._obj_in_split(obj_name):
                    continue

                label_path = rgb_path.parent / rgb_path.name.replace("_crop.png", "_label.png")
                if label_path.exists():
                    samples.append((str(rgb_path), str(label_path), "png"))

            if samples:
                break

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, mask_path, fmt = self.samples[idx]

        # Load RGB image as PIL
        image = Image.open(rgb_path).convert("RGB")

        # Load affordance mask
        if fmt == "mat":
            mask_np = _load_mat_label(mask_path)
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
        else:
            mask_pil = Image.open(mask_path)

        mask_pil = self.mask_transform(mask_pil)
        mask = torch.from_numpy(np.array(mask_pil)).long()

        # Clamp to valid range
        mask = mask.clamp(0, NUM_CLASSES - 1)

        return image, mask

    @staticmethod
    def collate_fn(batch):
        """Custom collate that keeps images as list (for encoder preprocessing)."""
        images, masks = zip(*batch)
        masks = torch.stack(masks)
        return list(images), masks
