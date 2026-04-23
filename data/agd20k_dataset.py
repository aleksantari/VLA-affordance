"""
AGD20K Dataset — PyTorch Dataset for Affordance Grounding

Loads egocentric (object-centric) images from AGD20K with their
affordance heatmap ground truths for evaluating verb-spatial binding
in Flux cross-attention maps.

Reference: Luo et al., "Learning Affordance Grounding from
Exocentric Images", CVPR 2022.

The dataset has 36 affordance categories. Each egocentric image has
a corresponding affordance heatmap indicating functional regions.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


# 36 AGD20K affordance categories
# Mapping from category name to verb for prompt construction
AGD20K_AFFORDANCES = {
    "beat": "beating",
    "boxing": "boxing",
    "brush_with": "brushing with",
    "carry": "carrying",
    "catch": "catching",
    "cut": "cutting",
    "cut_with": "cutting with",
    "drag": "dragging",
    "drink_with": "drinking with",
    "eat": "eating",
    "hit": "hitting",
    "hold": "holding",
    "jump": "jumping",
    "kick": "kicking",
    "lie_on": "lying on",
    "lift": "lifting",
    "look_out": "looking out of",
    "open": "opening",
    "pack": "packing",
    "peel": "peeling",
    "pick_up": "picking up",
    "pour": "pouring",
    "push": "pushing",
    "ride": "riding",
    "sip": "sipping",
    "sit_on": "sitting on",
    "stick": "sticking",
    "stir": "stirring",
    "swing": "swinging",
    "take_photo": "taking a photo with",
    "talk_on": "talking on",
    "text_on": "texting on",
    "throw": "throwing",
    "type_on": "typing on",
    "wash": "washing",
    "write": "writing",
}

# Verb roots for token identification in Flux attention maps
AGD20K_VERB_ROOTS = {
    "beat": ["beat", "beating"],
    "boxing": ["box", "boxing"],
    "brush_with": ["brush", "brushing"],
    "carry": ["carry", "carrying"],
    "catch": ["catch", "catching"],
    "cut": ["cut", "cutting"],
    "cut_with": ["cut", "cutting"],
    "drag": ["drag", "dragging"],
    "drink_with": ["drink", "drinking"],
    "eat": ["eat", "eating"],
    "hit": ["hit", "hitting"],
    "hold": ["hold", "holding"],
    "jump": ["jump", "jumping"],
    "kick": ["kick", "kicking"],
    "lie_on": ["lie", "lying"],
    "lift": ["lift", "lifting"],
    "look_out": ["look", "looking"],
    "open": ["open", "opening"],
    "pack": ["pack", "packing"],
    "peel": ["peel", "peeling"],
    "pick_up": ["pick", "picking"],
    "pour": ["pour", "pouring"],
    "push": ["push", "pushing"],
    "ride": ["ride", "riding"],
    "sip": ["sip", "sipping"],
    "sit_on": ["sit", "sitting"],
    "stick": ["stick", "sticking"],
    "stir": ["stir", "stirring"],
    "swing": ["swing", "swinging"],
    "take_photo": ["take", "taking", "photo"],
    "talk_on": ["talk", "talking"],
    "text_on": ["text", "texting"],
    "throw": ["throw", "throwing"],
    "type_on": ["type", "typing"],
    "wash": ["wash", "washing"],
    "write": ["write", "writing"],
}


class AGD20KDataset(Dataset):
    """
    AGD20K Affordance Grounding Dataset — Egocentric split.
    
    Each sample provides:
    - image: PIL Image (RGB)
    - affordance: str (affordance category name)
    - verb_gerund: str (e.g., "cutting", "holding")
    - verb_roots: List[str] (root forms for token matching)
    - gt_heatmap: np.ndarray (H, W) normalized affordance heatmap
    - object_name: str (object category)
    - prompt: str (constructed prompt for Flux)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "seen",
        image_size: int = 512,
        prompt_template: str = "a person {verb} a {object}",
    ):
        """
        Args:
            data_dir: Path to AGD20K dataset root
            split: "seen" or "unseen" — determines train/test object overlap
            image_size: Target image size for Flux input
            prompt_template: Template for constructing Flux prompts
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.prompt_template = prompt_template
        
        # Discover dataset structure
        self.samples = self._discover_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {data_dir}. "
                f"Run `python data/download_agd20k.py` first."
            )
        
        print(f"AGD20K [{split}]: {len(self.samples)} samples across "
              f"{len(set(s['affordance'] for s in self.samples))} affordances")
    
    def _discover_samples(self) -> List[Dict]:
        """
        Discover egocentric image-heatmap pairs.
        
        AGD20K structure varies by download version. This handles
        the common layout from Cross-View-AG:
        
        AGD20K/
        ├── testset/
        │   ├── egocentric/
        │   │   ├── {affordance}/
        │   │   │   ├── {object}/
        │   │   │   │   ├── *.jpg (images)
        │   │   │   │   └── *.png (heatmaps)
        
        Also handles flat layout:
        AGD20K/
        ├── egocentric/
        │   ├── {affordance}_{object}_*.jpg
        """
        samples = []
        
        # Try hierarchical structure first
        ego_dirs = [
            self.data_dir / "testset" / "egocentric",
            self.data_dir / "egocentric",
            self.data_dir / "Seen" / "testset" / "egocentric",
            self.data_dir / "Unseen" / "testset" / "egocentric",
            self.data_dir,  # fallback: scan root
        ]
        
        for ego_dir in ego_dirs:
            if not ego_dir.exists():
                continue
            
            samples.extend(self._scan_hierarchical(ego_dir))
            
            if len(samples) > 0:
                break
        
        return samples
    
    def _scan_hierarchical(self, ego_dir: Path) -> List[Dict]:
        """Scan a hierarchical directory structure for image-heatmap pairs."""
        samples = []
        
        for affordance_dir in sorted(ego_dir.iterdir()):
            if not affordance_dir.is_dir():
                continue
            
            affordance_name = affordance_dir.name
            
            # Skip if not a known affordance
            if affordance_name not in AGD20K_AFFORDANCES and \
               affordance_name.lower() not in AGD20K_AFFORDANCES:
                # Could be an object dir — scan one level deeper
                for sub_dir in sorted(affordance_dir.iterdir()):
                    if sub_dir.is_dir():
                        samples.extend(
                            self._scan_affordance_object_dir(
                                sub_dir, affordance_name, sub_dir.name
                            )
                        )
                continue
            
            # Check if affordance dir contains object subdirs or direct images
            has_subdirs = any(p.is_dir() for p in affordance_dir.iterdir())
            
            if has_subdirs:
                for object_dir in sorted(affordance_dir.iterdir()):
                    if object_dir.is_dir():
                        samples.extend(
                            self._scan_affordance_object_dir(
                                object_dir, affordance_name, object_dir.name
                            )
                        )
            else:
                # Images directly in affordance dir
                samples.extend(
                    self._scan_affordance_object_dir(
                        affordance_dir, affordance_name, "object"
                    )
                )
        
        return samples
    
    def _scan_affordance_object_dir(
        self, 
        dir_path: Path, 
        affordance: str, 
        object_name: str
    ) -> List[Dict]:
        """Scan a directory for image + heatmap pairs."""
        samples = []
        
        # Find all images
        image_files = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        )
        
        for img_path in image_files:
            # Try to find corresponding heatmap
            # Common naming: image.jpg -> image_heatmap.png or image_gt.png
            heatmap_path = self._find_heatmap(img_path)
            
            if heatmap_path is None:
                # If no separate heatmap, the .png files might BE the heatmaps
                # and .jpg files are the images
                if img_path.suffix.lower() in ('.jpg', '.jpeg'):
                    # Look for any .png with similar name
                    png_candidates = list(dir_path.glob(f"{img_path.stem}*.png"))
                    if png_candidates:
                        heatmap_path = png_candidates[0]
            
            affordance_key = affordance.lower()
            verb_gerund = AGD20K_AFFORDANCES.get(
                affordance_key, f"{affordance_key}ing"
            )
            verb_roots = AGD20K_VERB_ROOTS.get(
                affordance_key, [affordance_key]
            )
            
            prompt = self.prompt_template.format(
                verb=verb_gerund,
                object=object_name.replace("_", " ")
            )
            
            samples.append({
                "image_path": str(img_path),
                "heatmap_path": str(heatmap_path) if heatmap_path else None,
                "affordance": affordance_key,
                "verb_gerund": verb_gerund,
                "verb_roots": verb_roots,
                "object_name": object_name,
                "prompt": prompt,
            })
        
        return samples
    
    def _find_heatmap(self, img_path: Path) -> Optional[Path]:
        """Try to find a heatmap file corresponding to an image."""
        stem = img_path.stem
        parent = img_path.parent
        
        # Common heatmap naming patterns
        candidates = [
            parent / f"{stem}_heatmap.png",
            parent / f"{stem}_gt.png",
            parent / f"{stem}_mask.png",
            parent / f"{stem}.png" if img_path.suffix != '.png' else None,
            parent / "heatmaps" / f"{stem}.png",
            parent / "gt" / f"{stem}.png",
        ]
        
        for c in candidates:
            if c is not None and c.exists() and c != img_path:
                return c
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Load ground truth heatmap
        gt_heatmap = None
        if sample["heatmap_path"] and os.path.exists(sample["heatmap_path"]):
            heatmap_raw = Image.open(sample["heatmap_path"]).convert("L")
            heatmap_raw = heatmap_raw.resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            gt_heatmap = np.array(heatmap_raw, dtype=np.float32) / 255.0
            
            # Normalize to probability distribution
            total = gt_heatmap.sum()
            if total > 0:
                gt_heatmap = gt_heatmap / total
        
        return {
            "image": image,
            "gt_heatmap": gt_heatmap,
            "affordance": sample["affordance"],
            "verb_gerund": sample["verb_gerund"],
            "verb_roots": sample["verb_roots"],
            "object_name": sample["object_name"],
            "prompt": sample["prompt"],
            "image_path": sample["image_path"],
        }
    
    def get_affordance_categories(self) -> List[str]:
        """Return list of unique affordance categories in this dataset."""
        return sorted(set(s["affordance"] for s in self.samples))
    
    def get_samples_by_affordance(self, affordance: str) -> List[int]:
        """Return indices of samples with a given affordance category."""
        return [
            i for i, s in enumerate(self.samples)
            if s["affordance"] == affordance
        ]
