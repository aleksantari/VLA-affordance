# Dataset Setup

This project uses the **UMD RGB-D Part Affordance Dataset** (Myers et al., ICRA 2015) for affordance probing experiments.

The dataset is not included in this repository due to its size (~3.1 GB). Follow the instructions below to download and set up the data.

## UMD Part Affordance Dataset

### Citation

```bibtex
@inproceedings{myers2015affordance,
  title={Affordance detection of tool parts from geometric features},
  author={Myers, Austin and Teo, Ching L. and Ferm{\"u}ller, Cornelia and Aloimonos, Yiannis},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1374--1381},
  year={2015}
}
```

### Download Instructions

The original dataset is hosted at UMD:

```bash
# From the project root (VLA-affordance/)
mkdir -p data/umd_dataset
cd data/umd_dataset

# Tools dataset (2.9 GB) — 105 tools across 17 categories
wget https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz

# Clutter scenes (0.2 GB) — 3 cluttered multi-object scenes
wget https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-clutter.tar.gz

# Extract
tar xzf part-affordance-dataset-tools.tar.gz
tar xzf part-affordance-dataset-clutter.tar.gz
```

### Dataset homepage

https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/

### Expected Structure

After extraction, `data/umd_dataset/` should contain tool directories under `tools/`:

```
data/umd_dataset/
├── tools/
│   ├── bowl_01/
│   │   ├── bowl_01_00000001_rgb.jpg          # RGB image (480x640)
│   │   ├── bowl_01_00000001_depth.png        # Depth image (480x640)
│   │   ├── bowl_01_00000001_label.mat        # Affordance label (480x640, MATLAB)
│   │   ├── bowl_01_00000001_label_rank.mat   # Ranked affordance labels (480x640x7)
│   │   └── ...
│   ├── bowl_02/
│   ├── ...
│   └── trowel_05/
├── clutter/
│   ├── scene_01/
│   ├── scene_02/
│   └── scene_03/
├── category_split.txt           # Train(1)/Test(2) binary split
├── category_10_fold.txt         # Leave-one-out cross-validation folds
├── tool_categories.txt          # 17 category groupings
├── novel_split.txt              # Unseen category evaluation split
└── README.txt                   # Original dataset README
```

### Affordance Labels

**7 affordance categories + background (8 classes total):**

| ID | Affordance | Description |
|----|-----------|-------------|
| 0 | background | Non-tool pixels |
| 1 | grasp | Handle, grip region |
| 2 | cut | Blade, cutting edge |
| 3 | scoop | Concave scooping surface |
| 4 | contain | Bowl, cup interior |
| 5 | pound | Hammer head, striking surface |
| 6 | support | Flat support surface (shovel blade, turner) |
| 7 | wrap-grasp | Cylindrical grasp region |

Labels are stored in `*_label.mat` files as MATLAB arrays with variable name `gt_label` (480x640, integer values 0-7).

### Dataset Composition

**105 tools across 6 functional categories:**
- Cut (25): knives, saws, scissors, shears
- Scoop (17): scoops, spoons, trowels
- Contain (43): bowls, cups, ladles, mugs, pots
- Support (10): shovels, turners
- Pound (10): hammers, mallets, tenderizers
- Plus tools with wrap-grasp and grasp affordances

**3 cluttered scenes** with multiple overlapping tools.

### Train/Test Split

Use `category_split.txt` for the binary train/test split:
- Value 1 = train
- Value 2 = test

This is the split used by Zhang et al. (arXiv 2602.20501) and the Probing3D protocol.

### Verification

```bash
# Should show 105+ tool directories
ls -d data/umd_dataset/tools/*_*/ | wc -l

# Should show RGB images
ls data/umd_dataset/tools/bowl_01/*_rgb.jpg | head -3

# Check label files exist
ls data/umd_dataset/tools/bowl_01/*_label.mat | head -3
```

### Note on Label Format

The primary labels are in MATLAB `.mat` format. The `umd_dataset.py` data loader handles loading these via `scipy.io.loadmat`. Ground truth labels exist for every third frame; the remaining frames have automatically generated labels. For probing, we use only the ground-truth labeled frames.

### Fallback: UMD+GT (Google Drive)

If the original hosting becomes unavailable, a derivative dataset (UMD+GT) with 6 affordances (missing "support") is available via Google Drive from the [AffKpNet repo](https://github.com/ivalab/AffKpNet):
https://drive.google.com/drive/folders/1QaPBG4uavuNfdo3Po0RJMkddaEtmEEpC
