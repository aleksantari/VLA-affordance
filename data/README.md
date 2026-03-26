# Dataset Setup

This project uses the **UMD RGB-D Part Affordance Dataset** (Myers et al., ICRA 2015) for affordance probing experiments.

The dataset is not included in this repository due to its size (~7.5 GB). Follow the instructions below to download and set up the data.

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

The original UMD hosting (umiacs.umd.edu) is no longer available. The dataset is hosted on Google Drive via the [AffKpNet](https://github.com/ivalab/AffKpNet) project.

**Prerequisites:**
```bash
pip install gdown
```

**Download and extract:**
```bash
# From the project root (VLA-affordance/)
mkdir -p data/umd_dataset
cd data/umd_dataset

# Main dataset - RGB-D images with affordance labels (7.46 GB)
gdown "1lWJDKyHILxOtMZ5nctxvY86igH5tFQoS" -O UMD_GT.zip

# Affordance masks (18 MB)
gdown "1bB94rvWacpXF-Uo21bGEH8Ti2egSbU63" -O UMD_GT_MASK.zip

# Category split file - defines train/test split by object instance
gdown "1FGBrBhdbtEwcVWdJxMTSi1oaaq-RrE1g" -O umd_gt_category_split.txt

# Extract
unzip UMD_GT.zip
unzip UMD_GT_MASK.zip
```

When prompted to replace `README.txt` or split files during extraction, press **A** (All) to overwrite.

### Expected Structure

After extraction, `data/umd_dataset/` should contain 104 object directories:

```
data/umd_dataset/
├── bowl_01/
│   ├── bowl_01_00000001_rgb.jpg        # RGB image (480x640)
│   ├── bowl_01_00000001_depth.png      # Depth image (480x640)
│   ├── bowl_01_00000001_label.mat      # Affordance label (480x640, MATLAB)
│   ├── bowl_01_00000001_labelid.png    # Affordance label as PNG
│   ├── bowl_01_00000001_label_rank.mat # Ranked affordance labels (480x640x7)
│   ├── bowl_01_00000001_keypoint.txt   # Affordance keypoint annotations
│   └── ...
├── bowl_02/
├── ...
├── trowel_05/
├── umd_gt_category_split.txt           # Train(1)/Test(2) split by object
├── README.txt                          # Original dataset README
└── UMD_GT.zip                          # (can delete after extraction)
```

### Dataset Details

- **104 objects** across 17 categories (bowl, cup, fork, hammer, knife, mallet, mug, pen, plate, pot, scoop, screwdriver, shovel, spoon, tenderizer, trowel)
- **6 affordance labels**: grasp (1), cut (2), scoop (3), contain (4), pound (5), wrap-grasp (6), background (0)
- **Train/test split**: defined in `umd_gt_category_split.txt` — value 1 = train, value 2 = test

### Verification

To verify the dataset was downloaded correctly:
```bash
# Should show 104 object directories
ls -d data/umd_dataset/*_*/ | wc -l

# Should show RGB images
ls data/umd_dataset/bowl_01/*_rgb.jpg | head -3
```

### Google Drive Links (for reference)

If `gdown` fails due to download quotas, you can download manually from this Google Drive folder and extract into `data/umd_dataset/`:

https://drive.google.com/drive/folders/1QaPBG4uavuNfdo3Po0RJMkddaEtmEEpC

### Note on Affordance Count

This is the **UMD+GT** version of the dataset with 6 affordances. The original UMD dataset includes a 7th affordance ("support"), but that version is no longer available for download.
