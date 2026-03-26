"""Download and prepare the UMD Part Affordance Dataset.

URL: http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/
Contents: RGB-D images of common tools with pixel-wise affordance segmentation masks.
Categories (7): grasp, cut, scoop, contain, pound, support, wrap-grasp
Split: ~11,800 training images, ~14,020 testing images
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path


UMD_BASE_URL = "http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset"

# The dataset is split into parts by object category
UMD_FILES = [
    "rgbd-dataset.tar.gz",
]


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"Downloading {url} -> {dest_path}")
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping.")
        return

    def progress_hook(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f"\r  {pct:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()


def extract_archive(archive_path, dest_dir):
    """Extract tar.gz or zip archive."""
    print(f"Extracting {archive_path} -> {dest_dir}")
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)
    print("  Done.")


def download_umd_dataset(output_dir="./data/umd_dataset"):
    """Download and extract the UMD Part Affordance Dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = output_dir / "archives"
    archive_dir.mkdir(exist_ok=True)

    for filename in UMD_FILES:
        url = f"{UMD_BASE_URL}/{filename}"
        archive_path = archive_dir / filename

        download_file(url, str(archive_path))

        if archive_path.exists():
            extract_archive(str(archive_path), str(output_dir))

    # Verify the dataset
    verify_dataset(output_dir)


def verify_dataset(dataset_dir):
    """Verify the downloaded dataset has expected structure."""
    dataset_dir = Path(dataset_dir)

    # Count images
    rgb_files = list(dataset_dir.rglob("*_rgb.png")) + list(dataset_dir.rglob("*_crop.png"))
    mask_files = list(dataset_dir.rglob("*_label.png")) + list(dataset_dir.rglob("*_mask.png"))

    print(f"\nDataset verification:")
    print(f"  RGB images found: {len(rgb_files)}")
    print(f"  Mask files found: {len(mask_files)}")
    print(f"  Dataset directory: {dataset_dir}")

    if len(rgb_files) == 0:
        print("  WARNING: No RGB images found. Check download and extraction.")
        print("  The dataset structure may differ from expected. Inspect manually.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download UMD Part Affordance Dataset")
    parser.add_argument("--output_dir", default="./data/umd_dataset",
                        help="Output directory for dataset")
    args = parser.parse_args()
    download_umd_dataset(args.output_dir)
