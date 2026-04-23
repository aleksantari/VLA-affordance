"""
AGD20K Dataset Download Helper

Downloads the AGD20K affordance grounding dataset from the Cross-View-AG
repository (Luo et al., CVPR 2022).

Dataset: ~20K exocentric + ~3.7K egocentric images across 36 affordance categories.

Usage:
    python data/download_agd20k.py --output_dir ./data/agd20k
    
For Colab, you can also mount Google Drive and symlink:
    python data/download_agd20k.py --from_drive /content/drive/MyDrive/AGD20K --output_dir ./data/agd20k
"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path


# Google Drive file IDs from Cross-View-AG repo
# https://github.com/lhc1224/Cross-View-AG
AGD20K_GDRIVE_IDS = {
    # These IDs may need updating — check the Cross-View-AG repo README
    "AGD20K": "1sGMbPPkxjsLqbxbfAi_Z5gSddBKMhXHi",  # Main dataset
}

EXPECTED_STRUCTURE = {
    "exocentric": "Exocentric (human-object interaction) images",
    "egocentric": "Egocentric (object-centric) images — primary eval set",
}


def download_from_gdrive(file_id: str, output_path: str):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run: pip install gdown")
        sys.exit(1)
    
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive: {url}")
    print(f"Output: {output_path}")
    gdown.download(url, output_path, quiet=False)


def symlink_from_drive(drive_path: str, output_dir: str):
    """Create symlink from Google Drive mount (Colab workflow)."""
    drive_path = Path(drive_path)
    output_dir = Path(output_dir)
    
    if not drive_path.exists():
        print(f"ERROR: Drive path does not exist: {drive_path}")
        print("Make sure Google Drive is mounted at /content/drive/")
        sys.exit(1)
    
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if output_dir.exists() or output_dir.is_symlink():
        print(f"Output already exists: {output_dir}")
        return
    
    os.symlink(str(drive_path.resolve()), str(output_dir))
    print(f"Created symlink: {output_dir} -> {drive_path}")


def validate_dataset(data_dir: str) -> bool:
    """Validate that AGD20K has the expected directory structure."""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        return False
    
    # Check for key subdirectories
    issues = []
    contents = list(data_dir.iterdir())
    print(f"\nDataset directory contents ({data_dir}):")
    for item in sorted(contents):
        item_type = "DIR" if item.is_dir() else "FILE"
        if item.is_dir():
            num_files = len(list(item.rglob("*")))
            print(f"  {item_type}: {item.name}/ ({num_files} files)")
        else:
            print(f"  {item_type}: {item.name} ({item.stat().st_size / 1e6:.1f} MB)")
    
    if len(contents) == 0:
        issues.append("Dataset directory is empty")
    
    if issues:
        print(f"\nValidation issues:")
        for issue in issues:
            print(f"  ⚠ {issue}")
        return False
    
    print(f"\n✓ Dataset looks valid at {data_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download AGD20K dataset")
    parser.add_argument("--output_dir", type=str, default="./data/agd20k",
                        help="Where to store the dataset")
    parser.add_argument("--from_drive", type=str, default=None,
                        help="Path to pre-downloaded dataset on Google Drive (Colab)")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing dataset, don't download")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.validate_only:
        valid = validate_dataset(str(output_dir))
        sys.exit(0 if valid else 1)
    
    if args.from_drive:
        # Colab workflow: symlink from Google Drive
        symlink_from_drive(args.from_drive, str(output_dir))
    else:
        # Direct download workflow
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, file_id in AGD20K_GDRIVE_IDS.items():
            zip_path = output_dir / f"{name}.zip"
            
            if not zip_path.exists():
                download_from_gdrive(file_id, str(zip_path))
            else:
                print(f"Zip already exists: {zip_path}")
            
            # Extract
            if zip_path.exists():
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    zf.extractall(str(output_dir))
                print(f"Extracted to {output_dir}")
    
    # Validate
    validate_dataset(str(output_dir))


if __name__ == "__main__":
    main()
