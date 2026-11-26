"""
download_via_api.py
====================
Downloads negative and hard-negative datasets using the Kaggle API.

Datasets downloaded:
  1. Scene-15 (yiklunchow/scene15) - general background scenes (86MB)
  2. Sports Balls (gpiosenka/sports-balls) - tennis, golf, cricket balls (300MB)
  3. Fruits-360 (moltean/fruits) - apples, oranges, lemons, limes (762MB)
  4. Count Coins (balabaskar/count-coins-image-dataset) - coins (113MB)
"""

import os
import zipfile
import shutil
import json
from pathlib import Path

# Set Kaggle API credentials
os.environ["KAGGLE_API_TOKEN"] = "KGAT_69cb3ca1f8c25147f96d6c145756fb19"

import kaggle

# ── Project paths ──────────────────────────────────
PROJECT = Path(r"D:\Final Year Project\PoolBallLaserProject")
NEG_IMG   = PROJECT / "datasets" / "negatives" / "images"
NEG_LBL   = PROJECT / "datasets" / "negatives" / "labels"
HARD_IMG  = PROJECT / "datasets" / "hard_negatives" / "images"
HARD_LBL  = PROJECT / "datasets" / "hard_negatives" / "labels"
TEMP      = PROJECT / "datasets" / "_temp_downloads"

for d in [NEG_IMG, NEG_LBL, HARD_IMG, HARD_LBL, TEMP]:
    d.mkdir(parents=True, exist_ok=True)

# List of datasets to download
DATASETS_TO_DOWNLOAD = {
    "scene15": {
        "ref": "yiklunchow/scene15",
        "zip_name": "scene15.zip",
        "extract_to": TEMP / "scene15"
    },
    "sports_balls": {
        "ref": "samuelcortinhas/sports-balls-multiclass-image-classification",
        "zip_name": "sports-balls-multiclass-image-classification.zip",
        "extract_to": TEMP / "sports_balls"
    },
    "fruits": {
        "ref": "barisyasli/fruit360",
        "zip_name": "fruit360.zip",
        "extract_to": TEMP / "fruits"
    },
    "coins": {
        "ref": "balabaskar/count-coins-image-dataset",
        "zip_name": "count-coins-image-dataset.zip",
        "extract_to": TEMP / "coins"
    }
}


def download_dataset(name: str, config: dict):
    """Download a dataset zip from Kaggle using the Kaggle API."""
    print(f"\nDownloading dataset: {config['ref']}...")
    zip_path = TEMP / config["zip_name"]
    
    # Check if zip already exists
    if zip_path.exists():
        print(f"  Zip file {config['zip_name']} already exists. Skipping download.")
        return True
        
    try:
        kaggle.api.dataset_download_files(config["ref"], path=str(TEMP), unzip=False)
        print("  Download finished. Locating downloaded zip...")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {config['ref']}: {e}")
        return False


def extract_zips():
    """Extract downloaded zips to their respective directories."""
    print("\nExtracting zips...")
    for name, config in DATASETS_TO_DOWNLOAD.items():
        extract_dir = config["extract_to"]
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  {name} already extracted. Skipping.")
            continue
            
        extract_dir.mkdir(parents=True, exist_ok=True)
        # Find the zip file in TEMP
        zip_files = list(TEMP.glob("*.zip"))
        # Match zip file by name prefix
        zip_file = None
        for z in zip_files:
            if name in z.name.replace("-", "_") or config["zip_name"] == z.name:
                zip_file = z
                break
                
        if not zip_file:
            # Let's try matching the dataset name parts
            ref_tail = config["ref"].split("/")[-1]
            for z in zip_files:
                if ref_tail in z.name:
                    zip_file = z
                    break
                    
        if zip_file:
            print(f"  Extracting {zip_file.name} to {extract_dir}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"  Done.")
        else:
            print(f"  [WARN] Zip for {name} not found in {TEMP}")


def create_empty_label(img_path: Path, label_dir: Path):
    """Create a 0-byte YOLO label file."""
    label_path = label_dir / (img_path.stem + ".txt")
    label_path.touch(exist_ok=True)


def process_scene15():
    """Extract background negative images from Scene15."""
    print("\nProcessing Scene-15...")
    src_dir = DATASETS_TO_DOWNLOAD["scene15"]["extract_to"]
    if not src_dir.exists():
        print("  [ERROR] Scene-15 folder does not exist")
        return 0
        
    count = 0
    # Copy from numbered folders 00 through 14
    for i in range(15):
        cat = f"{i:02d}"
        cat_dir = src_dir / "15-Scene" / cat
        if not cat_dir.exists():
            # Try finding recursively
            matching = [d for d in src_dir.rglob(cat) if d.is_dir()]
            if matching:
                cat_dir = matching[0]
                
        if cat_dir.exists():
            images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
            # Take up to 25 images per category
            for img in images[:25]:
                dest = NEG_IMG / f"scene15_{cat}_{count:03d}{img.suffix}"
                shutil.copy2(img, dest)
                create_empty_label(dest, NEG_LBL)
                count += 1
                
    print(f"  -> Extracted {count} background images from Scene-15")
    return count


def process_sports_balls():
    """Extract hard-negative sports ball images."""
    print("\nProcessing Sports Balls...")
    src_dir = DATASETS_TO_DOWNLOAD["sports_balls"]["extract_to"]
    if not src_dir.exists():
        print("  [ERROR] Sports Balls folder does not exist")
        return 0
        
    count = 0
    # Non-pool-ball categories
    ball_categories = ["tennis ball", "golf ball", "cricket ball", "ping pong ball", "baseball", "basketball", "soccer ball"]
    
    # Sports-balls dataset structure: train/test/valid containing folders for each ball
    for subset in ["train", "valid", "test"]:
        subset_dir = src_dir / subset
        if not subset_dir.exists():
            continue
            
        for cat in ball_categories:
            cat_dir = subset_dir / cat
            if not cat_dir.exists():
                # Try finding recursively
                matching = list(src_dir.rglob(cat))
                if matching:
                    cat_dir = matching[0]
                    
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                # Take up to 40 images per category
                for img in images[:40]:
                    dest = HARD_IMG / f"sports_{cat.replace(' ', '_')}_{count:03d}{img.suffix}"
                    shutil.copy2(img, dest)
                    create_empty_label(dest, HARD_LBL)
                    count += 1
                    
    print(f"  -> Extracted {count} hard-negative sports ball images")
    return count


def process_fruits():
    """Extract hard-negative fruit images (apples, oranges, lemons, limes)."""
    print("\nProcessing Fruits...")
    src_dir = DATASETS_TO_DOWNLOAD["fruits"]["extract_to"]
    if not src_dir.exists():
        print("  [ERROR] Fruits folder does not exist")
        return 0
        
    count = 0
    fruit_categories = ["Apple Red", "Apple Golden", "Apple Green", "Orange", "Lemon", "Lime"]
    
    # Fruits-360 structure: fruits-360/Training and fruits-360/Test
    for subset in ["Training", "Test"]:
        # Find path recursively to accommodate nesting
        sub_dirs = list(src_dir.rglob(subset))
        if not sub_dirs:
            continue
        subset_dir = sub_dirs[0]
        
        for cat in fruit_categories:
            # Match folders starting with the fruit category name
            matching_dirs = [d for d in subset_dir.iterdir() if d.is_dir() and cat.lower() in d.name.lower()]
            for cat_dir in matching_dirs:
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                # Take up to 30 images per category folder
                for img in images[:30]:
                    dest = HARD_IMG / f"fruit_{cat_dir.name.replace(' ', '_')}_{count:03d}{img.suffix}"
                    shutil.copy2(img, dest)
                    create_empty_label(dest, HARD_LBL)
                    count += 1
                    
    print(f"  -> Extracted {count} hard-negative fruit images")
    return count


def process_coins():
    """Extract hard-negative coin images."""
    print("\nProcessing Coins...")
    src_dir = DATASETS_TO_DOWNLOAD["coins"]["extract_to"]
    if not src_dir.exists():
        print("  [ERROR] Coins folder does not exist")
        return 0
        
    count = 0
    # Copy images recursively from the coin dataset
    images = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png")) + list(src_dir.rglob("*.jpeg"))
    
    # Take up to 150 coin images
    for img in images[:150]:
        dest = HARD_IMG / f"coin_{count:03d}{img.suffix}"
        shutil.copy2(img, dest)
        create_empty_label(dest, HARD_LBL)
        count += 1
        
    print(f"  -> Extracted {count} coin images")
    return count


def main():
    print("=" * 60)
    print("  KAGGLE API DATASET DOWNLOAD & EXTRACTION")
    print("=" * 60)
    
    # Download all zips
    for name, config in DATASETS_TO_DOWNLOAD.items():
        download_dataset(name, config)
        
    # Extract all zips
    extract_zips()
    
    # Process and copy images
    neg_count = process_scene15()
    hard_count = 0
    hard_count += process_sports_balls()
    hard_count += process_fruits()
    hard_count += process_coins()
    
    # Save source metadata
    sources_neg_path = PROJECT / "datasets" / "negatives" / "sources.json"
    sources_hard_path = PROJECT / "datasets" / "hard_negatives" / "sources.json"
    
    neg_sources = [{
        "source": "Kaggle (yiklunchow/scene15)",
        "category": "background_scenes",
        "count": neg_count,
        "license": "CC0"
    }]
    
    # Re-run counts properly to match exact copied counts
    final_neg = len(list(NEG_IMG.glob("*")))
    final_hard = len(list(HARD_IMG.glob("*")))
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD & PROCESSING COMPLETED")
    print(f"  Total Negative Images:      {final_neg}")
    print(f"  Total Hard-Negative Images:  {final_hard}")
    print(f"  Grand Total:                {final_neg + final_hard}")
    print("=" * 60)


if __name__ == "__main__":
    main()
