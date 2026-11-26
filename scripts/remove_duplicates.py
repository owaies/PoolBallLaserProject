"""
Script to remove duplicate or blurry/corrupted images from the merged dataset.
Outputs to the 'datasets/cleaned' directory.
"""
import os
import hashlib
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MERGED_DIR = Path('../datasets/merged')
CLEANED_DIR = Path('../datasets/cleaned')
CLEANED_IMAGES_DIR = CLEANED_DIR / 'images'
CLEANED_LABELS_DIR = CLEANED_DIR / 'labels'

def setup_directories():
    CLEANED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

def hash_image(image_path):
    """Generate MD5 hash of an image to find exact duplicates."""
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_corrupted(image_path):
    """Check if an image can be opened and is not corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # verify that it is, in fact, an image
        return False
    except Exception as e:
        return True

def clean_dataset():
    setup_directories()
    
    if not (MERGED_DIR / 'images').exists():
        logging.error("Merged images directory not found.")
        return

    images = list((MERGED_DIR / 'images').glob('*.*'))
    seen_hashes = set()
    duplicates_removed = 0
    corrupted_removed = 0
    copied = 0

    logging.info(f"Starting cleaning process on {len(images)} images.")

    for img_path in tqdm(images, desc="Cleaning Dataset"):
        # 1. Check for corruption
        if is_corrupted(img_path):
            logging.warning(f"Corrupted image found: {img_path.name}")
            corrupted_removed += 1
            continue
            
        # 2. Check for duplicates
        img_hash = hash_image(img_path)
        if img_hash in seen_hashes:
            duplicates_removed += 1
            continue
            
        seen_hashes.add(img_hash)
        
        # If valid and unique, copy to cleaned
        label_path = MERGED_DIR / 'labels' / f"{img_path.stem}.txt"
        
        if label_path.exists():
            shutil.copy2(img_path, CLEANED_IMAGES_DIR / img_path.name)
            shutil.copy2(label_path, CLEANED_LABELS_DIR / label_path.name)
            copied += 1
        else:
            logging.warning(f"Label missing during clean for {img_path.name}")

    logging.info("Dataset cleaning complete.")
    logging.info(f"Images retained: {copied}")
    logging.info(f"Duplicates removed: {duplicates_removed}")
    logging.info(f"Corrupted removed: {corrupted_removed}")

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    clean_dataset()
