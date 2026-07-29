"""
Script to split the cleaned dataset into Train (70%), Valid (20%), and Test (10%).
Outputs to 'datasets/train', 'datasets/valid', and 'datasets/test'.
"""
import os
import shutil
import random
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLEANED_DIR = Path('../datasets/cleaned')
CLEANED_IMAGES_DIR = CLEANED_DIR / 'images'
CLEANED_LABELS_DIR = CLEANED_DIR / 'labels'

BASE_DIR = Path('../datasets')
SPLITS = {
    'train': 0.7,
    'valid': 0.2,
    'test': 0.1
}

def setup_directories():
    for split in SPLITS.keys():
        (BASE_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

def split_dataset():
    setup_directories()
    
    if not CLEANED_IMAGES_DIR.exists() or not CLEANED_LABELS_DIR.exists():
        logging.error("Cleaned directories not found. Run cleaning scripts first.")
        return

    images = list(CLEANED_IMAGES_DIR.glob('*.*'))
    if not images:
        logging.error("No images to split.")
        return

    logging.info(f"Total images to split: {len(images)}")
    
    # Shuffle for random split
    random.seed(42)  # For reproducibility
    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * SPLITS['train'])
    valid_end = train_end + int(total_images * SPLITS['valid'])

    splits_data = {
        'train': images[:train_end],
        'valid': images[train_end:valid_end],
        'test': images[valid_end:]
    }

    for split_name, split_images in splits_data.items():
        logging.info(f"Copying {len(split_images)} images to {split_name} split...")
        for img_path in tqdm(split_images, desc=f"{split_name.capitalize()}"):
            label_path = CLEANED_LABELS_DIR / f"{img_path.stem}.txt"
            
            # Copy image
            shutil.copy2(img_path, BASE_DIR / split_name / 'images' / img_path.name)
            # Copy label if exists
            if label_path.exists():
                shutil.copy2(label_path, BASE_DIR / split_name / 'labels' / label_path.name)

    logging.info("Dataset split complete.")

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    split_dataset()
