"""
Script to merge multiple datasets from the 'datasets/raw' directory into 'datasets/merged'.
"""
import os
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DIR = Path('../datasets/raw')
MERGED_DIR = Path('../datasets/merged')
MERGED_IMAGES_DIR = MERGED_DIR / 'images'
MERGED_LABELS_DIR = MERGED_DIR / 'labels'

def setup_directories():
    """Create necessary directories if they don't exist."""
    MERGED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

def merge_datasets():
    """Merge all images and labels from raw datasets into a single merged dataset."""
    setup_directories()
    
    if not RAW_DIR.exists():
        logging.error(f"Raw directory {RAW_DIR} does not exist.")
        return

    # Counter for renaming to avoid conflicts
    global_counter = 0

    dataset_folders = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    
    if not dataset_folders:
        logging.warning("No datasets found in raw directory.")
        return

    logging.info(f"Found {len(dataset_folders)} datasets to merge.")

    for dataset in dataset_folders:
        logging.info(f"Processing dataset: {dataset.name}")
        
        # Assuming YOLO format: images in 'images' folder, labels in 'labels' folder
        # or flat structure with .jpg and .txt together
        images = list(dataset.rglob('*.jpg')) + list(dataset.rglob('*.png')) + list(dataset.rglob('*.jpeg'))
        
        for img_path in tqdm(images, desc=f"Merging {dataset.name}"):
            # Find corresponding label
            label_name = img_path.stem + '.txt'
            # Check same dir or ../labels
            label_path = img_path.parent / label_name
            if not label_path.exists():
                label_path = img_path.parent.parent / 'labels' / label_name
            
            if label_path.exists():
                # Standardize names to prevent overwriting
                new_stem = f"{dataset.name}_{global_counter:06d}"
                new_img_path = MERGED_IMAGES_DIR / f"{new_stem}{img_path.suffix}"
                new_label_path = MERGED_LABELS_DIR / f"{new_stem}.txt"
                
                shutil.copy2(img_path, new_img_path)
                shutil.copy2(label_path, new_label_path)
                global_counter += 1
            else:
                logging.warning(f"Label missing for image {img_path.name}. Skipping.")

    logging.info(f"Successfully merged {global_counter} image-label pairs into {MERGED_DIR}.")

if __name__ == '__main__':
    # Change cwd to script location
    os.chdir(Path(__file__).parent)
    merge_datasets()
