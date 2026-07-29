"""
Script to verify labels, remove orphans, and standardize annotations.
Operates on the 'datasets/cleaned' directory.
"""
import os
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLEANED_DIR = Path('../datasets/cleaned')
CLEANED_IMAGES_DIR = CLEANED_DIR / 'images'
CLEANED_LABELS_DIR = CLEANED_DIR / 'labels'

def verify_labels():
    if not CLEANED_IMAGES_DIR.exists() or not CLEANED_LABELS_DIR.exists():
        logging.error("Cleaned directories not found.")
        return

    images = set([f.stem for f in CLEANED_IMAGES_DIR.glob('*.*')])
    labels = set([f.stem for f in CLEANED_LABELS_DIR.glob('*.txt')])
    
    logging.info(f"Found {len(images)} images and {len(labels)} labels.")

    # Find orphans
    images_without_labels = images - labels
    labels_without_images = labels - images
    
    if images_without_labels:
        logging.warning(f"Found {len(images_without_labels)} images without labels. Removing them.")
        for img_stem in tqdm(images_without_labels, desc="Removing orphan images"):
            for ext in ['.jpg', '.png', '.jpeg']:
                p = CLEANED_IMAGES_DIR / f"{img_stem}{ext}"
                if p.exists():
                    p.unlink()
                    
    if labels_without_images:
        logging.warning(f"Found {len(labels_without_images)} labels without images. Removing them.")
        for label_stem in tqdm(labels_without_images, desc="Removing orphan labels"):
            (CLEANED_LABELS_DIR / f"{label_stem}.txt").unlink()

    # Validate annotations content
    invalid_labels = 0
    valid_labels_files = list(CLEANED_LABELS_DIR.glob('*.txt'))
    
    for label_path in tqdm(valid_labels_files, desc="Verifying annotation format"):
        valid_lines = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        # Ensure class is int, coordinates are floats between 0 and 1
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if all(0.0 <= c <= 1.0 for c in coords):
                            valid_lines.append(line)
                    except ValueError:
                        pass
                        
        if len(valid_lines) != len(lines):
            invalid_labels += 1
            # Rewrite only valid lines
            with open(label_path, 'w') as f:
                f.writelines(valid_lines)
            
            # If no valid lines remain, we should ideally remove the image too, but for now we leave it empty (background image for YOLO)

    logging.info(f"Verification complete. Standardized annotations in {invalid_labels} files.")

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    verify_labels()
