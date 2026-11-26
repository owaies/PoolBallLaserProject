"""
Merge and Harmonize All Downloaded Datasets.
Converts all datasets to a unified YOLO format with a single consistent class map,
then splits into train/valid/test.

Sources:
  Roboflow:
    - pool_ball_v4       (YOLO format, classes: 0-15 + rack)
    - billiards           (YOLO format, classes: 0-15 + '94')
    - pool_ball_detection (YOLO format, classes: '1 ball'..'cue ball')
  GitHub:
    - snooker_ml_dataset  (Pascal VOC XML, classes: white/red/yellow/green/brown/blue/pink/black)
  Kaggle:
    - kaggle_pool_classification (Classification folders, NO bounding boxes - skip for detection)
    - kaggle_snooker_balls       (Classification crops, NO bounding boxes - skip for detection)
    - kaggle_snooker_couto       (Classification crops, NO bounding boxes - skip for detection)
    - kaggle_pool_balls          (Single image - skip)
"""
import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
MERGED_DIR = BASE_DIR / "datasets" / "merged"
FINAL_TRAIN = BASE_DIR / "datasets" / "train"
FINAL_VALID = BASE_DIR / "datasets" / "valid"
FINAL_TEST = BASE_DIR / "datasets" / "test"
LOGS_DIR = BASE_DIR / "logs"

# Setup logger
handler = logging.FileHandler(LOGS_DIR / "merge_harmonize.log", mode="w")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger = logging.getLogger("merge")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)

# ──────────────────────────────────────────────
# Unified class map: 0=cue_ball, 1-15=numbered balls
# ──────────────────────────────────────────────
UNIFIED_CLASSES: Dict[int, str] = {
    0: "cue_ball",
    1: "1_ball", 2: "2_ball", 3: "3_ball", 4: "4_ball",
    5: "5_ball", 6: "6_ball", 7: "7_ball", 8: "8_ball",
    9: "9_ball", 10: "10_ball", 11: "11_ball", 12: "12_ball",
    13: "13_ball", 14: "14_ball", 15: "15_ball",
}
NUM_CLASSES = len(UNIFIED_CLASSES)

# ──────────────────────────────────────────────
# Mapping tables for each source dataset
# ──────────────────────────────────────────────

# pool_ball_v4 data.yaml: names: ['0','1','10','11','12','13','14','15','2','3','4','5','6','7','8','9','rack']
# Index->name: 0->'0', 1->'1', 2->'10', 3->'11', 4->'12', 5->'13', 6->'14', 7->'15', 8->'2', 9->'3', 10->'4', 11->'5', 12->'6', 13->'7', 14->'8', 15->'9', 16->'rack'
POOL_BALL_V4_MAP: Dict[int, int] = {
    0: 0,   # '0'  -> cue_ball
    1: 1,   # '1'  -> 1_ball
    2: 10,  # '10' -> 10_ball
    3: 11,  # '11' -> 11_ball
    4: 12,  # '12' -> 12_ball
    5: 13,  # '13' -> 13_ball
    6: 14,  # '14' -> 14_ball
    7: 15,  # '15' -> 15_ball
    8: 2,   # '2'  -> 2_ball
    9: 3,   # '3'  -> 3_ball
    10: 4,  # '4'  -> 4_ball
    11: 5,  # '5'  -> 5_ball
    12: 6,  # '6'  -> 6_ball
    13: 7,  # '7'  -> 7_ball
    14: 8,  # '8'  -> 8_ball
    15: 9,  # '9'  -> 9_ball
    # 16: 'rack' -> SKIP (not a ball)
}

# billiards data.yaml: names: ['0','1','10','11','12','13','14','15','2','3','4','5','6','7','8','9','94']
# Same ordering as pool_ball_v4 except index 16 is '94' instead of 'rack'
BILLIARDS_MAP: Dict[int, int] = {
    0: 0,   # '0'  -> cue_ball
    1: 1,   # '1'  -> 1_ball
    2: 10,  # '10' -> 10_ball
    3: 11,  # '11' -> 11_ball
    4: 12,  # '12' -> 12_ball
    5: 13,  # '13' -> 13_ball
    6: 14,  # '14' -> 14_ball
    7: 15,  # '15' -> 15_ball
    8: 2,   # '2'  -> 2_ball
    9: 3,   # '3'  -> 3_ball
    10: 4,  # '4'  -> 4_ball
    11: 5,  # '5'  -> 5_ball
    12: 6,  # '6'  -> 6_ball
    13: 7,  # '7'  -> 7_ball
    14: 8,  # '8'  -> 8_ball
    15: 9,  # '9'  -> 9_ball
    # 16: '94' -> SKIP (unknown class)
}

# pool_ball_detection data.yaml: names: ['1 ball','10 ball','11 ball','12 ball','13 ball','14 ball','15 ball','2 ball','3 ball','4 ball','5 ball','6 ball','7 ball','8 ball','9ball','cue ball']
POOL_BALL_DET_MAP: Dict[int, int] = {
    0: 1,   # '1 ball'  -> 1_ball
    1: 10,  # '10 ball' -> 10_ball
    2: 11,  # '11 ball' -> 11_ball
    3: 12,  # '12 ball' -> 12_ball
    4: 13,  # '13 ball' -> 13_ball
    5: 14,  # '14 ball' -> 14_ball
    6: 15,  # '15 ball' -> 15_ball
    7: 2,   # '2 ball'  -> 2_ball
    8: 3,   # '3 ball'  -> 3_ball
    9: 4,   # '4 ball'  -> 4_ball
    10: 5,  # '5 ball'  -> 5_ball
    11: 6,  # '6 ball'  -> 6_ball
    12: 7,  # '7 ball'  -> 7_ball
    13: 8,  # '8 ball'  -> 8_ball
    14: 9,  # '9ball'   -> 9_ball
    15: 0,  # 'cue ball'-> cue_ball
}

# snooker_ml_dataset (Pascal VOC XML): classes are color names
# Snooker uses colors, not numbers. We map to the closest pool ball equivalent.
SNOOKER_COLOR_MAP: Dict[str, int] = {
    "white": 0,   # cue ball
    "red": 3,     # map red balls -> 3_ball (arbitrary, since snooker reds are identical)
    "yellow": 1,  # -> 1_ball
    "green": 6,   # -> 6_ball
    "brown": 7,   # -> 7_ball
    "blue": 2,    # -> 2_ball
    "pink": 4,    # -> 4_ball
    "black": 8,   # -> 8_ball
}


def remap_yolo_labels(src_dir: Path, dst_images: Path, dst_labels: Path,
                      class_map: Dict[int, int], prefix: str) -> int:
    """Copy images and remap YOLO label class IDs for a Roboflow-style dataset."""
    count = 0
    for split in ["train", "valid", "test"]:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue

            # Remap labels
            new_lines = []
            skip_image = False
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_cls = int(parts[0])
                    if old_cls not in class_map:
                        continue  # skip unknown classes like 'rack'
                    new_cls = class_map[old_cls]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

            if not new_lines:
                continue

            # Copy with unique prefix to avoid filename collisions
            new_name = f"{prefix}_{img_file.name}"
            shutil.copy2(img_file, dst_images / new_name)
            with open(dst_labels / (f"{prefix}_{img_file.stem}.txt"), "w") as f:
                f.write("\n".join(new_lines) + "\n")
            count += 1

    return count


def convert_voc_to_yolo(src_dir: Path, dst_images: Path, dst_labels: Path,
                        color_map: Dict[str, int], prefix: str) -> int:
    """Convert Pascal VOC XML annotations to YOLO format."""
    count = 0
    for split_dir in [src_dir / "table" / "train", src_dir / "table" / "test",
                      src_dir / "table-2" / "train" if (src_dir / "table-2").exists() else None]:
        if split_dir is None or not split_dir.exists():
            continue

        xml_files = list(split_dir.glob("*.xml"))
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image dimensions
                size = root.find("size")
                if size is None:
                    continue
                w_tag = size.find("WIDTH") or size.find("width")
                h_tag = size.find("HEIGHT") or size.find("height")
                if w_tag is None or h_tag is None:
                    continue
                img_w = int(w_tag.text)
                img_h = int(h_tag.text)
                if img_w == 0 or img_h == 0:
                    continue

                # Find the corresponding image
                filename = root.find("filename").text
                img_path = split_dir / filename
                if not img_path.exists():
                    continue

                yolo_lines = []
                for obj in root.findall("object"):
                    name = obj.find("name").text.lower().strip()
                    if name not in color_map:
                        continue
                    cls_id = color_map[name]

                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)

                    # Convert to YOLO format (normalized center x, y, w, h)
                    cx = ((xmin + xmax) / 2) / img_w
                    cy = ((ymin + ymax) / 2) / img_h
                    bw = (xmax - xmin) / img_w
                    bh = (ymax - ymin) / img_h

                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if not yolo_lines:
                    continue

                new_name = f"{prefix}_{img_path.name}"
                shutil.copy2(img_path, dst_images / new_name)
                with open(dst_labels / (f"{prefix}_{img_path.stem}.txt"), "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
                count += 1

            except Exception as e:
                logger.warning(f"Error parsing {xml_file.name}: {e}")

    return count


def split_merged(merged_images: Path, merged_labels: Path,
                 train_ratio=0.7, valid_ratio=0.2):
    """Split merged data into train/valid/test."""
    for d in [FINAL_TRAIN, FINAL_VALID, FINAL_TEST]:
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    all_images = sorted([f for f in merged_images.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    splits = {
        "train": all_images[:n_train],
        "valid": all_images[n_train:n_train + n_valid],
        "test": all_images[n_train + n_valid:],
    }

    dest_map = {"train": FINAL_TRAIN, "valid": FINAL_VALID, "test": FINAL_TEST}

    for split_name, files in splits.items():
        dest = dest_map[split_name]
        for img in tqdm(files, desc=f"Copying {split_name}"):
            lbl = merged_labels / (img.stem + ".txt")
            shutil.copy2(img, dest / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, dest / "labels" / lbl.name)

    return {k: len(v) for k, v in splits.items()}


def main():
    logger.info("=" * 60)
    logger.info("Starting Dataset Merge & Harmonize Pipeline")
    logger.info("=" * 60)

    # Clean old merged and split directories
    for d in [MERGED_DIR, FINAL_TRAIN, FINAL_VALID, FINAL_TEST]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            
    # Create merged directories
    merged_images = MERGED_DIR / "images"
    merged_labels = MERGED_DIR / "labels"
    merged_images.mkdir(parents=True, exist_ok=True)
    merged_labels.mkdir(parents=True, exist_ok=True)

    # 1. Pool Ball V4 (Roboflow)
    src = RAW_DIR / "pool_ball_v4"
    if src.exists():
        n = remap_yolo_labels(src, merged_images, merged_labels, POOL_BALL_V4_MAP, "pbv4")
        logger.info(f"Pool Ball V4: {n} images merged")
    else:
        logger.warning("Pool Ball V4 not found")

    # 2. Billiards (Roboflow)
    src = RAW_DIR / "billiards"
    if src.exists():
        n = remap_yolo_labels(src, merged_images, merged_labels, BILLIARDS_MAP, "bill")
        logger.info(f"Billiards: {n} images merged")
    else:
        logger.warning("Billiards not found")

    # 3. Pool Ball Detection (Roboflow)
    src = RAW_DIR / "pool_ball_detection"
    if src.exists():
        n = remap_yolo_labels(src, merged_images, merged_labels, POOL_BALL_DET_MAP, "pbdet")
        logger.info(f"Pool Ball Detection: {n} images merged")
    else:
        logger.warning("Pool Ball Detection not found")

    # 4. Snooker ML Dataset (GitHub - Pascal VOC XML)
    src = RAW_DIR / "snooker_ml_dataset"
    if src.exists():
        n = convert_voc_to_yolo(src, merged_images, merged_labels, SNOOKER_COLOR_MAP, "snk")
        logger.info(f"Snooker ML Dataset: {n} images merged")
    else:
        logger.warning("Snooker ML Dataset not found")

    # 4.5. Synthetic Pool Balls
    src = RAW_DIR / "synthetic_pool_balls"
    if src.exists():
        synth_count = 0
        img_dir = src / "images"
        lbl_dir = src / "labels"
        if img_dir.exists() and lbl_dir.exists():
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    lbl_file = lbl_dir / (img_file.stem + ".txt")
                    if lbl_file.exists():
                        new_name = f"synth_{img_file.name}"
                        shutil.copy2(img_file, merged_images / new_name)
                        shutil.copy2(lbl_file, merged_labels / f"synth_{img_file.stem}.txt")
                        synth_count += 1
        logger.info(f"Synthetic Pool Balls: {synth_count} images merged")
    else:
        logger.warning("Synthetic Pool Balls not found")

    # 5. Negative backgrounds
    neg_img_dir = BASE_DIR / "datasets" / "negatives" / "images"
    neg_lbl_dir = BASE_DIR / "datasets" / "negatives" / "labels"
    if neg_img_dir.exists():
        neg_count = 0
        for img_file in neg_img_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                lbl_file = neg_lbl_dir / (img_file.stem + ".txt")
                new_name = f"neg_{img_file.name}"
                shutil.copy2(img_file, merged_images / new_name)
                if lbl_file.exists():
                    shutil.copy2(lbl_file, merged_labels / f"neg_{img_file.stem}.txt")
                else:
                    (merged_labels / f"neg_{img_file.stem}.txt").touch()
                neg_count += 1
        logger.info(f"Negative backgrounds: {neg_count} images merged")

    # 6. Hard-negative distractors
    hn_img_dir = BASE_DIR / "datasets" / "hard_negatives" / "images"
    hn_lbl_dir = BASE_DIR / "datasets" / "hard_negatives" / "labels"
    if hn_img_dir.exists():
        hn_count = 0
        for img_file in hn_img_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                lbl_file = hn_lbl_dir / (img_file.stem + ".txt")
                new_name = f"hn_{img_file.name}"
                shutil.copy2(img_file, merged_images / new_name)
                if lbl_file.exists():
                    shutil.copy2(lbl_file, merged_labels / f"hn_{img_file.stem}.txt")
                else:
                    (merged_labels / f"hn_{img_file.stem}.txt").touch()
                hn_count += 1
        logger.info(f"Hard-negative distractors: {hn_count} images merged")

    total_merged = len(list(merged_images.glob("*")))
    logger.info(f"\nTotal merged images: {total_merged}")

    # Split into train/valid/test
    logger.info("\nSplitting into train/valid/test (70/20/10)...")
    split_counts = split_merged(merged_images, merged_labels)
    logger.info(f"Split results: {split_counts}")

    # Update dataset.yaml
    yaml_content = f"""path: datasets
train: train/images
val: valid/images
test: test/images

nc: {NUM_CLASSES}
names:
"""
    for i in range(NUM_CLASSES):
        yaml_content += f"  {i}: {UNIFIED_CLASSES[i]}\n"

    yaml_path = BASE_DIR / "configs" / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"\nUpdated {yaml_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MERGE & HARMONIZE COMPLETE")
    logger.info(f"Total images: {total_merged}")
    logger.info(f"Train: {split_counts.get('train', 0)}")
    logger.info(f"Valid: {split_counts.get('valid', 0)}")
    logger.info(f"Test:  {split_counts.get('test', 0)}")
    logger.info(f"Classes: {NUM_CLASSES}")
    for i, name in UNIFIED_CLASSES.items():
        logger.info(f"  {i}: {name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    main()
