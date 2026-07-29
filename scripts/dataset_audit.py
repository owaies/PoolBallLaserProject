"""
Script to automatically inspect the dataset for incorrect labels, duplicates,
wrong class IDs, missing labels, corrupted images, and imbalanced classes.
Generates a complete dataset quality report in docs/dataset_quality_report.md.
"""
import os
import cv2
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DOCS_DIR = BASE_DIR / "docs"

def calculate_iou(box1, box2):
    """Calculate Intersection-over-Union (IoU) of two bounding boxes in normalized coords."""
    # Convert from xywh to xmin, ymin, xmax, ymax
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xmin1, ymin1 = x1 - w1/2, y1 - h1/2
    xmax1, ymax1 = x1 + w1/2, y1 + h1/2
    xmin2, ymin2 = x2 - w2/2, y2 - h2/2
    xmax2, ymax2 = x2 + w2/2, y2 + h2/2
    
    # Inter Area
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    # Union Area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def run_audit():
    logging.info("Starting dataset quality audit...")
    splits = ["train", "valid", "test"]
    
    report_lines = [
        "# Dataset Quality Audit Report",
        "",
        "This report was automatically generated to analyze the health, consistency, and annotations of the merged pool ball detection dataset.",
        ""
    ]
    
    overall_total_images = 0
    overall_corrupt_images = []
    overall_missing_labels = []
    overall_empty_labels_count = 0
    overall_out_of_bounds = []
    overall_wrong_classes = defaultdict(list)
    overall_duplicate_labels = []
    class_distribution = defaultdict(int)
    
    split_details = {}
    
    for split in splits:
        img_dir = DATASETS_DIR / split / "images"
        lbl_dir = DATASETS_DIR / split / "labels"
        
        if not img_dir.exists():
            logging.warning(f"Split directory {split} not found. Skipping.")
            continue
            
        img_files = list(img_dir.glob("*.*"))
        overall_total_images += len(img_files)
        
        split_corrupt = 0
        split_missing_lbl = 0
        split_empty_lbl = 0
        split_out_of_bounds = 0
        split_wrong_class = 0
        split_duplicates = 0
        split_box_count = 0
        
        for img_path in img_files:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
                
            # 1. Corrupted Image Check
            img = cv2.imread(str(img_path))
            if img is None:
                overall_corrupt_images.append(str(img_path.relative_to(BASE_DIR)))
                split_corrupt += 1
                continue
                
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            
            # 2. Missing Label File Check
            if not lbl_path.exists():
                overall_missing_labels.append(str(img_path.relative_to(BASE_DIR)))
                split_missing_lbl += 1
                continue
                
            # Read label contents
            boxes = []
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                
            if not lines or len([line.strip() for line in lines if line.strip()]) == 0:
                # Background image (allowed in YOLO)
                split_empty_lbl += 1
                overall_empty_labels_count += 1
                continue
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    overall_out_of_bounds.append((str(lbl_path.relative_to(BASE_DIR)), idx, f"Invalid format: {line.strip()}"))
                    split_out_of_bounds += 1
                    continue
                    
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    overall_out_of_bounds.append((str(lbl_path.relative_to(BASE_DIR)), idx, f"Non-numeric values: {line.strip()}"))
                    split_out_of_bounds += 1
                    continue
                    
                # 3. Wrong Class ID Check (Allowed classes: 0 to 15)
                if cls_id < 0 or cls_id > 15:
                    overall_wrong_classes[cls_id].append(str(lbl_path.relative_to(BASE_DIR)))
                    split_wrong_class += 1
                    continue
                    
                class_distribution[cls_id] += 1
                split_box_count += 1
                
                # 4. Out of bounds coordinates Check
                if any(c < 0.0 or c > 1.0 for c in coords):
                    overall_out_of_bounds.append((str(lbl_path.relative_to(BASE_DIR)), idx, f"Out of bounds coordinate: {coords}"))
                    split_out_of_bounds += 1
                    continue
                    
                boxes.append((cls_id, coords, idx))
                
            # 5. Duplicate Label Check (IoU > 0.95 or exact duplicates)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    c1, coord1, idx1 = boxes[i]
                    c2, coord2, idx2 = boxes[j]
                    
                    iou = calculate_iou(coord1, coord2)
                    if iou > 0.95:
                        overall_duplicate_labels.append(
                            f"{lbl_path.relative_to(BASE_DIR)}: Lines {idx1+1} & {idx2+1} overlap with IoU={iou:.3f} (Classes {c1} vs {c2})"
                        )
                        split_duplicates += 1
                        
        split_details[split] = {
            "total_images": len(img_files),
            "corrupt": split_corrupt,
            "missing_lbl": split_missing_lbl,
            "empty_lbl": split_empty_lbl,
            "out_of_bounds": split_out_of_bounds,
            "wrong_class": split_wrong_class,
            "duplicates": split_duplicates,
            "boxes": split_box_count
        }

    # Generate Summary Metrics
    report_lines.append("## Overview Table")
    report_lines.append("| Split | Total Images | Total Bounding Boxes | Corrupt Images | Missing Labels | Background Images | Out of Bounds Coordinates | Wrong Class IDs | Duplicate Labels |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|")
    
    for split in splits:
        if split not in split_details:
            continue
        d = split_details[split]
        report_lines.append(
            f"| **{split.capitalize()}** | {d['total_images']} | {d['boxes']} | {d['corrupt']} | {d['missing_lbl']} | {d['empty_lbl']} | {d['out_of_bounds']} | {d['wrong_class']} | {d['duplicates']} |"
        )
    report_lines.append("")
    
    # Wrong Class IDs Section
    report_lines.append("## Wrong Class IDs Analysis")
    if not overall_wrong_classes:
        report_lines.append("✓ **No invalid class IDs found.** All annotations lie strictly within the 0-15 range.")
    else:
        report_lines.append("> [!WARNING]")
        report_lines.append("> Invalid class IDs found outside the 0-15 range:")
        for cls_id, paths in overall_wrong_classes.items():
            report_lines.append(f"- **Class ID {cls_id}:** {len(paths)} occurrences. Example files:")
            for p in paths[:5]:
                report_lines.append(f"  - [{Path(p).name}](file:///{BASE_DIR.as_posix()}/{p})")
    report_lines.append("")

    # Duplicate Labels Section
    report_lines.append("## Duplicate Bounding Box Detections")
    if not overall_duplicate_labels:
        report_lines.append("✓ **No duplicate annotations found in the dataset.**")
    else:
        report_lines.append("> [!WARNING]")
        report_lines.append(f"> Found **{len(overall_duplicate_labels)}** duplicate annotations (overlapping > 95% IoU):")
        for dup in overall_duplicate_labels[:15]:
            report_lines.append(f"- {dup}")
        if len(overall_duplicate_labels) > 15:
            report_lines.append(f"- ... and {len(overall_duplicate_labels) - 15} more.")
    report_lines.append("")

    # Corrupt Images Section
    report_lines.append("## Corrupted Images")
    if not overall_corrupt_images:
        report_lines.append("✓ **No corrupted images found.** All image headers parsed successfully.")
    else:
        report_lines.append("> [!CAUTION]")
        report_lines.append(f"> Found **{len(overall_corrupt_images)}** corrupted images:")
        for c in overall_corrupt_images:
            report_lines.append(f"- {c}")
    report_lines.append("")

    # Class Imbalances Section
    report_lines.append("## Class Distribution and Imbalance Analysis")
    report_lines.append("| Class ID | Class Name | Instances | Percentage |")
    report_lines.append("|---|---|---|---|")
    
    class_names = {
        0: "cue_ball", 1: "1_ball", 2: "2_ball", 3: "3_ball", 4: "4_ball",
        5: "5_ball", 6: "6_ball", 7: "7_ball", 8: "8_ball", 9: "9_ball",
        10: "10_ball", 11: "11_ball", 12: "12_ball", 13: "13_ball", 14: "14_ball", 15: "15_ball"
    }
    
    total_boxes = sum(class_distribution.values())
    if total_boxes > 0:
        counts = [class_distribution[i] for i in range(16)]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_cls = np.argmin(counts)
        max_cls = np.argmax(counts)
        imbalance_ratio = counts[max_cls] / max(1, counts[min_cls])
        
        for i in range(16):
            cnt = class_distribution[i]
            pct = (cnt / total_boxes) * 100
            report_lines.append(f"| {i} | {class_names[i]} | {cnt} | {pct:.2f}% |")
            
        report_lines.append("")
        report_lines.append("### Class Balance Metrics")
        report_lines.append(f"- **Mean Instances per Class:** {mean_count:.1f}")
        report_lines.append(f"- **Standard Deviation:** {std_count:.1f}")
        report_lines.append(f"- **Most Populated Class:** Class {max_cls} ({class_names[max_cls]}) with {counts[max_cls]} instances.")
        report_lines.append(f"- **Least Populated Class:** Class {min_cls} ({class_names[min_cls]}) with {counts[min_cls]} instances.")
        report_lines.append(f"- **Class Imbalance Ratio (Max/Min):** {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 4.0:
            report_lines.append("> [!WARNING]")
            report_lines.append(f"> Significant class imbalance detected (Ratio={imbalance_ratio:.2f} > 4.0). Consider using focal loss or oversampling strip classes in training.")
    else:
        report_lines.append("No instances found.")

    # Save report
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / "dataset_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
        
    logging.info(f"Audit completed. Quality report written to {report_path}")

if __name__ == "__main__":
    run_audit()
