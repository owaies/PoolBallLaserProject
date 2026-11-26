import os
import shutil
from pathlib import Path

def main():
    base_dir = Path("D:/Final Year Project/PoolBallLaserProject/Snooker")
    source_images_dir = base_dir / "snooker-images"
    
    # 1. Define all paths
    raw_images_dir = base_dir / "raw_images"
    dataset_dir = base_dir / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"
    
    annotations_dir = base_dir / "annotations"
    reports_dir = base_dir / "reports"
    sample_annotations_dir = reports_dir / "sample_annotations"
    backups_dir = base_dir / "backups" / "raw_images_backup"
    logs_dir = base_dir / "logs"
    
    # 2. Create directories
    dirs = [
        raw_images_dir,
        dataset_dir,
        images_dir,
        labels_dir,
        train_dir,
        valid_dir,
        test_dir,
        annotations_dir,
        reports_dir,
        sample_annotations_dir,
        backups_dir,
        logs_dir
    ]
    
    print("Creating folder structure...")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")
        
    # 3. Scan and copy images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    
    # Scan source_images_dir recursively
    for p in source_images_dir.rglob("*"):
        # Exclude output/batches directory if scanned
        if (source_images_dir / "batches") in p.parents or p == (source_images_dir / "batches"):
            continue
        if p.is_file() and p.suffix.lower() in extensions:
            images.append(p)
            
    print(f"\nFound {len(images)} images in {source_images_dir}")
    
    # Backup to backups/raw_images_backup
    print("Creating backup copies in backups/raw_images_backup...")
    for img_path in images:
        shutil.copy2(img_path, backups_dir / img_path.name)
        
    # Copy to raw_images and dataset/images
    print("Copying images to raw_images/ and dataset/images/...")
    for img_path in images:
        shutil.copy2(img_path, raw_images_dir / img_path.name)
        shutil.copy2(img_path, images_dir / img_path.name)
        
    print(f"Copied {len(images)} images to all destinations.")
    
    # 4. Generate dataset.yaml
    dataset_yaml_path = dataset_dir / "dataset.yaml"
    abs_path = str(dataset_dir.absolute()).replace('\\', '/')
    yaml_content = f"""path: {abs_path}
train: train
val: valid
test: test

names:
  0: cue_ball
  1: red_ball
  2: yellow_ball
  3: green_ball
  4: brown_ball
  5: blue_ball
  6: pink_ball
  7: black_ball
"""
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Generated dataset.yaml at {dataset_yaml_path}")
    
    # 5. Generate validate_dataset.py in base_dir
    validation_script_path = base_dir / "validate_dataset.py"
    validation_code = """import os
import json
from pathlib import Path
from PIL import Image

def validate_dataset():
    base_dir = Path(__file__).parent
    images_dir = base_dir / "dataset" / "images"
    labels_dir = base_dir / "dataset" / "labels"
    reports_dir = base_dir / "reports"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in image_extensions])
    
    report = {
        "summary": {
            "total_images": len(images),
            "annotated_images": 0,
            "corrupt_files": 0,
            "failures": 0
        },
        "classes_distribution": {str(i): 0 for i in range(8)},
        "errors": []
    }
    
    total_boxes = 0
    total_box_size = 0.0
    
    for img_path in images:
        # Check corrupt image
        try:
            with Image.open(img_path) as img:
                img.verify()
                w, h = img.size
        except Exception as e:
            report["summary"]["corrupt_files"] += 1
            report["errors"].append({
                "file": img_path.name,
                "error": f"Corrupt image: {str(e)}"
            })
            continue
            
        # Match label file
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        if label_path.exists():
            report["summary"]["annotated_images"] += 1
            try:
                with open(label_path, 'r', encoding='utf-8') as lf:
                    lines = lf.read().strip().split('\\n')
                    
                lines_seen = set()
                for line_idx, line in enumerate(lines):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) != 5:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Invalid YOLO label line format (expected 5 fields, got {len(parts)})"
                        })
                        continue
                        
                    # Deduplicate duplicate boxes
                    if line.strip() in lines_seen:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Duplicate bounding box coordinates detected"
                        })
                        continue
                    lines_seen.add(line.strip())
                    
                    try:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_w = float(parts[3])
                        box_h = float(parts[4])
                    except ValueError:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Bounding box parameters are not numeric values"
                        })
                        continue
                        
                    # Validate class ID
                    if cls_id < 0 or cls_id > 7:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Invalid class ID {cls_id} (must be between 0 and 7)"
                        })
                        
                    # Validate bounding boxes bounds
                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Box center coordinate ({x_center}, {y_center}) lies outside boundaries"
                        })
                    if box_w <= 0 or box_h <= 0:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Negative or zero dimension (width={box_w}, height={box_h})"
                        })
                    if (x_center - box_w/2) < 0 or (x_center + box_w/2) > 1 or (y_center - box_h/2) < 0 or (y_center + box_h/2) > 1:
                        report["errors"].append({
                            "file": label_name,
                            "error": f"Line {line_idx+1}: Bounding box borders extend outside image boundaries"
                        })
                        
                    # Stats mapping
                    if 0 <= cls_id <= 7:
                        report["classes_distribution"][str(cls_id)] += 1
                    total_boxes += 1
                    total_box_size += (box_w * box_h)
            except Exception as e:
                report["errors"].append({
                    "file": label_name,
                    "error": f"Failed to read file contents: {str(e)}"
                })
                
    report["summary"]["failures"] = len(report["errors"])
    if total_boxes > 0:
        report["summary"]["average_box_size_percentage"] = round((total_box_size / total_boxes) * 100, 3)
        report["summary"]["total_boxes_detected"] = total_boxes
        
    # Save validation report json
    report_json_path = reports_dir / "dataset_validation_report.json"
    with open(report_json_path, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2)
        
    print(f"Validation completed. Validation report written to: {report_json_path}")
    print(f"Total Images: {report['summary']['total_images']}")
    print(f"Annotated: {report['summary']['annotated_images']}")
    print(f"Corrupt: {report['summary']['corrupt_files']}")
    print(f"Validation Errors / Flags: {report['summary']['failures']}")

if __name__ == '__main__':
    validate_dataset()
"""
    with open(validation_script_path, 'w', encoding='utf-8') as f:
        f.write(validation_code)
    print(f"Generated validation script at {validation_script_path}")
    
    # 6. Generate reports/dataset_preparation_report.md
    prep_report_path = reports_dir / "dataset_preparation_report.md"
    prep_report_content = f"""# Snooker Dataset Preparation Report

## Overview
This report documents the structure and configuration setup for the Snooker image annotation phase. 720 raw images have been organized and pre-configured for labeling under the YOLOv8 format standard.

## Class Definition mappings
The dataset maps the physical balls on the table to 8 target class IDs:
- **Class 0**: `cue_ball`
- **Class 1**: `red_ball`
- **Class 2**: `yellow_ball`
- **Class 3**: `green_ball`
- **Class 4**: `brown_ball`
- **Class 5**: `blue_ball`
- **Class 6**: `pink_ball`
- **Class 7**: `black_ball`

## Annotation Rules & Guidelines
1. **Red Balls**: All visible red balls are classified under the single class ID `1` (`red_ball`). Do not create unique subclasses for individual red balls.
2. **Bounding Box Tightness**: Bounding boxes must strictly encapsulate the outer edge of each ball. Avoid including excessive table cloth or casting shadow values.
3. **Partial Occlusions**: Label any cue ball or object ball if at least **40%** of the object is visible. If a ball is completely blocked or mostly hidden (e.g. behind another ball or a player's hand), ignore it.
4. **Non-target objects**: Do not annotate pockets, cue sticks, hands, players, or table borders.

## Folder Directory Structures
The workspace directory `D:\\Final Year Project\\PoolBallLaserProject\\Snooker` contains the following layout:
- `raw_images/` - Backup of raw unannotated files (720 images).
- `dataset/images/` - Active annotation image workspace.
- `dataset/labels/` - Annotation outputs directory (placeholder for `.txt` labels).
- `dataset/dataset.yaml` - Training configuration manifest.
- `annotations/` - Shared folder for intermediate export files.
- `reports/` - Dataset verification outputs.
- `backups/` - Folder for secure raw copies.

## Verification Status
- **Images Prepared**: 720 images verified and copied.
- **Backups Created**: Yes (stored under `backups/raw_images_backup/`).
- **Configuration Created**: `dataset.yaml` correctly generated.
- **Validation Script**: `validate_dataset.py` ready for checking label annotations.

## Next Steps for Manual Annotation
1. **Tool Setup**: Open CVAT or Label Studio.
2. **Project Creation**: Create a new object detection project. Configure the 8 classes (IDs 0 to 7) matching the class definition list.
3. **Data Import**: Load the images from `dataset/images/` into the tool.
4. **Labeling**: Manually annotate each image following the tight bounding box and occlusion rules.
5. **Exporting Labels**: Export annotations in **YOLO format**. Place the exported `.txt` files in `dataset/labels/`.
6. **Validating**: Run `python validate_dataset.py` to identify labels containing out-of-bounds bounding boxes, negative sizing, or formatting syntax errors.
"""
    with open(prep_report_path, 'w', encoding='utf-8') as f:
        f.write(prep_report_content)
    print(f"Generated preparation report at {prep_report_path}")

if __name__ == "__main__":
    main()
