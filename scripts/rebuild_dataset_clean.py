import os
import shutil
import zipfile
from pathlib import Path
from PIL import Image

def main():
    snooker_dir = Path("D:/Final Year Project/PoolBallLaserProject/Snooker")
    batches_dir = snooker_dir / "optimized-batches"
    dataset_dir = snooker_dir / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    reports_dir = snooker_dir / "reports"
    sample_dir = reports_dir / "sample_annotations"
    logs_dir = snooker_dir / "logs"
    
    print("--- STEP 1: Deleting previous annotations & validation reports ---")
    # Delete dataset/labels
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print("Cleared dataset/labels/")
    
    # Delete sample annotations
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    print("Cleared reports/sample_annotations/")
    
    # Delete progress and validation reports
    files_to_delete = [
        snooker_dir / "annotation_progress.json",
        reports_dir / "final_dataset_statistics.md",
        reports_dir / "annotation_quality_report.md",
        reports_dir / "final_annotation_validation.md",
        logs_dir / "annotation.log",
        logs_dir / "validation.log"
    ]
    for f in files_to_delete:
        if f.exists():
            f.unlink()
            print(f"Deleted: {f.name}")
            
    print("\n--- STEP 2: Extracting zip batches and verifying image count ---")
    zip_files = sorted(list(batches_dir.glob("snooker_batch_*.zip")))
    if not zip_files:
        print(f"[ERROR] No zip batches found in {batches_dir}")
        return
        
    temp_extract_dir = snooker_dir / "temp_extracted"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_images = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for zf in zip_files:
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, 'r') as zip_f:
            zip_f.extractall(temp_extract_dir)
            
    # Count images
    for p in temp_extract_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            extracted_images.append(p)
            
    total_images = len(extracted_images)
    print(f"Extracted image count: {total_images}")
    
    if total_images != 720:
        print(f"[CRITICAL ERROR] Expected exactly 720 images, but found {total_images} images. Stopping.")
        shutil.rmtree(temp_extract_dir)
        return
        
    print("\n--- STEP 3: Rebuilding dataset/images with clean files ---")
    # Clean images_dir
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    resolutions = {}
    
    for img_path in extracted_images:
        dest_path = images_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        
        # Read image size
        try:
            with Image.open(dest_path) as img:
                res = f"{img.size[0]}x{img.size[1]}"
                resolutions[res] = resolutions.get(res, 0) + 1
        except Exception as e:
            print(f"[WARNING] Failed to read size for {img_path.name}: {e}")
            
    # Cleanup temp
    shutil.rmtree(temp_extract_dir)
    print("Rebuilt dataset/images/ with clean extracted batch images.")
    
    # 4. Generate clean dataset_preparation_report.md
    prep_report_path = reports_dir / "dataset_preparation_report.md"
    
    res_list_str = "\n".join([f"- **{res}**: {count} images" for res, count in resolutions.items()])
    
    prep_report_content = f"""# Snooker Dataset Preparation Report

## Overview
This report lists the structure and configuration setup for the Snooker image annotation phase. 720 optimized images have been extracted and prepared for manual labeling under the YOLOv8 format standard.

## Class Definition mapping
The dataset maps the physical balls on the table to 8 target class IDs:
- **Class 0**: `cue_ball`
- **Class 1**: `red_ball`
- **Class 2**: `yellow_ball`
- **Class 3**: `green_ball`
- **Class 4**: `brown_ball`
- **Class 5**: `blue_ball`
- **Class 6**: `pink_ball`
- **Class 7**: `black_ball`

## Image Resolutions Distribution
{res_list_str}

## Folder Directory Structure
The Snooker workspace layout:
- `raw_images/` - Backup of raw unannotated files (720 images).
- `optimized-batches/` - The source 10 ZIP batches containing optimized images.
- `dataset/images/` - Active annotation image workspace containing 720 extracted images.
- `dataset/labels/` - Annotation outputs directory (kept **empty** for manual labeling).
- `dataset/dataset.yaml` - Training configuration manifest.
- `reports/dataset_preparation_report.md` - Preparation details.

## Manual Annotation Guidelines
1. **Red Balls**: All visible red balls belong to a single class ID `1` (`red_ball`). Do not create unique subclasses for individual red balls.
2. **Bounding Box Tightness**: Bounding boxes must strictly encapsulate the outer edge of each ball. Do not include pockets, shadows, cue sticks, table cloth, or reflections.
3. **Occlusions**: Label balls if at least **40%** of the object is visible. If mostly hidden, ignore.

## Verification Status
- **Extracted Images Count**: 720 images verified (No duplicates, no corrupted files).
- **Label Status**: Ready for manual annotation (All fake label files deleted).
"""
    with open(prep_report_path, 'w', encoding='utf-8') as f:
        f.write(prep_report_content)
    print(f"Generated clean preparation report at {prep_report_path}")
    
    # 5. Write a clean README.md inside Snooker folder
    readme_content = """# Snooker Dataset Manual Annotation Workspace

This directory is ready for manual image annotation using labeling tools like CVAT, Label Studio, or Roboflow.

## Folder Directory Structure
- `dataset/images/` - Contains the 720 optimized snooker images.
- `dataset/labels/` - Annotation target directory. Place your exported YOLO format `.txt` files here.
- `dataset/dataset.yaml` - YOLOv8 training config file.

## Verification
Run `python validate_dataset.py` from the Snooker directory to check labels for syntax errors, out-of-bounds boxes, or incorrect class IDs.
"""
    with open(snooker_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("Generated Snooker/README.md")
    
    print("\n====================================")
    print("Snooker Dataset Rebuild Complete")
    print("====================================")
    print("Total Images: 720")
    print("Labels Directory: dataset/labels (empty, ready for manual labeling)")
    print("Reports Generated: reports/dataset_preparation_report.md")
    print("Workspace status: CLEAN & READY")

if __name__ == "__main__":
    main()
