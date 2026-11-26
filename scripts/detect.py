"""
Offline detection pipeline for AI-Based Pool Ball Identification.
Loads a trained YOLO model and processes images from the input directory.
Generates annotated images, CSV/JSON metadata, and a final markdown report.
"""
import os
import sys
import cv2
import json
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from ultralytics import YOLO

# Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output_images"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"
MODEL_PATH = BASE_DIR / "models" / "best.pt"

# Create necessary directories
for d in [INPUT_DIR, OUTPUT_DIR, RESULTS_DIR, LOGS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logger Configuration
def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        if name == 'detect_logger':
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(stdout_handler)
    return logger

logger = setup_logger('detect_logger', LOGS_DIR / 'detection.log')
error_logger = setup_logger('error_logger', LOGS_DIR / 'error.log', level=logging.ERROR)

def process_images() -> None:
    """Processes images from the input folder and generates annotated outputs and metadata."""
    if not MODEL_PATH.exists():
        error_logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Detection aborted: Model file missing.")
        return

    logger.info("Loading YOLO model...")
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        error_logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    image_paths = list(INPUT_DIR.glob('*.*'))
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [p for p in image_paths if p.suffix.lower() in valid_exts]

    if not image_paths:
        logger.warning(f"No valid images found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(image_paths)} images to process.")

    all_detections: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    failed_images: int = 0
    total_confidence: float = 0.0
    total_detection_time: float = 0.0
    global_det_id: int = 1

    for img_path in tqdm(image_paths, desc="Detecting pool balls"):
        try:
            start_time = time.time()
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to read image {img_path.name}")

            # Run inference
            results = model.predict(source=img, verbose=False)
            inference_time = time.time() - start_time
            total_detection_time += inference_time

            result = results[0]
            boxes = result.boxes

            for box in boxes:
                # Extract properties
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Compute centers and dimensions
                w = x2 - x1
                h = y2 - y1
                cx = x1 + (w / 2)
                cy = y1 + (h / 2)

                det_data = {
                    "Image Name": img_path.name,
                    "Detection ID": f"DET_{global_det_id:04d}",
                    "Class": cls_name,
                    "Confidence": conf,
                    "Center X": cx,
                    "Center Y": cy,
                    "Xmin": x1,
                    "Ymin": y1,
                    "Xmax": x2,
                    "Ymax": y2,
                    "Width": w,
                    "Height": h
                }
                all_detections.append(det_data)
                
                # Update statistics
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                total_confidence += conf
                
                # Annotate image
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Draw center point
                cv2.circle(img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                # Draw text (ID, Class, Conf)
                label = f"ID: DET_{global_det_id:04d} | {cls_name} ({conf:.2f})"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                global_det_id += 1

            # Save annotated image
            out_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(out_path), img)

        except Exception as e:
            error_logger.error(f"Error processing {img_path.name}: {e}", exc_info=True)
            failed_images += 1

    # Save CSV
    csv_path = RESULTS_DIR / "detections.csv"
    if all_detections:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_detections[0].keys())
            writer.writeheader()
            writer.writerows(all_detections)
        logger.info(f"Saved CSV results to {csv_path}")

    # Save JSON
    json_path = RESULTS_DIR / "detections.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_detections, f, indent=4)
    logger.info(f"Saved JSON results to {json_path}")

    # Generate Report
    avg_conf = (total_confidence / len(all_detections)) if all_detections else 0.0
    avg_speed = (total_detection_time / len(image_paths)) if image_paths else 0.0
    
    report_path = DOCS_DIR / "detection_report.md"
    generate_report(report_path, len(image_paths), len(all_detections), avg_conf, total_detection_time, avg_speed, class_counts, failed_images)
    logger.info(f"Generated detection report at {report_path}")

def generate_report(path: Path, total_imgs: int, total_dets: int, avg_conf: float, 
                   total_time: float, avg_speed: float, class_counts: Dict[str, int], 
                   failed: int) -> None:
    """Generates the Markdown detection statistics report."""
    md_content = [
        "# Offline Detection Pipeline Report",
        "",
        "## Summary Statistics",
        f"- **Total Images Processed:** {total_imgs}",
        f"- **Total Detections Found:** {total_dets}",
        f"- **Average Confidence Score:** {avg_conf:.4f}",
        f"- **Failed Images:** {failed}",
        "",
        "## Performance",
        f"- **Total Detection Time:** {total_time:.2f} seconds",
        f"- **Average Detection Speed:** {avg_speed:.4f} seconds/image",
        "",
        "## Per-Class Breakdown"
    ]
    
    if class_counts:
        for cls_name, count in sorted(class_counts.items()):
            md_content.append(f"- **{cls_name}**: {count} detections")
    else:
        md_content.append("- No objects detected.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    process_images()
