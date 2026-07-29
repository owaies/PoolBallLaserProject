from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.schemas import ProjectStatisticsResponse
from backend.app.core.config import settings
from backend.app.services.yolo_service import yolo_service
import csv
import logging
from pathlib import Path

router = APIRouter()
logger = logging.getLogger("backend_logger")

@router.get("/statistics", response_model=ProjectStatisticsResponse)
def get_statistics():
    try:
        # 1. Count images in dataset folders
        dataset_dir = settings.BASE_DIR / "datasets"
        num_images = 0
        for split in ["train", "valid", "test"]:
            img_dir = dataset_dir / split / "images"
            if img_dir.exists():
                num_images += len(list(img_dir.glob("*.*")))

        # 2. Read historical detections from results/detections.csv
        csv_file = settings.BASE_DIR / "results" / "detections.csv"
        num_detections = 0
        avg_confidence = 0.0
        
        if csv_file.exists():
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    confidences = []
                    for row in reader:
                        num_detections += 1
                        if "Confidence" in row:
                            confidences.append(float(row["Confidence"]))
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
            except Exception as e:
                logger.warning(f"Error reading detections statistics CSV: {e}")

        # 3. Model details
        model_version = "YOLOv8 Nano"
        
        # Get training date (modification time of best.pt)
        best_pt_path = settings.BASE_DIR / "models" / "best.pt"
        training_date = None
        if best_pt_path.exists():
            mtime = best_pt_path.stat().st_mtime
            import datetime
            training_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

        return {
            "number_of_images": num_images,
            "number_of_detections": num_detections,
            "average_confidence": round(avg_confidence, 4),
            "model_version": model_version,
            "training_date": training_date
        }
    except Exception as e:
        logger.error(f"Failed to gather project statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error compiling project statistics: {str(e)}"
        )
