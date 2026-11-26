import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body, Form
from pydantic import BaseModel
from backend.app.services.yolo_service import yolo_service
from backend.app.services.restoration_service import restoration_service
from backend.app.core.config import settings

router = APIRouter(prefix="/snooker", tags=["Snooker"])
logger = logging.getLogger("backend_logger")

class BallStateSchema(BaseModel):
    id: str
    class_id: int
    class_name: str
    x: float
    y: float
    norm_x: float
    norm_y: float
    confidence: float
    color: Optional[str] = "#FFFFFF"

class TableComparePayload(BaseModel):
    before_balls: List[BallStateSchema]
    after_balls: List[BallStateSchema]
    threshold_norm: Optional[float] = 0.003

@router.post("/detect")
async def detect_snooker_photo(
    file: UploadFile = File(...),
    confidence: float = Form(0.65),
    iou: float = Form(0.45)
):
    """
    Runs YOLO inference on uploaded photo using Snooker V1 model (models/snooker/best.pt).
    """
    ext = file.filename[file.filename.rfind('.'):].lower() if '.' in file.filename else ''
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(valid_exts)}"
        )

    try:
        content = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(content))
        width, height = img.size

        # Temporarily ensure Snooker model is loaded
        from backend.app.core.game_manager import game_manager
        game_manager.select_game("snooker")
        yolo_service.initialize()

        detections, all_detections, annotated_url, debug_annotated_url, proc_time = yolo_service.predict_image(
            content, file.filename, confidence, iou
        )

        return {
            "success": True,
            "game_id": "snooker",
            "model_path": "models/snooker/best.pt",
            "image_width": width,
            "image_height": height,
            "processing_time": round(proc_time, 4),
            "detections": detections,
            "all_detections": all_detections,
            "annotated_image_url": annotated_url,
            "total_detected": len(detections)
        }
    except Exception as e:
        logger.error(f"Snooker detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Snooker YOLO inference error: {str(e)}"
        )

@router.post("/compare-states")
def compare_table_states(payload: TableComparePayload):
    """
    Compares 'Before Shot' and 'After Shot' table states to determine displaced balls & restoration queue.
    """
    try:
        before_list = [b.dict() for b in payload.before_balls]
        after_list = [a.dict() for a in payload.after_balls]

        matched_pairs = restoration_service.spatial_match_balls(before_list, after_list)
        displaced_balls = restoration_service.detect_displaced_balls(matched_pairs, payload.threshold_norm or 0.003)
        restoration_queue = restoration_service.create_restoration_queue(displaced_balls)

        return {
            "success": True,
            "matched_pairs_count": len(matched_pairs),
            "displaced_count": len(displaced_balls),
            "restoration_queue": restoration_queue
        }
    except Exception as e:
        logger.error(f"Table state comparison failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing table state comparison: {str(e)}"
        )
