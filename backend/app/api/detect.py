from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body, Form
from backend.app.schemas.schemas import DetectionResponse, FolderDetectionResponse
from backend.app.services.yolo_service import yolo_service
from backend.app.core.config import settings
import logging

router = APIRouter()
logger = logging.getLogger("backend_logger")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.65),
    iou: float = Form(0.45)
):
    # 1. Validate confidence & iou bounds
    if not (0.10 <= confidence <= 0.95):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Confidence threshold must be between 0.10 and 0.95"
        )
    if not (0.10 <= iou <= 0.95):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="IoU threshold must be between 0.10 and 0.95"
        )

    # 2. Validate File Extension
    ext = file.filename[file.filename.rfind('.'):].lower() if '.' in file.filename else ''
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(VALID_EXTENSIONS)}"
        )

    # 3. Read File content and validate size
    try:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read upload file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading uploaded file contents"
        )

    # 4. Predict and annotate
    try:
        from PIL import Image
        import io
        from backend.app.core.game_manager import game_manager
        
        try:
            img_temp = Image.open(io.BytesIO(content))
            width, height = img_temp.size
        except Exception:
            width, height = None, None

        detections, all_detections, annotated_url, debug_annotated_url, processing_time = yolo_service.predict_image(content, file.filename, confidence, iou)
        return {
            "success": True,
            "confidence_threshold": confidence,
            "detections": detections,
            "all_detections": all_detections,
            "annotated_image_url": annotated_url,
            "debug_annotated_image_url": debug_annotated_url,
            "processing_time": round(processing_time, 4),
            "game_id": game_manager.current_game_id,
            "model_name": Path(yolo_service.model_path).name if yolo_service.model_path else "None",
            "image_width": width,
            "image_height": height
        }
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"YOLO inference error: {str(e)}"
        )

@router.post("/detect/folder", response_model=FolderDetectionResponse)
def detect_folder(folder_path: str = Body(..., embed=True)):
    # Prevent directory traversal or invalid paths
    if ".." in folder_path or folder_path.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid folder path. Relative directory traversal is blocked."
        )

    try:
        results = yolo_service.predict_folder(folder_path)
        return results
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Folder detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing folder detection: {str(e)}"
        )
