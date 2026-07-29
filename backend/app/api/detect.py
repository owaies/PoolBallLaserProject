from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body
from backend.app.schemas.schemas import DetectionResponse, FolderDetectionResponse
from backend.app.services.yolo_service import yolo_service
from backend.app.core.config import settings
import logging

router = APIRouter()
logger = logging.getLogger("backend_logger")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    # 1. Validate File Extension
    ext = file.filename[file.filename.rfind('.'):].lower() if '.' in file.filename else ''
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(VALID_EXTENSIONS)}"
        )

    # 2. Read File content and validate size
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

    # 3. Predict and annotate
    try:
        detections, all_detections, annotated_url, debug_annotated_url, processing_time = yolo_service.predict_image(content, file.filename)
        return {
            "detections": detections,
            "all_detections": all_detections,
            "annotated_image_url": annotated_url,
            "debug_annotated_image_url": debug_annotated_url,
            "processing_time": round(processing_time, 4)
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
