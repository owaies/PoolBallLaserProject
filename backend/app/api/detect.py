from pathlib import Path
from uuid import uuid4

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
    # Use only the suffix from the client filename. The original name is never
    # used as an output path because upload filenames are untrusted.
    original_name = (file.filename or "").replace("\\", "/")
    ext = f".{original_name.rsplit('.', 1)[-1].lower()}" if "." in original_name else ""
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    # Read at most one byte beyond the configured limit. Checking after an
    # unbounded read would allow oversized uploads to consume arbitrary memory.
    try:
        content = await file.read(settings.MAX_UPLOAD_SIZE + 1)
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE / (1024 * 1024):.1f}MB"
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to read uploaded image")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading uploaded file contents"
        ) from None

    # A generated filename prevents traversal and collisions in annotated output
    # storage while preserving the validated image extension for OpenCV.
    output_name = f"{uuid4().hex}{ext}"

    try:
        detections, all_detections, annotated_url, debug_annotated_url, processing_time = (
            yolo_service.predict_image(content, output_name)
        )
        return {
            "detections": detections,
            "all_detections": all_detections,
            "annotated_image_url": annotated_url,
            "debug_annotated_image_url": debug_annotated_url,
            "processing_time": round(processing_time, 4)
        }
    except ValueError as exc:
        logger.info("Rejected invalid image upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image."
        ) from exc
    except Exception:
        logger.exception("YOLO inference failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Object detection is temporarily unavailable."
        ) from None


def resolve_dataset_folder(folder_path: str) -> Path:
    """Resolve a requested folder and ensure it stays inside the dataset root."""
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError("A detection folder is required.")

    requested = Path(folder_path).expanduser()
    if not requested.is_absolute():
        requested = settings.BASE_DIR / requested

    target_dir = requested.resolve(strict=True)
    datasets_root = settings.DATASETS_DIR.resolve(strict=True)
    if not target_dir.is_dir():
        raise ValueError("Detection path must be a directory.")
    if not target_dir.is_relative_to(datasets_root):
        raise ValueError("Detection folder must be inside the datasets directory.")
    return target_dir


@router.post("/detect/folder", response_model=FolderDetectionResponse)
def detect_folder(folder_path: str = Body(..., embed=True)):
    # Keep the existing explicit traversal rejection before filesystem resolution.
    if ".." in folder_path or folder_path.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid folder path. Relative directory traversal is blocked."
        )

    try:
        target_dir = resolve_dataset_folder(folder_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection folder not found."
        ) from None
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Detection folder must be an existing directory inside the datasets directory."
        ) from None

    try:
        return yolo_service.predict_folder(str(target_dir))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection folder not found."
        ) from None
    except Exception:
        logger.exception("Folder detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Folder detection failed."
        ) from None
