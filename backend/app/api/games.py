import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body, Form
from ultralytics import YOLO
from backend.app.core.game_manager import game_manager
from backend.app.services.yolo_service import yolo_service
from backend.app.core.config import settings

router = APIRouter(prefix="/games", tags=["Games"])
logger = logging.getLogger("backend_logger")

@router.get("")
def get_games():
    games = game_manager.get_games_list()
    return {
        "current_game_id": game_manager.current_game_id,
        "games": games,
        "model_info": yolo_service.get_info()
    }

@router.get("/current")
def get_current_game():
    cfg = game_manager.get_current_game_config()
    model_info = yolo_service.get_info()
    return {
        "game_id": game_manager.current_game_id,
        "config": cfg,
        "classes": cfg.get("classes", {}),
        "table": cfg.get("table", {}),
        "camera": cfg.get("camera", {}),
        "mapping": cfg.get("mapping", {}),
        "model_info": model_info
    }

@router.post("/select")
def select_game(game_id: str = Body(..., embed=True)):
    cfg = game_manager.select_game(game_id)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game '{game_id}' is not configured."
        )
    # Load corresponding model
    yolo_service.initialize()
    
    return {
        "success": True,
        "current_game_id": game_manager.current_game_id,
        "model_info": yolo_service.get_info()
    }

@router.post("/upload-model")
async def upload_model(
    game_id: str = Form(...),
    file: UploadFile = File(...)
):
    if game_id not in game_manager.configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game '{game_id}' is not configured."
        )

    # 1. Validate file extension
    ext = file.filename[file.filename.rfind('.'):].lower() if '.' in file.filename else ''
    if ext != ".pt":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported model format. Only Ultralytics YOLO '.pt' files are supported."
        )

    cfg = game_manager.configs[game_id]
    target_path = Path(settings.BASE_DIR) / cfg["model_path"]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temp file to validate first
    temp_path = target_path.with_suffix(".tmp.pt")
    
    try:
        # 2. Save file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 3. Size check (<= 25MB)
        if temp_path.stat().st_size > 26214400:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Model file exceeds maximum size limit of 25MB."
            )
            
        # 4. Validate YOLO compatibility
        try:
            temp_model = YOLO(str(temp_path))
            _ = temp_model.names
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file is not a valid YOLO model: {str(e)}"
            )

        # Move to target path
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(temp_path), str(target_path))
        
        # Reload model if uploaded for current active game
        if game_manager.current_game_id == game_id:
            yolo_service.initialize()
            
        return {
            "success": True,
            "message": f"Successfully uploaded and loaded model for '{game_id}'.",
            "model_info": yolo_service.get_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to upload model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing model upload: {str(e)}"
        )
