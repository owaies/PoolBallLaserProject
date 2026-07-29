from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.schemas import ModelInfoResponse, ModelLoadRequest, ModelLoadResponse
from backend.app.services.yolo_service import yolo_service

router = APIRouter()

@router.get("/model", response_model=ModelInfoResponse)
def get_model_info():
    if not yolo_service.model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="YOLO model is not loaded"
        )
    return yolo_service.get_info()

@router.post("/model/load", response_model=ModelLoadResponse)
def load_model(request: ModelLoadRequest):
    success = yolo_service.load_model(request.model_path)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load model from path: {request.model_path}"
        )
    return {
        "success": True,
        "message": f"Successfully loaded model from {request.model_path}"
    }
