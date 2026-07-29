import time
from fastapi import APIRouter
from backend.app.schemas.schemas import HealthResponse
from backend.app.services.yolo_service import yolo_service

router = APIRouter()
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse)
def get_health():
    uptime = time.time() - START_TIME
    
    # Check GPU availability using torch if present, else fallback
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
        
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": round(uptime, 2),
        "gpu_available": gpu_available,
        "current_model": yolo_service.model_path
    }
