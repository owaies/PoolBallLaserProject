from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.schemas import CalibrationInfoResponse, MappingRequest, MappingResponse
from backend.app.services.mapping_service import mapping_service

router = APIRouter()

@router.get("/calibration", response_model=CalibrationInfoResponse)
def get_calibration_info():
    status_info = mapping_service.get_calibration_status()
    return status_info

@router.post("/mapping", response_model=MappingResponse)
def map_coordinates(request: MappingRequest):
    try:
        world_x, world_y = mapping_service.map_coordinates(request.pixel_x, request.pixel_y)
        return {
            "world_x": round(world_x, 2),
            "world_y": round(world_y, 2)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coordinate mapping error: {str(e)}"
        )
