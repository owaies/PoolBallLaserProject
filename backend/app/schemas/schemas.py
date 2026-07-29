from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    uptime: float = Field(..., example=3600.5)
    gpu_available: bool = Field(..., example=True)
    current_model: Optional[str] = Field(None, example="models/best.pt")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="yolov8n.pt")
    classes: Dict[int, str] = Field(..., example={0: "cue_ball", 1: "1_ball"})
    image_size: int = Field(..., example=640)
    confidence_threshold: float = Field(..., example=0.25)
    device: str = Field(..., example="cuda:0")

class ModelLoadRequest(BaseModel):
    model_path: str = Field(..., example="models/best.pt")

class ModelLoadResponse(BaseModel):
    success: bool = Field(..., example=True)
    message: str = Field(..., example="Model loaded successfully")

class DetectionItem(BaseModel):
    detection_id: str = Field(..., example="DET_0001")
    class_name: str = Field(..., example="8_ball")
    confidence: float = Field(..., example=0.95)
    xmin: float = Field(..., example=100.0)
    ymin: float = Field(..., example=150.0)
    xmax: float = Field(..., example=150.0)
    ymax: float = Field(..., example=200.0)
    center_x: float = Field(..., example=125.0)
    center_y: float = Field(..., example=175.0)
    width: float = Field(..., example=50.0)
    height: float = Field(..., example=50.0)
    is_accepted: bool = Field(default=True, example=True)
    rejection_reason: Optional[str] = Field(default=None, example="Outside Table ROI")
    aspect_ratio: Optional[float] = Field(default=None, example=1.0)
    circularity: Optional[float] = Field(default=None, example=0.92)

class DetectionResponse(BaseModel):
    detections: List[DetectionItem]
    all_detections: List[DetectionItem] = Field(default_factory=list)
    annotated_image_url: str = Field(..., example="/static/uploads/annotated_img.jpg")
    debug_annotated_image_url: Optional[str] = Field(default=None, example="/static/uploads/debug_annotated_img.jpg")
    processing_time: float = Field(..., example=0.15)

class FolderDetectionResponse(BaseModel):
    total_images: int = Field(..., example=10)
    total_detections: int = Field(..., example=25)
    csv_path: str = Field(..., example="results/detections.csv")
    json_path: str = Field(..., example="results/detections.json")

class CalibrationInfoResponse(BaseModel):
    camera_matrix: Optional[List[List[float]]] = Field(None, example=[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    distortion_coefficients: Optional[List[float]] = Field(None, example=[-0.2, 0.1, 0.0, 0.0, 0.0])
    is_calibrated: bool = Field(..., example=True)

class MappingRequest(BaseModel):
    pixel_x: float = Field(..., example=400.0)
    pixel_y: float = Field(..., example=300.0)

class MappingResponse(BaseModel):
    world_x: float = Field(..., example=990.5)
    world_y: float = Field(..., example=495.2)

class ProjectStatisticsResponse(BaseModel):
    number_of_images: int = Field(..., example=5709)
    number_of_detections: int = Field(..., example=38157)
    average_confidence: float = Field(..., example=0.72)
    model_version: str = Field(..., example="YOLOv8 Nano")
    training_date: Optional[str] = Field(None, example="2026-07-27")

class ErrorResponse(BaseModel):
    status: str = Field("error", example="error")
    message: str = Field(..., example="File upload failed")
    details: Optional[Any] = Field(None, example="File type not supported")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
