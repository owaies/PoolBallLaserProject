import pytest
import numpy as np
import cv2
import io
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.yolo_service import yolo_service

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    """Ensure the YOLO model singleton is loaded before tests run."""
    yolo_service.initialize()
    return yolo_service

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "gpu_available" in data
    assert "current_model" in data

def test_model_info_endpoint():
    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "classes" in data
    assert "image_size" in data
    assert "device" in data

def test_coordinate_mapping():
    # Test valid mapping coordinates
    request_data = {"pixel_x": 400.0, "pixel_y": 300.0}
    response = client.post("/api/mapping", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "world_x" in data
    assert "world_y" in data
    
    # Check that they map to the correct linear transformation
    # X = 3.301667 * 400 - 330.166667 = 990.5
    # Y = 2.475 * 300 - 247.5 = 495.0
    assert abs(data["world_x"] - 990.5) < 1.0
    assert abs(data["world_y"] - 495.0) < 1.0

def test_detect_image_endpoint():
    # 1. Create a dummy solid color JPEG image in memory
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    
    # 2. Upload to detection endpoint
    files = {"file": ("test_img.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "annotated_image_url" in data
    assert "processing_time" in data
    assert isinstance(data["detections"], list)

def test_detect_image_invalid_type():
    # Test uploading an invalid file extension (txt)
    files = {"file": ("test.txt", io.BytesIO(b"dummy text"), "text/plain")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]

def test_detect_folder_invalid_path():
    # Test directory traversal protection
    response = client.post("/api/detect/folder", json={"folder_path": "../datasets/raw"})
    assert response.status_code == 400
    assert "Relative directory traversal is blocked" in response.json()["detail"]

def test_statistics_endpoint():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "number_of_images" in data
    assert "number_of_detections" in data
    assert "average_confidence" in data

def test_calibration_endpoint():
    response = client.get("/api/calibration")
    assert response.status_code == 200
    data = response.json()
    assert "camera_matrix" in data
    assert "distortion_coefficients" in data
    assert "is_calibrated" in data

def test_logs_endpoint():
    response = client.get("/api/logs?lines=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
