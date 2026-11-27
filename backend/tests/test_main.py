import asyncio
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from backend.app.main import app, global_exception_handler
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
    request_data = {"pixel_x": 400.0, "pixel_y": 300.0}
    response = client.post("/api/mapping", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "world_x" in data
    assert "world_y" in data
    assert abs(data["world_x"] - 990.5) < 1.0
    assert abs(data["world_y"] - 495.0) < 1.0


def test_detect_image_endpoint():
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    files = {"file": ("test_img.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "annotated_image_url" in data
    assert "processing_time" in data
    assert isinstance(data["detections"], list)


def test_detect_image_invalid_type():
    files = {"file": ("test.txt", io.BytesIO(b"dummy text"), "text/plain")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_detect_folder_invalid_path():
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
    assert isinstance(response.json(), list)


def test_cors_allows_configured_local_origin():
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_rejects_unconfigured_origin():
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_global_error_response_does_not_expose_exception_details():
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/test",
        "raw_path": b"/api/test",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }
    response = asyncio.run(
        global_exception_handler(
            Request(scope),
            RuntimeError("private filesystem path"),
        )
    )

    payload = json.loads(response.body)
    assert response.status_code == 500
    assert "details" not in payload
    assert "private filesystem path" not in response.body.decode("utf-8")
