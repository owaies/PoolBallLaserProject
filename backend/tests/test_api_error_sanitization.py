from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api import games as games_api
from backend.app.core.config import settings
from backend.app.core.game_manager import game_manager
from backend.app.main import app
from backend.app.services.restoration_service import restoration_service
from backend.app.services.yolo_service import yolo_service


client = TestClient(app)


def make_png():
    image = Image.new("RGB", (2, 2), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def raise_private_error(*_args, **_kwargs):
    raise RuntimeError("private model path: /srv/models/internal.pt")


def test_image_detection_hides_inference_exception(monkeypatch):
    monkeypatch.setattr(yolo_service, "predict_image", raise_private_error)

    response = client.post(
        "/api/detect/image",
        files={"file": ("table.png", make_png(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Image detection failed. Please try again later."
    assert "/srv/models/internal.pt" not in response.text


def test_folder_detection_hides_server_path(monkeypatch):
    def missing_folder(_path):
        raise FileNotFoundError("/srv/private/datasets/missing")

    monkeypatch.setattr(yolo_service, "predict_folder", missing_folder)

    response = client.post("/api/detect/folder", json={"folder_path": "datasets/missing"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Requested folder was not found or is not accessible."
    assert "/srv/private" not in response.text


def test_snooker_detection_hides_inference_exception(monkeypatch):
    monkeypatch.setattr(game_manager, "select_game", lambda _game_id: {"game_id": "snooker"})
    monkeypatch.setattr(yolo_service, "initialize", lambda: True)
    monkeypatch.setattr(yolo_service, "predict_image", raise_private_error)

    response = client.post(
        "/api/snooker/detect",
        files={"file": ("table.png", make_png(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Snooker detection failed. Please try again later."
    assert "/srv/models/internal.pt" not in response.text


def test_table_comparison_hides_service_exception(monkeypatch):
    monkeypatch.setattr(restoration_service, "spatial_match_balls", raise_private_error)

    response = client.post(
        "/api/snooker/compare-states",
        json={"before_balls": [], "after_balls": []},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Table state comparison failed. Please try again later."
    assert "/srv/models/internal.pt" not in response.text


def test_model_upload_hides_loader_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(
        game_manager,
        "configs",
        {"test": {"model_path": "models/test.pt"}},
    )
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)

    def reject_model(_path):
        raise RuntimeError("private loader details from /srv/models")

    monkeypatch.setattr(games_api, "YOLO", reject_model)

    response = client.post(
        "/api/games/upload-model",
        data={"game_id": "test"},
        files={"file": ("model.pt", b"not-a-model", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid YOLO model."
    assert "/srv/models" not in response.text
    assert not (tmp_path / "models" / "test.tmp.pt").exists()
