import cv2
import numpy as np
import pytest

from backend.app.services.roboflow_service import RoboflowService


class EmptyInferenceClient:
    def infer(self, image_path, model_id):
        return {"predictions": []}


def make_service(tmp_path, client):
    service = RoboflowService.__new__(RoboflowService)
    service.client = client
    service.model_id = "test/model"
    service.upload_dir = tmp_path
    return service


def test_missing_roboflow_key_fails_before_writing_temp_image(tmp_path):
    service = make_service(tmp_path, client=None)

    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        service.infer_and_format(b"not-an-image", "sample.jpg")

    assert list(tmp_path.iterdir()) == []


def test_invalid_image_bytes_are_rejected_and_temp_file_is_removed(tmp_path):
    service = make_service(tmp_path, client=EmptyInferenceClient())

    with pytest.raises(ValueError, match="could not be decoded"):
        service.infer_and_format(b"not-an-image", "sample.jpg")

    assert list(tmp_path.iterdir()) == []


def test_failed_annotated_image_write_raises_error(tmp_path, monkeypatch):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    service = make_service(tmp_path, client=EmptyInferenceClient())
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="Could not save annotated"):
        service.infer_and_format(encoded.tobytes(), "sample.jpg")

    assert list(tmp_path.iterdir()) == []
