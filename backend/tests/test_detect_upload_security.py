from __future__ import annotations

import asyncio
import re
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.api import detect as detect_module


def _call_detect(filename: str, data: bytes = b"image-bytes"):
    async def invoke():
        upload = UploadFile(filename=filename, file=BytesIO(data))
        return await detect_module.detect_image(upload)

    return asyncio.run(invoke())


def test_image_upload_reads_only_one_byte_beyond_configured_limit(monkeypatch):
    monkeypatch.setattr(detect_module.settings, "MAX_UPLOAD_SIZE", 4)
    monkeypatch.setattr(
        detect_module.yolo_service,
        "predict_image",
        lambda *_args: pytest.fail("inference must not run for an oversized upload"),
    )

    with pytest.raises(HTTPException) as raised:
        _call_detect("shelf.jpg", b"12345")

    assert raised.value.status_code == 413


def test_detection_uses_generated_output_name_for_untrusted_upload_filename(monkeypatch):
    captured = {}

    def fake_predict(content, filename):
        captured["content"] = content
        captured["filename"] = filename
        return [], [], "/clean.jpg", "/debug.jpg", 0.01

    monkeypatch.setattr(detect_module.yolo_service, "predict_image", fake_predict)

    _call_detect("../../outside.jpg", b"image-bytes")

    assert captured["content"] == b"image-bytes"
    assert re.fullmatch(r"[0-9a-f]{32}\.jpg", captured["filename"])


def test_invalid_image_content_returns_bad_request(monkeypatch):
    def reject_image(_content, _filename):
        raise ValueError("Invalid image file")

    monkeypatch.setattr(detect_module.yolo_service, "predict_image", reject_image)

    with pytest.raises(HTTPException) as raised:
        _call_detect("not-really-an-image.jpg")

    assert raised.value.status_code == 400
    assert raised.value.detail == "Uploaded file is not a valid image."


def test_inference_failure_does_not_expose_internal_error(monkeypatch):
    def fail_inference(_content, _filename):
        raise RuntimeError("private /srv/application/model path")

    monkeypatch.setattr(detect_module.yolo_service, "predict_image", fail_inference)

    with pytest.raises(HTTPException) as raised:
        _call_detect("shelf.jpg")

    assert raised.value.status_code == 503
    assert raised.value.detail == "Object detection is temporarily unavailable."
    assert "/srv/application" not in raised.value.detail
