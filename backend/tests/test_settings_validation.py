from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("CONFIDENCE_THRESHOLD", -0.01),
        ("CONFIDENCE_THRESHOLD", 1.01),
        ("IOU_THRESHOLD", -0.01),
        ("IOU_THRESHOLD", 1.01),
        ("IMAGE_SIZE", 0),
        ("IMAGE_SIZE", -640),
        ("MAX_UPLOAD_SIZE", 0),
        ("MAX_UPLOAD_SIZE", -1),
        ("ADMIN_JWT_EXPIRE_HOURS", 0),
        ("ADMIN_JWT_EXPIRE_HOURS", -1),
        ("PORT", 0),
        ("PORT", 65536),
    ],
)
def test_rejects_out_of_range_runtime_settings(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_accepts_threshold_and_server_setting_boundaries() -> None:
    settings = Settings(
        _env_file=None,
        CONFIDENCE_THRESHOLD=0.0,
        IOU_THRESHOLD=1.0,
        IMAGE_SIZE=32,
        MAX_UPLOAD_SIZE=1,
        ADMIN_JWT_EXPIRE_HOURS=1,
        PORT=65535,
    )

    assert settings.CONFIDENCE_THRESHOLD == 0.0
    assert settings.IOU_THRESHOLD == 1.0
    assert settings.IMAGE_SIZE == 32
    assert settings.MAX_UPLOAD_SIZE == 1
    assert settings.ADMIN_JWT_EXPIRE_HOURS == 1
    assert settings.PORT == 65535


def test_default_runtime_settings_remain_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.CONFIDENCE_THRESHOLD == 0.60
    assert settings.IOU_THRESHOLD == 0.45
    assert settings.IMAGE_SIZE == 640
    assert settings.MAX_UPLOAD_SIZE == 10 * 1024 * 1024
    assert settings.PORT == 8000
