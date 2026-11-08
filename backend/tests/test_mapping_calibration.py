from __future__ import annotations

import numpy as np
import pytest

from backend.app.core.config import settings
from backend.app.services.mapping_service import MappingService


def _service_with_files(monkeypatch, tmp_path, camera_matrix, dist_coeffs):
    calibration_dir = tmp_path / "calibration" / "camera_matrix"
    calibration_dir.mkdir(parents=True)
    if camera_matrix is not None:
        np.save(calibration_dir / "camera_matrix.npy", camera_matrix)
    if dist_coeffs is not None:
        np.save(calibration_dir / "dist_coeffs.npy", dist_coeffs)

    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    service = MappingService.__new__(MappingService)
    service.camera_matrix = None
    service.dist_coeffs = None
    service.is_calibrated = False
    return service


@pytest.mark.parametrize("coefficient_count", [4, 5, 8, 12, 14])
def test_load_calibration_accepts_supported_distortion_shapes(
    monkeypatch, tmp_path, coefficient_count
):
    camera_matrix = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((1, coefficient_count), dtype=np.float64)
    service = _service_with_files(monkeypatch, tmp_path, camera_matrix, dist_coeffs)

    service.load_calibration()

    assert service.is_calibrated is True
    np.testing.assert_array_equal(service.camera_matrix, camera_matrix)
    np.testing.assert_array_equal(service.dist_coeffs, dist_coeffs)


@pytest.mark.parametrize(
    ("camera_matrix", "dist_coeffs"),
    [
        (np.zeros((2, 2)), np.zeros(5)),
        (np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros(5)),
        (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros(5)),
        (np.eye(3), np.zeros(3)),
        (np.eye(3), np.array([0.0, 0.0, np.nan, 0.0, 0.0])),
    ],
    ids=[
        "wrong-camera-matrix-shape",
        "non-finite-camera-matrix",
        "non-positive-focal-length",
        "unsupported-distortion-count",
        "non-finite-distortion-coefficient",
    ],
)
def test_load_calibration_rejects_malformed_arrays_and_falls_back(
    monkeypatch, tmp_path, camera_matrix, dist_coeffs
):
    service = _service_with_files(monkeypatch, tmp_path, camera_matrix, dist_coeffs)

    service.load_calibration()

    assert service.is_calibrated is False
    assert service.camera_matrix is None
    assert service.dist_coeffs is None


def test_missing_calibration_files_clear_previous_values(monkeypatch, tmp_path):
    (tmp_path / "calibration" / "camera_matrix").mkdir(parents=True)
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    service = MappingService.__new__(MappingService)
    service.camera_matrix = np.eye(3)
    service.dist_coeffs = np.zeros(5)
    service.is_calibrated = True

    service.load_calibration()

    assert service.is_calibrated is False
    assert service.camera_matrix is None
    assert service.dist_coeffs is None
