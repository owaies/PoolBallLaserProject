import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from backend.app.core.config import settings

logger = logging.getLogger("backend_logger")

class MappingService:
    def __init__(self):
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        self.H: Optional[np.ndarray] = None
        
        # Physical table size in millimeters
        self.TABLE_WIDTH_MM = 1981.0
        self.TABLE_HEIGHT_MM = 990.0
        
        self.load_calibration()
        self.compute_homography()

    def load_calibration(self) -> None:
        """Load and validate intrinsic camera parameters computed in Phase 5."""
        # Clear stale values before every load attempt so a failed reload cannot
        # leave the service marked uncalibrated while exposing old matrices.
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_calibrated = False

        calib_dir = settings.BASE_DIR / "calibration" / "camera_matrix"
        mtx_file = calib_dir / "camera_matrix.npy"
        dist_file = calib_dir / "dist_coeffs.npy"

        if not mtx_file.exists() or not dist_file.exists():
            logger.warning(
                f"Calibration parameters not found in {calib_dir}. "
                "Coordinates will map without lens correction."
            )
            return

        try:
            camera_matrix = np.load(str(mtx_file), allow_pickle=False)
            dist_coeffs = np.load(str(dist_file), allow_pickle=False)
            supported_distortion_sizes = {4, 5, 8, 12, 14}

            if (
                camera_matrix.shape != (3, 3)
                or not np.isfinite(camera_matrix).all()
                or camera_matrix[0, 0] <= 0
                or camera_matrix[1, 1] <= 0
            ):
                raise ValueError("Camera matrix must be a finite 3x3 matrix with positive focal lengths.")

            distortion_is_vector = dist_coeffs.ndim == 1 or (
                dist_coeffs.ndim == 2 and 1 in dist_coeffs.shape
            )
            if (
                not distortion_is_vector
                or dist_coeffs.size not in supported_distortion_sizes
                or not np.isfinite(dist_coeffs).all()
            ):
                raise ValueError(
                    "Distortion coefficients must be a finite 4, 5, 8, 12, or 14-value vector."
                )

            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.is_calibrated = True
            logger.info("Loaded and validated camera calibration parameters successfully.")
        except Exception as e:
            logger.error(f"Invalid camera calibration parameters in {calib_dir}: {e}", exc_info=True)

    def compute_homography(self) -> None:
        """Computes the perspective homography mapping table corners in pixels to real-world mm."""
        # Source corners of the table (matching Phase 6 mapper)
        src_pts = np.array([
            [100, 100],            # Top-Left
            [700, 100],            # Top-Right
            [700, 500],            # Bottom-Right
            [100, 500]             # Bottom-Left
        ], dtype=np.float32)

        # Destination corners (Real world plane in millimeters)
        dst_pts = np.array([
            [0, 0],
            [self.TABLE_WIDTH_MM, 0],
            [self.TABLE_WIDTH_MM, self.TABLE_HEIGHT_MM],
            [0, self.TABLE_HEIGHT_MM]
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(src_pts, dst_pts)
        logger.info("Perspective transformation Homography matrix calculated.")

    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration configuration for API response."""
        return {
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "distortion_coefficients": self.dist_coeffs.flatten().tolist() if self.dist_coeffs is not None else None,
            "is_calibrated": self.is_calibrated
        }

    def map_coordinates(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Undistort and transform finite pixel coordinates into table millimeters."""
        if not np.isfinite(pixel_x) or not np.isfinite(pixel_y):
            raise ValueError("Pixel coordinates must be finite numbers.")
        # OpenCV's point transforms use float32 here. Reject values that would
        # overflow during conversion instead of letting invalid coordinates
        # reach the native OpenCV routines.
        float32_limit = np.finfo(np.float32).max
        if abs(pixel_x) > float32_limit or abs(pixel_y) > float32_limit:
            raise ValueError("Pixel coordinates exceed the supported numeric range.")
        if self.H is None or self.H.shape != (3, 3) or not np.isfinite(self.H).all():
            raise ValueError("A valid homography matrix is required for coordinate mapping.")

        if self.is_calibrated and self.camera_matrix is not None and self.dist_coeffs is not None:
            # 1. Lens Undistortion
            pts = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            undistorted_pts = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        else:
            # Fallback directly to raw point if calibration parameters are missing
            undistorted_pts = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            
        # 2. Apply Homography
        world_pts = cv2.perspectiveTransform(undistorted_pts, self.H)
        world_x, world_y = world_pts[0][0]
        if not np.isfinite(world_x) or not np.isfinite(world_y):
            raise ValueError("Coordinate mapping produced a non-finite result.")
        
        return float(world_x), float(world_y)

mapping_service = MappingService()
