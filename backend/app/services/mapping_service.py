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
        """Load intrinsic camera parameters computed in Phase 5."""
        calib_dir = settings.BASE_DIR / "calibration" / "camera_matrix"
        mtx_file = calib_dir / "camera_matrix.npy"
        dist_file = calib_dir / "dist_coeffs.npy"
        
        if mtx_file.exists() and dist_file.exists():
            try:
                self.camera_matrix = np.load(str(mtx_file))
                self.dist_coeffs = np.load(str(dist_file))
                self.is_calibrated = True
                logger.info("Loaded camera calibration parameters successfully.")
            except Exception as e:
                logger.error(f"Error loading camera calibration: {e}", exc_info=True)
                self.is_calibrated = False
        else:
            logger.warning(f"Calibration parameters not found in {calib_dir}. Coordinates will map without lens correction.")
            self.is_calibrated = False

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
        """Undistorts and transforms coordinate points from pixel space to world table millimeters."""
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
        
        return float(world_x), float(world_y)

mapping_service = MappingService()
