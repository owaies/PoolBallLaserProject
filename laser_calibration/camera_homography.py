"""
Camera Homography and Laser Spot Detection Module for PoolBot.

Implements perspective transformation between phone-camera image coordinates
and physical table coordinates (in mm), plus an optional red laser spot detector.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from laser_calibration.table_geometry import TABLE


@dataclass
class ImageCornerSelection:
    """Stores the 4 inner felt corners selected in the photo."""
    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_right: Tuple[float, float]
    bottom_left: Tuple[float, float]

    def to_list(self) -> List[Tuple[float, float]]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]


class CameraHomography:
    """
    Computes and manages perspective homography mapping between:
    - 2D Phone camera image pixel coordinates (u, v)
    - 2D Physical table coordinates (X, Y) in mm
    """

    def __init__(self, corners: Optional[ImageCornerSelection] = None):
        self.corners: Optional[ImageCornerSelection] = corners
        self.H_img_to_table: Optional[np.ndarray] = None
        self.H_table_to_img: Optional[np.ndarray] = None
        if corners:
            self.compute_homography(corners)

    def is_calibrated(self) -> bool:
        """Returns True if homography matrix is computed and ready."""
        return self.H_img_to_table is not None

    def compute_homography(self, corners: ImageCornerSelection) -> bool:
        """
        Calculates perspective transform matrix from 4 image points
        to the physical table corners.

        Corner ordering:
        0: Top-Left     -> (0.0, 0.0)
        1: Top-Right    -> (TABLE.INNER_LENGTH_MM, 0.0)
        2: Bottom-Right -> (TABLE.INNER_LENGTH_MM, TABLE.INNER_BREADTH_MM)
        3: Bottom-Left  -> (0.0, TABLE.INNER_BREADTH_MM)
        """
        self.corners = corners

        src_pts = np.array(corners.to_list(), dtype=np.float32)
        dst_pts = np.array([
            [0.0, 0.0],
            [TABLE.INNER_LENGTH_MM, 0.0],
            [TABLE.INNER_LENGTH_MM, TABLE.INNER_BREADTH_MM],
            [0.0, TABLE.INNER_BREADTH_MM],
        ], dtype=np.float32)

        try:
            H, _ = cv2.findHomography(src_pts, dst_pts)
            if H is not None and H.shape == (3, 3):
                self.H_img_to_table = H
                self.H_table_to_img = np.linalg.inv(H)
                return True
        except Exception as e:
            self.H_img_to_table = None
            self.H_table_to_img = None

        return False

    def image_to_table(self, u_px: float, v_px: float) -> Optional[Tuple[float, float]]:
        """
        Converts an image pixel coordinate (u, v) to physical table coordinates (X, Y) in mm.
        Returns None if not calibrated.
        """
        if self.H_img_to_table is None:
            return None

        pt = np.array([u_px, v_px, 1.0], dtype=np.float64).reshape((3, 1))
        mapped = np.dot(self.H_img_to_table, pt)
        w = mapped[2, 0]
        if abs(w) < 1e-9:
            return None

        x_mm = mapped[0, 0] / w
        y_mm = mapped[1, 0] / w
        return float(x_mm), float(y_mm)

    def table_to_image(self, x_mm: float, y_mm: float) -> Optional[Tuple[float, float]]:
        """
        Converts a physical table coordinate (x, y) in mm to image pixel coordinate (u, v).
        Returns None if not calibrated.
        """
        if self.H_table_to_img is None:
            return None

        pt = np.array([x_mm, y_mm, 1.0], dtype=np.float64).reshape((3, 1))
        mapped = np.dot(self.H_table_to_img, pt)
        w = mapped[2, 0]
        if abs(w) < 1e-9:
            return None

        u_px = mapped[0, 0] / w
        v_px = mapped[1, 0] / w
        return float(u_px), float(v_px)

    def save_calibration(self, filepath: str) -> None:
        """Save corner calibration to JSON file."""
        if not self.corners:
            return
        data = {
            "top_left": list(self.corners.top_left),
            "top_right": list(self.corners.top_right),
            "bottom_right": list(self.corners.bottom_right),
            "bottom_left": list(self.corners.bottom_left),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath: str) -> bool:
        """Load corner calibration from JSON file and recompute homography."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        corners = ImageCornerSelection(
            top_left=tuple(data["top_left"]),
            top_right=tuple(data["top_right"]),
            bottom_right=tuple(data["bottom_right"]),
            bottom_left=tuple(data["bottom_left"]),
        )
        return self.compute_homography(corners)


def detect_red_laser_spot(
    image_bgr: np.ndarray,
    min_intensity: int = 200,
    red_dominance_threshold: int = 50,
) -> Optional[Tuple[float, float, float]]:
    """
    Optional automatic red-laser spot detector.
    Looks for localized pixel clusters with high red intensity and high
    red-to-green/blue contrast: R - max(G, B) > threshold.

    Returns:
        (u_px, v_px, confidence) or None if no distinct laser spot detected.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Separate channels
    b, g, r = cv2.split(image_bgr)

    # Calculate red dominance
    max_bg = np.maximum(b, g)
    red_diff = cv2.subtract(r, max_bg)

    # Threshold for bright red
    mask = (r >= min_intensity) & (red_diff >= red_dominance_threshold)
    mask_u8 = mask.astype(np.uint8) * 255

    # Slight morphological opening to remove small noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # Find connected components / contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort contours by brightness/area
    best_pt = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Too large to be a laser dot
            continue
        # Centroid using moments
        m = cv2.moments(cnt)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            # Intensity at centroid
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= iy < image_bgr.shape[0] and 0 <= ix < image_bgr.shape[1]:
                score = float(red_diff[iy, ix]) + float(r[iy, ix])
                if score > best_score:
                    best_score = score
                    best_pt = (float(cx), float(cy), min(1.0, score / 510.0))

    return best_pt
