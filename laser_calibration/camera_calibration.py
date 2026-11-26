"""
OpenCV Camera Calibration & Planar Homography Module for Snooker Table
Corners selected in exact order:
0: Top-Right    -> (0.0, 0.0) mm
1: Top-Left     -> (SNOOKER.WIDTH_MM, 0.0) mm
2: Bottom-Left  -> (SNOOKER.WIDTH_MM, SNOOKER.LENGTH_MM) mm
3: Bottom-Right -> (0.0, SNOOKER.LENGTH_MM) mm
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from laser_calibration.snooker_geometry import SNOOKER

class SnookerCameraHomography:
    def __init__(self):
        self.H_img_to_table: Optional[np.ndarray] = None
        self.H_table_to_img: Optional[np.ndarray] = None
        self.corners_px: List[Tuple[float, float]] = []

    def calibrate(self, corners: List[Tuple[float, float]]) -> bool:
        if len(corners) != 4:
            return False
        self.corners_px = corners
        src = np.array(corners, dtype=np.float32)
        dst = np.array([
            [0.0, 0.0],
            [SNOOKER.WIDTH_MM, 0.0],
            [SNOOKER.WIDTH_MM, SNOOKER.LENGTH_MM],
            [0.0, SNOOKER.LENGTH_MM]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src, dst)
        if H is not None:
            self.H_img_to_table = H
            self.H_table_to_img = np.linalg.inv(H)
            return True
        return False

    def pixel_to_table(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        if self.H_img_to_table is None:
            return None
        vec = np.array([u, v, 1.0], dtype=np.float64).reshape((3, 1))
        res = np.dot(self.H_img_to_table, vec)
        w = res[2, 0]
        if abs(w) < 1e-9:
            return None
        return float(res[0, 0] / w), float(res[1, 0] / w)

    def table_to_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self.H_table_to_img is None:
            return None
        vec = np.array([x, y, 1.0], dtype=np.float64).reshape((3, 1))
        res = np.dot(self.H_table_to_img, vec)
        w = res[2, 0]
        if abs(w) < 1e-9:
            return None
        return float(res[0, 0] / w), float(res[1, 0] / w)

    def warp_top_down(self, image: np.ndarray, out_w: int = 460, out_h: int = 922) -> Optional[np.ndarray]:
        if not self.corners_px or len(self.corners_px) != 4:
            return None
        src = np.array(self.corners_px, dtype=np.float32)
        dst = np.array([[out_w, 0], [0, 0], [0, out_h], [out_w, out_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (out_w, out_h))

    def auto_calibrate_from_frame(self, image_bgr: np.ndarray) -> bool:
        """
        Automatically detect the green snooker baize quadrilateral from an overhead frame
        and calibrate homography in [TR, TL, BL, BR] order.
        """
        if image_bgr is None or image_bgr.size == 0:
            return False

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Green felt HSV range (covers indoor tungsten/daylight variation)
        mask = cv2.inRange(hsv, np.array([32, 35, 30]), np.array([95, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        h_img, w_img = image_bgr.shape[:2]
        if cv2.contourArea(largest) < 0.08 * (h_img * w_img):
            return False

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(largest)
            pts = cv2.boxPoints(rect).astype(np.float32)

        # Order into [TL, TR, BR, BL] first, then convert to [TR, TL, BL, BR]
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        ordered_tr_tl_bl_br = [
            (float(tr[0]), float(tr[1])),
            (float(tl[0]), float(tl[1])),
            (float(bl[0]), float(bl[1])),
            (float(br[0]), float(br[1])),
        ]
        return self.calibrate(ordered_tr_tl_bl_br)

