"""
OpenCV Red Laser Spot Detector
Isolates bright red laser spot and computes sub-pixel centroid.
"""
import cv2
import numpy as np
from typing import Optional, Tuple

def detect_laser_spot(image_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Separate channels
    b, g, r = cv2.split(image_bgr)
    diff = cv2.subtract(r, np.maximum(b, g))
    
    # Strict mask for laser brilliance
    mask = (diff > 45) & (r > 190)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_pt = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            continue
        m = cv2.moments(cnt)
        if m["m00"] > 0:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            score = float(area) + float(r[int(cy), int(cx)])
            if score > best_score:
                best_score = score
                best_pt = (cx, cy, min(1.0, score / 450.0))
    return best_pt

