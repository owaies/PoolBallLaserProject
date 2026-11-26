"""
YOLOv8 Snooker Ball Identification Module
Supports 8 canonical snooker classes:
- 15 individual Red balls
- Yellow, Green, Brown, Blue, Pink, Black
- White Cue Ball
Computes real-world table coordinates in mm via homography.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

SNOOKER_BALL_CLASSES = {
    0: "black-ball",
    1: "blue-ball",
    2: "brown-ball",
    3: "green-ball",
    4: "pink-ball",
    5: "red-ball",
    6: "white-ball",
    7: "yellow-ball"
}

@dataclass
class SnookerBall:
    id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_px: Tuple[float, float]
    table_x_mm: Optional[float] = None
    table_y_mm: Optional[float] = None

class SnookerBallDetector:
    def __init__(self, model_path: str = "models/snooker/best.pt"):
        self.model = None
        self.tracker_id_counter = 1
        import os
        if not os.path.isabs(model_path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base, model_path)
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Loaded trained YOLOv8 snooker model from {model_path} with {len(self.model.names)} classes")
        except Exception as e:
            print(f"YOLOv8 detector running in fallback/synthetic mode ({e})")

    def detect(self, image_bgr: np.ndarray, homography=None, conf: float = 0.20) -> List[SnookerBall]:
        balls: List[SnookerBall] = []
        if image_bgr is None:
            return balls

        if self.model is not None:
            results = self.model(image_bgr, verbose=False, conf=conf, imgsz=1024, iou=0.45)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if hasattr(self.model, "names") and isinstance(self.model.names, dict) and cls_id in self.model.names:
                        label = self.model.names[cls_id]
                    else:
                        label = SNOOKER_BALL_CLASSES.get(cls_id, f"Ball_{cls_id}")

                    tx, ty = None, None
                    if homography is not None:
                        table_pos = homography.pixel_to_table(cx, cy)
                        if table_pos:
                            tx, ty = table_pos

                    ball = SnookerBall(
                        id=self.tracker_id_counter,
                        label=label,
                        confidence=conf_val,
                        bbox=(x1, y1, x2, y2),
                        center_px=(cx, cy),
                        table_x_mm=round(tx, 1) if tx is not None else None,
                        table_y_mm=round(ty, 1) if ty is not None else None
                    )
                    self.tracker_id_counter += 1
                    balls.append(ball)

        return self._suppress_duplicates(balls)

    def _suppress_duplicates(self, balls: List[SnookerBall]) -> List[SnookerBall]:
        unique: List[SnookerBall] = []
        for b in balls:
            is_dup = False
            for u in unique:
                # Regulation snooker balls touching each other are ~50mm apart center-to-center.
                # Only suppress duplicates if detections are basically overlapping on the exact same ball (< 22mm).
                if b.table_x_mm is not None and u.table_x_mm is not None:
                    dist = np.hypot(b.table_x_mm - u.table_x_mm, b.table_y_mm - u.table_y_mm)
                    if dist < 22.0:
                        is_dup = True
                        break
                else:
                    pdist = np.hypot(b.center_px[0] - u.center_px[0], b.center_px[1] - u.center_px[1])
                    if pdist < 14.0:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(b)
        return unique

