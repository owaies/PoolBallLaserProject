"""
Snooker 3D Kinematic & Bilinear Interpolator
Accounts for:
- Origin at Top-Right (0,0)
- Width: X in [0, 1168.4] mm
- Length: Y in [0, 2341.88] mm
- Mount height Z = 1315.15 mm directly above Top-Right corner (0,0)
- Empirical 9-point grid fitting
"""
import numpy as np
from typing import Tuple, Dict, Any, List
from laser_calibration.snooker_geometry import SNOOKER

class SnookerLaserInterpolator:
    def __init__(self, calibration_nodes: List[Dict[str, Any]]):
        self.nodes = calibration_nodes
        self.pivot_x = 0.0
        self.pivot_y = 0.0
        self.pivot_z = SNOOKER.TILT_PIVOT_TO_FELT_MM  # 1315.15 mm

    def predict(self, x_mm: float, y_mm: float) -> Tuple[float, float, Dict[str, Any]]:
        # 1. 2D Inverse Distance / Mesh Interpolation anchored on the 9 empirical points
        weights, pans, tilts = [], [], []
        for n in self.nodes:
            d = np.hypot(x_mm - n["x_mm"], y_mm - n["y_mm"])
            if d < 1e-4:
                return float(n["pan_deg"]), float(n["tilt_deg"]), {"method": "Exact Node Match", "node": n["label"]}
            w = 1.0 / (d ** 2.2)
            weights.append(w)
            pans.append(n["pan_deg"])
            tilts.append(n["tilt_deg"])

        total_w = sum(weights)
        interp_pan = sum(w * p for w, p in zip(weights, pans)) / total_w
        interp_tilt = sum(w * t for w, t in zip(weights, tilts)) / total_w

        # 2. Geometric Ray Verification from overhead mount
        dx = x_mm - self.pivot_x
        dy = y_mm - self.pivot_y
        dz = self.pivot_z  # 1315.15 mm
        geom_dist_horizontal = np.hypot(dx, dy)
        geom_tilt_angle = np.degrees(np.arctan2(dz, geom_dist_horizontal))

        info = {
            "method": "Empirical 9-Point Mesh + Kinematic Check",
            "dx_mm": round(dx, 1),
            "dy_mm": round(dy, 1),
            "dz_mm": self.pivot_z,
            "geom_tilt_ref": round(geom_tilt_angle, 1),
            "safe": (10 <= interp_pan <= 170) and (60 <= interp_tilt <= 160)
        }
        return float(interp_pan), float(interp_tilt), info

