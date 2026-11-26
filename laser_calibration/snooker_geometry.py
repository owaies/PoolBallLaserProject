"""
Snooker Table Geometry & Coordinate System Specification
Origin (0,0) at Top-Right inner felt corner.
+X directed towards Top-Left along the width (1168.4 mm)
+Y directed towards Bottom-Right along the length (2341.88 mm)
Table plane Z = 0.
"""
from dataclasses import dataclass
from typing import Tuple, List

@dataclass(frozen=True)
class SnookerDimensions:
    WIDTH_MM: float = 1168.40      # 46.0 inches (X-axis, width)
    LENGTH_MM: float = 2341.88     # 92.2 inches (Y-axis, length)
    POCKET_RADIUS_MM: float = 31.75 # 2.5 inches diameter / 2
    
    # Overhead mounting measurements
    FLOOR_TO_PLATE_MM: float = 1415.0
    PLATE_TO_TILT_PIVOT_MM: float = 69.85
    FLOOR_TO_FELT_MM: float = 30.0
    TILT_PIVOT_TO_FELT_MM: float = 1315.15  # 1415 - 69.85 - 30.0

SNOOKER = SnookerDimensions()

@dataclass
class SnookerPocket:
    name: str
    x_mm: float
    y_mm: float
    radius_mm: float = SNOOKER.POCKET_RADIUS_MM

def get_snooker_pockets() -> List[SnookerPocket]:
    """
    Returns the exact 6 regulation pockets for the snooker table in the top-right origin system:
    4 Corner pockets (TR, TL, BR, BL) and 2 Side/Middle pockets (MR, ML).
    Note: Top-centre and bottom-centre are calibration grid targets, NOT pockets.
    """
    half_y = SNOOKER.LENGTH_MM / 2.0
    return [
        SnookerPocket("Top-Right Pocket", 0.0, 0.0),
        SnookerPocket("Top-Left Pocket", SNOOKER.WIDTH_MM, 0.0),
        SnookerPocket("Middle-Right Pocket", 0.0, half_y),
        SnookerPocket("Middle-Left Pocket", SNOOKER.WIDTH_MM, half_y),
        SnookerPocket("Bottom-Right Pocket", 0.0, SNOOKER.LENGTH_MM),
        SnookerPocket("Bottom-Left Pocket", SNOOKER.WIDTH_MM, SNOOKER.LENGTH_MM),
    ]

def is_within_snooker_table(x_mm: float, y_mm: float) -> bool:
    return (0.0 <= x_mm <= SNOOKER.WIDTH_MM) and (0.0 <= y_mm <= SNOOKER.LENGTH_MM)

def clamp_to_snooker_table(x_mm: float, y_mm: float) -> Tuple[float, float]:
    return (max(0.0, min(SNOOKER.WIDTH_MM, float(x_mm))),
            max(0.0, min(SNOOKER.LENGTH_MM, float(y_mm))))

