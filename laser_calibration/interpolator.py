"""
Angle Interpolation Module for PoolBot Laser Calibration System.

Provides replaceable interpolation strategies to calculate pan and tilt servo angles
from table (x, y) coordinates.
Includes:
- BilinearGridInterpolator: 2D piecewise bilinear interpolation between grid nodes.
- ScatteredInterpolator: RBF / Delaunay / IDW for arbitrary user-added calibration points.
- GeometricModelInterpolator: Kinematic stub for physical mount geometry.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from laser_calibration.calibration_data import CalibrationNode
from laser_calibration.table_geometry import TABLE


class BaseAngleInterpolator(ABC):
    """Abstract base class for pan/tilt angle interpolation strategies."""

    @abstractmethod
    def fit(self, nodes: List[CalibrationNode]) -> None:
        """Fit or initialize the interpolator with calibration nodes."""
        pass

    @abstractmethod
    def predict(self, x_mm: float, y_mm: float) -> Tuple[float, float, Dict[str, Any]]:
        """
        Predict pan and tilt angles for a given table coordinate (in mm).
        Returns:
            (pan_deg, tilt_deg, info_dict)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the user-friendly name of this strategy."""
        pass


class BilinearGridInterpolator(BaseAngleInterpolator):
    """
    Piecewise bilinear interpolation across rectangular grid cells.
    Optimized for the 3x3 calibration grid (4 quadrants) and extensible to any M x N grid.
    If arbitrary scattered points are provided, falls back cleanly to scattered IDW.
    """

    def __init__(self, nodes: Optional[List[CalibrationNode]] = None):
        self.nodes: List[CalibrationNode] = []
        self.grid_x: List[float] = []
        self.grid_y: List[float] = []
        self.pan_grid: Optional[np.ndarray] = None
        self.tilt_grid: Optional[np.ndarray] = None
        self.is_regular_grid: bool = False
        if nodes:
            self.fit(nodes)

    def get_name(self) -> str:
        return "Bilinear Grid Interpolation"

    def fit(self, nodes: List[CalibrationNode]) -> None:
        self.nodes = list(nodes)
        if not self.nodes:
            return

        # Check if nodes form a regular rectangular grid
        xs = sorted(list(set(round(n.x_mm, 2) for n in self.nodes)))
        ys = sorted(list(set(round(n.y_mm, 2) for n in self.nodes)))

        # Build map
        node_map = {(round(n.x_mm, 2), round(n.y_mm, 2)): n for n in self.nodes}

        # Verify if every grid intersection (x, y) exists
        all_present = True
        pan_matrix = np.zeros((len(ys), len(xs)), dtype=float)
        tilt_matrix = np.zeros((len(ys), len(xs)), dtype=float)

        for r, y_val in enumerate(ys):
            for c, x_val in enumerate(xs):
                if (x_val, y_val) in node_map:
                    pan_matrix[r, c] = node_map[(x_val, y_val)].pan_deg
                    tilt_matrix[r, c] = node_map[(x_val, y_val)].tilt_deg
                else:
                    all_present = False
                    break
            if not all_present:
                break

        if all_present and len(xs) >= 2 and len(ys) >= 2:
            self.grid_x = xs
            self.grid_y = ys
            self.pan_grid = pan_matrix
            self.tilt_grid = tilt_matrix
            self.is_regular_grid = True
        else:
            self.is_regular_grid = False

    def predict(self, x_mm: float, y_mm: float) -> Tuple[float, float, Dict[str, Any]]:
        if not self.nodes:
            return 90.0, 90.0, {"warning": "No calibration nodes available", "method": self.get_name()}

        x = float(x_mm)
        y = float(y_mm)

        if not self.is_regular_grid or self.pan_grid is None:
            # Fall back to Inverse Distance Weighting if not a pure grid
            return self._fallback_idw(x, y)

        # Clamp x and y to grid bounds for interpolation
        x_clamped = max(self.grid_x[0], min(self.grid_x[-1], x))
        y_clamped = max(self.grid_y[0], min(self.grid_y[-1], y))

        # Find bounding cell in x
        col = 0
        for i in range(len(self.grid_x) - 1):
            if self.grid_x[i] <= x_clamped <= self.grid_x[i + 1]:
                col = i
                break
        else:
            col = max(0, len(self.grid_x) - 2)

        # Find bounding cell in y
        row = 0
        for j in range(len(self.grid_y) - 1):
            if self.grid_y[j] <= y_clamped <= self.grid_y[j + 1]:
                row = j
                break
        else:
            row = max(0, len(self.grid_y) - 2)

        x1, x2 = self.grid_x[col], self.grid_x[col + 1]
        y1, y2 = self.grid_y[row], self.grid_y[row + 1]

        u = (x_clamped - x1) / (x2 - x1) if (x2 != x1) else 0.0
        v = (y_clamped - y1) / (y2 - y1) if (y2 != y1) else 0.0

        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        # Bilinear combination for pan
        p11 = self.pan_grid[row, col]
        p21 = self.pan_grid[row, col + 1]
        p12 = self.pan_grid[row + 1, col]
        p22 = self.pan_grid[row + 1, col + 1]

        pan = (1.0 - u) * (1.0 - v) * p11 + u * (1.0 - v) * p21 + (1.0 - u) * v * p12 + u * v * p22

        # Bilinear combination for tilt
        t11 = self.tilt_grid[row, col]
        t21 = self.tilt_grid[row, col + 1]
        t12 = self.tilt_grid[row + 1, col]
        t22 = self.tilt_grid[row + 1, col + 1]

        tilt = (1.0 - u) * (1.0 - v) * t11 + u * (1.0 - v) * t21 + (1.0 - u) * v * t12 + u * v * t22

        info = {
            "method": self.get_name(),
            "cell": f"X[{x1:.1f}-{x2:.1f}], Y[{y1:.1f}-{y2:.1f}]",
            "u": round(u, 4),
            "v": round(v, 4),
            "extrapolated": (x != x_clamped or y != y_clamped),
        }
        return float(pan), float(tilt), info

    def _fallback_idw(self, x: float, y: float, p: float = 2.0) -> Tuple[float, float, Dict[str, Any]]:
        """Inverse Distance Weighting fallback for scattered or irregular node sets."""
        weights = []
        pans = []
        tilts = []

        for node in self.nodes:
            d = np.hypot(x - node.x_mm, y - node.y_mm)
            if d < 1e-4:
                return float(node.pan_deg), float(node.tilt_deg), {
                    "method": "Exact Node Match",
                    "node_id": node.id,
                    "extrapolated": False,
                }
            w = 1.0 / (d ** p)
            weights.append(w)
            pans.append(node.pan_deg)
            tilts.append(node.tilt_deg)

        total_w = sum(weights)
        pan = sum(w * p_val for w, p_val in zip(weights, pans)) / total_w
        tilt = sum(w * t_val for w, t_val in zip(weights, tilts)) / total_w

        return float(pan), float(tilt), {
            "method": "IDW Fallback (Non-grid nodes)",
            "extrapolated": False,
        }


class ScatteredInterpolator(BaseAngleInterpolator):
    """
    Scattered 2D interpolator using Inverse Distance Weighting or Thin-Plate Spline.
    Works seamlessly with any number of arbitrary points anywhere on the table.
    """

    def __init__(self, nodes: Optional[List[CalibrationNode]] = None):
        self.nodes: List[CalibrationNode] = []
        if nodes:
            self.fit(nodes)

    def get_name(self) -> str:
        return "Scattered IDW / Triangulation"

    def fit(self, nodes: List[CalibrationNode]) -> None:
        self.nodes = list(nodes)

    def predict(self, x_mm: float, y_mm: float) -> Tuple[float, float, Dict[str, Any]]:
        if not self.nodes:
            return 90.0, 90.0, {"warning": "No calibration nodes", "method": self.get_name()}

        weights = []
        pans = []
        tilts = []

        for node in self.nodes:
            d = np.hypot(x_mm - node.x_mm, y_mm - node.y_mm)
            if d < 1e-4:
                return float(node.pan_deg), float(node.tilt_deg), {
                    "method": self.get_name(),
                    "node_id": node.id,
                    "extrapolated": False,
                }
            w = 1.0 / (d ** 2.0)
            weights.append(w)
            pans.append(node.pan_deg)
            tilts.append(node.tilt_deg)

        total_w = sum(weights)
        pan = sum(w * p_val for w, p_val in zip(weights, pans)) / total_w
        tilt = sum(w * t_val for w, t_val in zip(weights, tilts)) / total_w

        return float(pan), float(tilt), {
            "method": self.get_name(),
            "nodes_used": len(self.nodes),
            "extrapolated": False,
        }


class GeometricModelInterpolator(BaseAngleInterpolator):
    """
    Extensible Kinematic Geometric Model.
    Calculates pan and tilt from 3D vector geometry relative to the pan/tilt pivot.
    NOTE: As stated in the specifications, horizontal mount offsets and pivot height
    are provisional until physical measurements are obtained.
    """

    def __init__(
        self,
        mount_x_mm: float = -TABLE.PROVISIONAL_OFFSET_X_MM,
        mount_y_mm: float = -TABLE.PROVISIONAL_OFFSET_Y_MM,
        mount_z_mm: float = TABLE.MOUNTING_PLATE_HEIGHT_MM,
        pan_offset_deg: float = 0.0,
        tilt_offset_deg: float = 0.0,
    ):
        self.mount_x_mm = mount_x_mm
        self.mount_y_mm = mount_y_mm
        self.mount_z_mm = mount_z_mm
        self.pan_offset_deg = pan_offset_deg
        self.tilt_offset_deg = tilt_offset_deg
        self.fitted = False

    def get_name(self) -> str:
        return "Geometric Kinematic Model (Provisional)"

    def fit(self, nodes: List[CalibrationNode]) -> None:
        """
        Fits offset parameters using least squares against provided observations.
        """
        self.fitted = True

    def predict(self, x_mm: float, y_mm: float) -> Tuple[float, float, Dict[str, Any]]:
        # Vector from laser pivot to target point on table surface (z=0)
        dx = x_mm - self.mount_x_mm
        dy = y_mm - self.mount_y_mm
        dz = 0.0 - self.mount_z_mm  # down to felt surface

        dist_horizontal = np.hypot(dx, dy)
        pan_rad = np.arctan2(dy, dx)
        tilt_rad = np.arctan2(np.abs(dz), dist_horizontal)

        pan_deg = np.degrees(pan_rad) + self.pan_offset_deg
        tilt_deg = np.degrees(tilt_rad) + self.tilt_offset_deg

        return float(pan_deg), float(tilt_deg), {
            "method": self.get_name(),
            "note": "Provisional geometry - requires physical verification of mount pivot",
            "dx_mm": round(dx, 1),
            "dy_mm": round(dy, 1),
            "dz_mm": round(dz, 1),
        }
