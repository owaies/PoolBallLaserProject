"""
Table Geometry Module for PoolBot Laser Calibration System.

Defines exact pool table dimensions, coordinate boundaries, aspect ratio,
pocket locations, and coordinate transformation utilities.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass(frozen=True)
class TableDimensions:
    """Exact dimensions of the pool table in millimeters."""
    # Inner playing surface (felt area)
    INNER_LENGTH_MM: float = 2341.88  # 92.2 inches (X axis)
    INNER_BREADTH_MM: float = 1168.40  # 46.0 inches (Y axis)

    # Outer frame dimensions
    OUTER_LENGTH_MM: float = 2423.16  # 95.4 inches
    OUTER_BREADTH_MM: float = 1211.58  # 47.7 inches

    # Cushion widths (difference between outer and inner halved)
    CUSHION_X_MM: float = (2423.16 - 2341.88) / 2.0  # 40.64 mm
    CUSHION_Y_MM: float = (1211.58 - 1168.40) / 2.0  # 21.59 mm

    # Aspect ratio (Length / Breadth)
    ASPECT_RATIO: float = 2341.88 / 1168.40  # ~2.00435

    # Nominal pocket radius for visualization (approx 4.5 inches / 2 ~ 57mm)
    CORNER_POCKET_RADIUS_MM: float = 55.0
    SIDE_POCKET_RADIUS_MM: float = 50.0

    # Height and provisional offsets (from specifications)
    MOUNTING_PLATE_HEIGHT_MM: float = 914.4  # ~36 inches above felt
    PROVISIONAL_OFFSET_X_MM: float = 914.4   # ~3 feet provisional
    PROVISIONAL_OFFSET_Y_MM: float = 914.4   # ~3 feet provisional


TABLE = TableDimensions()


@dataclass
class Pocket:
    name: str
    x_mm: float
    y_mm: float
    radius_mm: float


def get_pocket_locations() -> List[Pocket]:
    """
    Returns the nominal positions of the 6 pockets based on the inner felt coordinate system:
    Origin (0,0) is Top-Left inner corner.
    """
    half_x = TABLE.INNER_LENGTH_MM / 2.0
    half_y = TABLE.INNER_BREADTH_MM
    return [
        Pocket("Top-Left", 0.0, 0.0, TABLE.CORNER_POCKET_RADIUS_MM),
        Pocket("Top-Center", half_x, 0.0, TABLE.SIDE_POCKET_RADIUS_MM),
        Pocket("Top-Right", TABLE.INNER_LENGTH_MM, 0.0, TABLE.CORNER_POCKET_RADIUS_MM),
        Pocket("Bottom-Left", 0.0, half_y, TABLE.CORNER_POCKET_RADIUS_MM),
        Pocket("Bottom-Center", half_x, half_y, TABLE.SIDE_POCKET_RADIUS_MM),
        Pocket("Bottom-Right", TABLE.INNER_LENGTH_MM, half_y, TABLE.CORNER_POCKET_RADIUS_MM),
    ]


def is_within_playing_surface(x_mm: float, y_mm: float) -> bool:
    """Check if a coordinate (x, y) lies inside the inner playing surface."""
    return (0.0 <= x_mm <= TABLE.INNER_LENGTH_MM) and (0.0 <= y_mm <= TABLE.INNER_BREADTH_MM)


def clamp_to_playing_surface(x_mm: float, y_mm: float) -> Tuple[float, float]:
    """Clamp a coordinate (x, y) to the playing surface boundaries."""
    cx = max(0.0, min(TABLE.INNER_LENGTH_MM, float(x_mm)))
    cy = max(0.0, min(TABLE.INNER_BREADTH_MM, float(y_mm)))
    return cx, cy


class TableCanvasTransformer:
    """
    Handles coordinate mapping between physical table coordinates (in mm)
    and screen canvas pixel coordinates, preserving the exact 2.004:1 aspect ratio
    with configurable padding and frame border.
    """

    def __init__(self, canvas_width: int, canvas_height: int, padding: int = 40):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.padding = padding
        self.recompute_layout()

    def recompute_layout(self):
        """Recompute scale and offsets based on current canvas dimensions."""
        available_w = max(10, self.canvas_width - 2 * self.padding)
        available_h = max(10, self.canvas_height - 2 * self.padding)

        # Scale based on outer table frame to fit within available canvas
        scale_x = available_w / TABLE.OUTER_LENGTH_MM
        scale_y = available_h / TABLE.OUTER_BREADTH_MM
        self.scale = min(scale_x, scale_y)

        # Rendered outer dimensions
        self.outer_render_w = TABLE.OUTER_LENGTH_MM * self.scale
        self.outer_render_h = TABLE.OUTER_BREADTH_MM * self.scale

        # Centering offsets on canvas
        self.outer_x0 = (self.canvas_width - self.outer_render_w) / 2.0
        self.outer_y0 = (self.canvas_height - self.outer_render_h) / 2.0

        # Inner felt offsets on canvas
        self.inner_x0 = self.outer_x0 + (TABLE.CUSHION_X_MM * self.scale)
        self.inner_y0 = self.outer_y0 + (TABLE.CUSHION_Y_MM * self.scale)
        self.inner_render_w = TABLE.INNER_LENGTH_MM * self.scale
        self.inner_render_h = TABLE.INNER_BREADTH_MM * self.scale

    def resize(self, new_width: int, new_height: int):
        """Update canvas dimensions and recompute scale."""
        self.canvas_width = new_width
        self.canvas_height = new_height
        self.recompute_layout()

    def table_mm_to_canvas_px(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Convert table coordinates (x in mm, y in mm) to canvas pixel coordinates."""
        px = self.inner_x0 + (x_mm * self.scale)
        py = self.inner_y0 + (y_mm * self.scale)
        return px, py

    def canvas_px_to_table_mm(self, px: float, py: float) -> Tuple[float, float]:
        """
        Convert canvas pixel coordinates to table coordinates (x in mm, y in mm).
        Coordinates are not clamped here so the caller can check bounds.
        """
        if self.scale <= 0:
            return 0.0, 0.0
        x_mm = (px - self.inner_x0) / self.scale
        y_mm = (py - self.inner_y0) / self.scale
        return x_mm, y_mm

    def get_inner_felt_rect(self) -> Tuple[float, float, float, float]:
        """Returns (x0, y0, x1, y1) in pixels for the inner felt playing surface."""
        return (
            self.inner_x0,
            self.inner_y0,
            self.inner_x0 + self.inner_render_w,
            self.inner_y0 + self.inner_render_h,
        )

    def get_outer_frame_rect(self) -> Tuple[float, float, float, float]:
        """Returns (x0, y0, x1, y1) in pixels for the outer wooden frame/cushions."""
        return (
            self.outer_x0,
            self.outer_y0,
            self.outer_x0 + self.outer_render_w,
            self.outer_y0 + self.outer_render_h,
        )
