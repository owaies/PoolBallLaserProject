"""
Calibration Data Management Module for PoolBot Laser Calibration System.

Handles storing, loading, updating, and saving calibration points (pan/tilt angles
associated with physical table coordinates).
"""

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Optional, Any

from laser_calibration.table_geometry import TABLE


@dataclass
class CalibrationNode:
    """Represents a single calibration observation node."""
    id: str
    label: str
    x_mm: float
    y_mm: float
    pan_deg: float
    tilt_deg: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationNode":
        return cls(
            id=str(data.get("id", uuid.uuid4().hex[:8])),
            label=str(data.get("label", "Point")),
            x_mm=float(data["x_mm"]),
            y_mm=float(data["y_mm"]),
            pan_deg=float(data["pan_deg"]),
            tilt_deg=float(data["tilt_deg"]),
            notes=str(data.get("notes", "")),
        )


def get_default_nine_samples() -> List[CalibrationNode]:
    """
    Returns the initial 9 empirical observations provided in the specification.
    Note: These are empirical observations, preserved as editable calibration data.
    """
    half_x = round(TABLE.INNER_LENGTH_MM / 2.0, 2)  # 1170.94 mm
    max_x = round(TABLE.INNER_LENGTH_MM, 2)          # 2341.88 mm
    half_y = round(TABLE.INNER_BREADTH_MM / 2.0, 2) # 584.20 mm
    max_y = round(TABLE.INNER_BREADTH_MM, 2)         # 1168.40 mm

    samples = [
        # Top row
        CalibrationNode(
            id="node_tl",
            label="Top-left",
            x_mm=0.0,
            y_mm=0.0,
            pan_deg=129.0,
            tilt_deg=61.0,
            notes="Empirical initial sample at top-left corner"
        ),
        CalibrationNode(
            id="node_tc",
            label="Top-centre",
            x_mm=half_x,
            y_mm=0.0,
            pan_deg=109.0,
            tilt_deg=81.0,
            notes="Empirical initial sample at top-centre side"
        ),
        CalibrationNode(
            id="node_tr",
            label="Top-right",
            x_mm=max_x,
            y_mm=0.0,
            pan_deg=104.0,
            tilt_deg=94.0,
            notes="Empirical initial sample at top-right corner"
        ),
        # Middle row
        CalibrationNode(
            id="node_ml",
            label="Middle-left",
            x_mm=0.0,
            y_mm=half_y,
            pan_deg=154.0,
            tilt_deg=70.0,
            notes="Empirical initial sample at middle-left boundary"
        ),
        CalibrationNode(
            id="node_c",
            label="Centre",
            x_mm=half_x,
            y_mm=half_y,
            pan_deg=125.0,
            tilt_deg=84.0,
            notes="Empirical initial sample at table centre"
        ),
        CalibrationNode(
            id="node_mr",
            label="Middle-right",
            x_mm=max_x,
            y_mm=half_y,
            pan_deg=115.0,
            tilt_deg=94.0,
            notes="Empirical initial sample at middle-right boundary"
        ),
        # Bottom row
        CalibrationNode(
            id="node_bl",
            label="Bottom-left",
            x_mm=0.0,
            y_mm=max_y,
            pan_deg=167.0,
            tilt_deg=79.0,
            notes="Empirical initial sample at bottom-left corner"
        ),
        CalibrationNode(
            id="node_bc",
            label="Bottom-centre",
            x_mm=half_x,
            y_mm=max_y,
            pan_deg=140.0,
            tilt_deg=87.0,
            notes="Empirical initial sample at bottom-centre side"
        ),
        CalibrationNode(
            id="node_br",
            label="Bottom-right",
            x_mm=max_x,
            y_mm=max_y,
            pan_deg=125.0,
            tilt_deg=94.0,
            notes="Empirical initial sample at bottom-right corner"
        ),
    ]
    return samples


@dataclass
class CalibrationDataset:
    """Manages the full collection of calibration points and file persistence."""
    nodes: List[CalibrationNode] = field(default_factory=get_default_nine_samples)
    version: str = "1.0.0"
    description: str = "PoolBot Pan/Tilt Calibration Dataset"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_node(self, node: CalibrationNode) -> None:
        """Add a new calibration node."""
        self.nodes.append(node)
        self.updated_at = datetime.now().isoformat()

    def update_node(self, node_id: str, x_mm: float, y_mm: float, pan_deg: float, tilt_deg: float,
                    label: Optional[str] = None, notes: Optional[str] = None) -> bool:
        """Update an existing node by ID."""
        for n in self.nodes:
            if n.id == node_id:
                n.x_mm = float(x_mm)
                n.y_mm = float(y_mm)
                n.pan_deg = float(pan_deg)
                n.tilt_deg = float(tilt_deg)
                if label is not None:
                    n.label = label
                if notes is not None:
                    n.notes = notes
                self.updated_at = datetime.now().isoformat()
                return True
        return False

    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID. Requires at least 3 points to remain."""
        if len(self.nodes) <= 3:
            return False
        initial_len = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.id != node_id]
        if len(self.nodes) < initial_len:
            self.updated_at = datetime.now().isoformat()
            return True
        return False

    def reset_to_defaults(self) -> None:
        """Reset dataset to initial 9 empirical samples."""
        self.nodes = get_default_nine_samples()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "table_dimensions": {
                "inner_length_mm": TABLE.INNER_LENGTH_MM,
                "inner_breadth_mm": TABLE.INNER_BREADTH_MM,
                "aspect_ratio": TABLE.ASPECT_RATIO,
            },
            "nodes": [n.to_dict() for n in self.nodes]
        }

    def save_to_file(self, filepath: str) -> None:
        """Save dataset to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "CalibrationDataset":
        """Load dataset from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = [CalibrationNode.from_dict(item) for item in data.get("nodes", [])]
        dataset = cls(
            nodes=nodes,
            version=data.get("version", "1.0.0"),
            description=data.get("description", "PoolBot Pan/Tilt Calibration Dataset"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
        return dataset
