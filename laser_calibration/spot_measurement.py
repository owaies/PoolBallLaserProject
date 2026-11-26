"""
Spot Measurement and Error Analysis Module for PoolBot Laser Calibration.

Records test targets, commanded angles, measured spot coordinates, and calculates
detailed error statistics (ex, ey, Euclidean, Mean, Median, Max, RMS).
Supports CSV export and session management.
"""

import csv
import math
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SpotMeasurementRecord:
    """Represents a single test spot measurement and error evaluation."""
    id: str
    timestamp: str
    target_x_mm: float
    target_y_mm: float
    commanded_pan_deg: float
    commanded_tilt_deg: float
    measured_x_mm: float
    measured_y_mm: float
    error_x_mm: float
    error_y_mm: float
    euclidean_error_mm: float
    notes: str = ""
    image_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        target_x_mm: float,
        target_y_mm: float,
        commanded_pan_deg: float,
        commanded_tilt_deg: float,
        measured_x_mm: float,
        measured_y_mm: float,
        notes: str = "",
        image_file: str = "",
    ) -> "SpotMeasurementRecord":
        """Factory method that calculates ex, ey, and euclidean error in mm."""
        ex = measured_x_mm - target_x_mm
        ey = measured_y_mm - target_y_mm
        e = math.hypot(ex, ey)
        return cls(
            id=uuid.uuid4().hex[:8],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            target_x_mm=round(float(target_x_mm), 2),
            target_y_mm=round(float(target_y_mm), 2),
            commanded_pan_deg=round(float(commanded_pan_deg), 1),
            commanded_tilt_deg=round(float(commanded_tilt_deg), 1),
            measured_x_mm=round(float(measured_x_mm), 2),
            measured_y_mm=round(float(measured_y_mm), 2),
            error_x_mm=round(float(ex), 2),
            error_y_mm=round(float(ey), 2),
            euclidean_error_mm=round(float(e), 2),
            notes=notes,
            image_file=image_file,
        )


@dataclass
class ErrorStatistics:
    """Aggregate error metrics across multiple calibration measurements."""
    count: int = 0
    mean_error_mm: float = 0.0
    median_error_mm: float = 0.0
    max_error_mm: float = 0.0
    rms_error_mm: float = 0.0
    std_error_mm: float = 0.0
    mean_ex_mm: float = 0.0
    mean_ey_mm: float = 0.0


class MeasurementSession:
    """Tracks a sequence of verification measurements during a calibration session."""

    def __init__(self):
        self.records: List[SpotMeasurementRecord] = []

    def add_record(self, record: SpotMeasurementRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def count(self) -> int:
        return len(self.records)

    def calculate_statistics(self) -> ErrorStatistics:
        """Computes statistical metrics over all recorded measurements."""
        if not self.records:
            return ErrorStatistics()

        errors = np.array([r.euclidean_error_mm for r in self.records], dtype=float)
        exs = np.array([r.error_x_mm for r in self.records], dtype=float)
        eys = np.array([r.error_y_mm for r in self.records], dtype=float)

        count = len(errors)
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        max_err = float(np.max(errors))
        rms_err = float(np.sqrt(np.mean(errors ** 2)))
        std_err = float(np.std(errors))
        mean_ex = float(np.mean(exs))
        mean_ey = float(np.mean(eys))

        return ErrorStatistics(
            count=count,
            mean_error_mm=round(mean_err, 2),
            median_error_mm=round(median_err, 2),
            max_error_mm=round(max_err, 2),
            rms_error_mm=round(rms_err, 2),
            std_error_mm=round(std_err, 2),
            mean_ex_mm=round(mean_ex, 2),
            mean_ey_mm=round(mean_ey, 2),
        )

    def export_to_csv(self, filepath: str) -> None:
        """Export session measurements to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fieldnames = [
            "id",
            "timestamp",
            "target_x_mm",
            "target_y_mm",
            "commanded_pan_deg",
            "commanded_tilt_deg",
            "measured_x_mm",
            "measured_y_mm",
            "error_x_mm",
            "error_y_mm",
            "euclidean_error_mm",
            "notes",
            "image_file",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.to_dict())

    def import_from_csv(self, filepath: str) -> int:
        """Import measurements from CSV into session."""
        if not os.path.exists(filepath):
            return 0
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = SpotMeasurementRecord(
                    id=row.get("id", uuid.uuid4().hex[:8]),
                    timestamp=row.get("timestamp", ""),
                    target_x_mm=float(row.get("target_x_mm", 0.0)),
                    target_y_mm=float(row.get("target_y_mm", 0.0)),
                    commanded_pan_deg=float(row.get("commanded_pan_deg", 90.0)),
                    commanded_tilt_deg=float(row.get("commanded_tilt_deg", 90.0)),
                    measured_x_mm=float(row.get("measured_x_mm", 0.0)),
                    measured_y_mm=float(row.get("measured_y_mm", 0.0)),
                    error_x_mm=float(row.get("error_x_mm", 0.0)),
                    error_y_mm=float(row.get("error_y_mm", 0.0)),
                    euclidean_error_mm=float(row.get("euclidean_error_mm", 0.0)),
                    notes=row.get("notes", ""),
                    image_file=row.get("image_file", ""),
                )
                self.records.append(rec)
                loaded += 1
        return loaded
