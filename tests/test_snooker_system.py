"""
Unit Tests for Snooker AI Laser Positioning System
"""
import unittest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from laser_calibration.snooker_geometry import SNOOKER, is_within_snooker_table, clamp_to_snooker_table, get_snooker_pockets
from laser_calibration.snooker_interpolator import SnookerLaserInterpolator
from laser_calibration.camera_calibration import SnookerCameraHomography
from laser_calibration.laser_spot_detector import detect_laser_spot

class TestSnookerSystem(unittest.TestCase):
    def setUp(self):
        self.sample_nodes = [
            {"id": "node_tr", "label": "TR", "x_mm": 0.0,    "y_mm": 0.0,     "pan_deg": 10.0,  "tilt_deg": 150.0},
            {"id": "node_tc", "label": "TC", "x_mm": 584.2,  "y_mm": 0.0,     "pan_deg": 105.0, "tilt_deg": 126.0},
            {"id": "node_tl", "label": "TL", "x_mm": 1168.4, "y_mm": 0.0,     "pan_deg": 108.0, "tilt_deg": 107.0},
            {"id": "node_mr", "label": "MR", "x_mm": 0.0,    "y_mm": 1170.94, "pan_deg": 11.0,  "tilt_deg": 104.0},
            {"id": "node_c",  "label": "C",  "x_mm": 584.2,  "y_mm": 1170.94, "pan_deg": 35.0,  "tilt_deg": 102.0},
            {"id": "node_ml", "label": "ML", "x_mm": 1168.4, "y_mm": 1170.94, "pan_deg": 56.0,  "tilt_deg": 95.0},
            {"id": "node_br", "label": "BR", "x_mm": 0.0,    "y_mm": 2341.88, "pan_deg": 11.0,  "tilt_deg": 81.0},
            {"id": "node_bc", "label": "BC", "x_mm": 584.2,  "y_mm": 2341.88, "pan_deg": 25.0,  "tilt_deg": 81.0},
            {"id": "node_bl", "label": "BL", "x_mm": 1168.4, "y_mm": 2341.88, "pan_deg": 35.0,  "tilt_deg": 81.0},
        ]
        self.interpolator = SnookerLaserInterpolator(self.sample_nodes)

    def test_dimensions(self):
        self.assertAlmostEqual(SNOOKER.WIDTH_MM, 1168.4, places=1)
        self.assertAlmostEqual(SNOOKER.LENGTH_MM, 2341.88, places=1)
        self.assertAlmostEqual(SNOOKER.TILT_PIVOT_TO_FELT_MM, 1315.15, places=2)

    def test_exact_six_pockets(self):
        pockets = get_snooker_pockets()
        self.assertEqual(len(pockets), 6, "Snooker table must have exactly 6 pockets")
        pocket_names = [p.name for p in pockets]
        self.assertIn("Top-Right Pocket", pocket_names)
        self.assertIn("Top-Left Pocket", pocket_names)
        self.assertIn("Middle-Right Pocket", pocket_names)
        self.assertIn("Middle-Left Pocket", pocket_names)
        self.assertIn("Bottom-Right Pocket", pocket_names)
        self.assertIn("Bottom-Left Pocket", pocket_names)
        # Ensure no top-centre or bottom-centre pockets exist
        for name in pocket_names:
            self.assertNotIn("Top-Centre", name)
            self.assertNotIn("Bottom-Centre", name)

    def test_bounds(self):
        self.assertTrue(is_within_snooker_table(0.0, 0.0))
        self.assertTrue(is_within_snooker_table(1168.4, 2341.88))
        self.assertFalse(is_within_snooker_table(-5.0, 100.0))

    def test_exact_node_predictions(self):
        p, t, _ = self.interpolator.predict(0.0, 0.0)
        self.assertAlmostEqual(p, 10.0, places=1)
        self.assertAlmostEqual(t, 150.0, places=1)

    def test_homography(self):
        homo = SnookerCameraHomography()
        corners = [(100.0, 50.0), (800.0, 55.0), (820.0, 950.0), (80.0, 940.0)]
        self.assertTrue(homo.calibrate(corners))
        tx, ty = homo.pixel_to_table(100.0, 50.0)
        self.assertAlmostEqual(tx, 0.0, delta=1.0)
        self.assertAlmostEqual(ty, 0.0, delta=1.0)

if __name__ == "__main__":
    unittest.main()
