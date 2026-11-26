"""
Unit Test Suite for PoolBot Laser Calibration System.

Tests:
1. Table geometry, boundaries, and canvas scaling.
2. Calibration data structures and JSON persistence.
3. Angle interpolators (Bilinear Grid exact nodes and midpoint continuity, Scattered, Geometric).
4. ESP32 client behavior (simulation mode, soft limits, confirmation safety, mock HTTP routes).
5. Camera homography perspective transformations and projection accuracy.
6. Spot measurement, Euclidean error calculations, and statistical metrics.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser_calibration.table_geometry import (
    TABLE,
    is_within_playing_surface,
    clamp_to_playing_surface,
    TableCanvasTransformer,
    get_pocket_locations,
)
from laser_calibration.calibration_data import (
    CalibrationNode,
    CalibrationDataset,
    get_default_nine_samples,
)
from laser_calibration.interpolator import (
    BilinearGridInterpolator,
    ScatteredInterpolator,
    GeometricModelInterpolator,
)
from laser_calibration.esp32_client import ESP32Client, ServoSoftLimits, ESP32Status
from laser_calibration.camera_homography import (
    CameraHomography,
    ImageCornerSelection,
    detect_red_laser_spot,
)
from laser_calibration.spot_measurement import (
    SpotMeasurementRecord,
    MeasurementSession,
    ErrorStatistics,
)


class TestTableGeometry(unittest.TestCase):
    def test_dimensions(self):
        self.assertAlmostEqual(TABLE.INNER_LENGTH_MM, 2341.88, places=2)
        self.assertAlmostEqual(TABLE.INNER_BREADTH_MM, 1168.40, places=2)
        self.assertAlmostEqual(TABLE.OUTER_LENGTH_MM, 2423.16, places=2)
        self.assertAlmostEqual(TABLE.OUTER_BREADTH_MM, 1211.58, places=2)
        self.assertAlmostEqual(TABLE.ASPECT_RATIO, 2341.88 / 1168.40, places=4)

    def test_bounds(self):
        self.assertTrue(is_within_playing_surface(0.0, 0.0))
        self.assertTrue(is_within_playing_surface(2341.88, 1168.40))
        self.assertTrue(is_within_playing_surface(1170.94, 584.20))
        self.assertFalse(is_within_playing_surface(-1.0, 500.0))
        self.assertFalse(is_within_playing_surface(500.0, 1200.0))

        cx, cy = clamp_to_playing_surface(-50.0, 1500.0)
        self.assertEqual(cx, 0.0)
        self.assertEqual(cy, TABLE.INNER_BREADTH_MM)

    def test_pockets(self):
        pockets = get_pocket_locations()
        self.assertEqual(len(pockets), 6)
        names = [p.name for p in pockets]
        self.assertIn("Top-Left", names)
        self.assertIn("Bottom-Right", names)
        self.assertIn("Top-Center", names)

    def test_canvas_transformer_roundtrip(self):
        transformer = TableCanvasTransformer(canvas_width=800, canvas_height=500)
        test_points = [
            (0.0, 0.0),
            (2341.88, 0.0),
            (1170.94, 584.20),
            (2341.88, 1168.40),
        ]
        for x_mm, y_mm in test_points:
            px, py = transformer.table_mm_to_canvas_px(x_mm, y_mm)
            rx, ry = transformer.canvas_px_to_table_mm(px, py)
            self.assertAlmostEqual(x_mm, rx, places=1)
            self.assertAlmostEqual(y_mm, ry, places=1)


class TestCalibrationData(unittest.TestCase):
    def test_nine_empirical_samples(self):
        samples = get_default_nine_samples()
        self.assertEqual(len(samples), 9)

        labels = {s.label: (s.pan_deg, s.tilt_deg) for s in samples}
        expected = {
            "Top-left": (129.0, 61.0),
            "Top-centre": (109.0, 81.0),
            "Top-right": (104.0, 94.0),
            "Middle-left": (154.0, 70.0),
            "Centre": (125.0, 84.0),
            "Middle-right": (115.0, 94.0),
            "Bottom-left": (167.0, 79.0),
            "Bottom-centre": (140.0, 87.0),
            "Bottom-right": (125.0, 94.0),
        }
        for label, (pan, tilt) in expected.items():
            self.assertIn(label, labels)
            self.assertEqual(labels[label], (pan, tilt))

    def test_crud_and_json_persistence(self):
        dataset = CalibrationDataset()
        initial_count = len(dataset.nodes)

        # Add node
        new_node = CalibrationNode(
            id="test_node_1",
            label="Test Node",
            x_mm=500.0,
            y_mm=300.0,
            pan_deg=130.0,
            tilt_deg=75.0,
        )
        dataset.add_node(new_node)
        self.assertEqual(len(dataset.nodes), initial_count + 1)

        # Update node
        dataset.update_node("test_node_1", x_mm=550.0, y_mm=350.0, pan_deg=132.0, tilt_deg=76.0)
        updated = next(n for n in dataset.nodes if n.id == "test_node_1")
        self.assertEqual(updated.x_mm, 550.0)
        self.assertEqual(updated.pan_deg, 132.0)

        # JSON save and load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            dataset.save_to_file(temp_path)
            loaded = CalibrationDataset.load_from_file(temp_path)
            self.assertEqual(len(loaded.nodes), len(dataset.nodes))
            loaded_node = next(n for n in loaded.nodes if n.id == "test_node_1")
            self.assertEqual(loaded_node.pan_deg, 132.0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Delete node
        deleted = dataset.delete_node("test_node_1")
        self.assertTrue(deleted)
        self.assertEqual(len(dataset.nodes), initial_count)


class TestAngleInterpolation(unittest.TestCase):
    def setUp(self):
        self.samples = get_default_nine_samples()
        self.bilinear = BilinearGridInterpolator(self.samples)

    def test_bilinear_exact_at_nodes(self):
        # The bilinear interpolator MUST exactly match the calibration nodes at grid vertices
        for node in self.samples:
            pan, tilt, _ = self.bilinear.predict(node.x_mm, node.y_mm)
            self.assertAlmostEqual(pan, node.pan_deg, delta=0.01,
                                   msg=f"Pan mismatch at {node.label} ({node.x_mm}, {node.y_mm})")
            self.assertAlmostEqual(tilt, node.tilt_deg, delta=0.01,
                                   msg=f"Tilt mismatch at {node.label} ({node.x_mm}, {node.y_mm})")

    def test_bilinear_midpoint_quadrant(self):
        # Midpoint of Top-Left quadrant: X = 1170.94 / 2 = 585.47, Y = 584.20 / 2 = 292.10
        # Corners:
        # TL (0, 0): Pan 129, Tilt 61
        # TC (1170.94, 0): Pan 109, Tilt 81
        # ML (0, 584.2): Pan 154, Tilt 70
        # C (1170.94, 584.2): Pan 125, Tilt 84
        # Expected midpoint pan: 0.25*(129 + 109 + 154 + 125) = 129.25
        # Expected midpoint tilt: 0.25*(61 + 81 + 70 + 84) = 74.00
        mid_x = 1170.94 / 2.0
        mid_y = 584.20 / 2.0
        pan, tilt, info = self.bilinear.predict(mid_x, mid_y)
        self.assertAlmostEqual(pan, 129.25, delta=0.05)
        self.assertAlmostEqual(tilt, 74.00, delta=0.05)
        self.assertEqual(info["method"], "Bilinear Grid Interpolation")

    def test_scattered_interpolator(self):
        scattered = ScatteredInterpolator(self.samples)
        # Should be exact at nodes
        for node in self.samples:
            pan, tilt, _ = scattered.predict(node.x_mm, node.y_mm)
            self.assertAlmostEqual(pan, node.pan_deg, delta=0.1)
            self.assertAlmostEqual(tilt, node.tilt_deg, delta=0.1)

    def test_geometric_model(self):
        geom = GeometricModelInterpolator()
        geom.fit(self.samples)
        pan, tilt, info = geom.predict(1170.94, 584.20)
        self.assertIsInstance(pan, float)
        self.assertIsInstance(tilt, float)


class TestESP32Client(unittest.TestCase):
    def setUp(self):
        self.soft_limits = ServoSoftLimits(pan_min=15, pan_max=175, tilt_min=25, tilt_max=155)
        self.client = ESP32Client(
            base_url="http://192.168.4.1",
            simulation_mode=True,
            soft_limits=self.soft_limits,
        )

    def test_simulation_move_and_status(self):
        success, msg = self.client.move_servos(120.0, 80.0, confirmed=True)
        self.assertTrue(success)
        status = self.client.get_status()
        self.assertEqual(status.pan, 120)
        self.assertEqual(status.tilt, 80)
        self.assertFalse(status.laser)

    def test_soft_limits_clamping(self):
        # Commanded 5° pan (below soft min 15)
        self.client.move_servos(5.0, 90.0, confirmed=True)
        status = self.client.get_status()
        self.assertEqual(status.pan, 15)

        # Commanded 180° pan (above soft max 175)
        self.client.move_servos(180.0, 90.0, confirmed=True)
        status = self.client.get_status()
        self.assertEqual(status.pan, 175)

    def test_require_confirmation(self):
        success, msg = self.client.move_servos(100.0, 90.0, require_confirmation=True, confirmed=False)
        self.assertFalse(success)
        self.assertIn("confirmation required", msg.lower())

    def test_emergency_stop_behavior(self):
        self.client.move_servos(110.0, 85.0, confirmed=True)
        stop_success, stop_msg = self.client.stop_servos()
        self.assertTrue(stop_success)

        status = self.client.get_status()
        self.assertTrue(status.stopped)

        # Movement blocked while stopped
        success, msg = self.client.move_servos(120.0, 80.0, confirmed=True)
        self.assertFalse(success)
        self.assertIn("stopped", msg.lower())

        # Resuming with center
        self.client.center_servos()
        status_after_center = self.client.get_status()
        self.assertFalse(status_after_center.stopped)
        self.assertEqual(status_after_center.pan, 90)
        self.assertEqual(status_after_center.tilt, 90)

    def test_laser_safety_acknowledgment(self):
        # Laser cannot be enabled without explicit safety acknowledgment
        success, msg = self.client.set_laser(True, safety_acknowledgment=False)
        self.assertFalse(success)
        self.assertIn("safety acknowledgment", msg.lower())

        # Enabled with acknowledgment
        success, msg = self.client.set_laser(True, safety_acknowledgment=True)
        self.assertTrue(success)
        self.assertTrue(self.client.get_status().laser)

        # Laser turned off
        self.client.set_laser(False)
        self.assertFalse(self.client.get_status().laser)

    @patch("requests.get")
    def test_real_http_calls(self, mock_get):
        self.client.set_simulation_mode(False)

        # Mock /status response
        mock_status_resp = MagicMock()
        mock_status_resp.status_code = 200
        mock_status_resp.json.return_value = {"pan": 90, "tilt": 90, "laser": False, "stopped": False}

        # Mock /move response
        mock_move_resp = MagicMock()
        mock_move_resp.status_code = 200
        mock_move_resp.text = "Target accepted"

        mock_get.side_effect = [mock_status_resp, mock_move_resp, mock_move_resp]

        success, msg = self.client.move_servos(130.0, 75.0, confirmed=True)
        self.assertTrue(success)

        # Verify calls made
        self.assertEqual(mock_get.call_count, 3)
        # Check call arguments
        calls = mock_get.call_args_list
        # Call 1: /status
        self.assertIn("/status", calls[0][0][0])
        # Call 2: /move?axis=pan&angle=130
        self.assertIn("/move", calls[1][0][0])
        self.assertEqual(calls[1][1]["params"], {"axis": "pan", "angle": 130})
        # Call 3: /move?axis=tilt&angle=75
        self.assertIn("/move", calls[2][0][0])
        self.assertEqual(calls[2][1]["params"], {"axis": "tilt", "angle": 75})


class TestCameraHomography(unittest.TestCase):
    def setUp(self):
        # Synthetic camera image corners: table corners mapped to an image of 1920x1080
        # with perspective tilt
        self.corners = ImageCornerSelection(
            top_left=(200.0, 150.0),
            top_right=(1720.0, 180.0),
            bottom_right=(1820.0, 950.0),
            bottom_left=(100.0, 920.0),
        )
        self.homography = CameraHomography(self.corners)

    def test_homography_exact_corner_recovery(self):
        self.assertTrue(self.homography.is_calibrated())

        # Top-Left image pixel should map to (0, 0)
        tl_x, tl_y = self.homography.image_to_table(200.0, 150.0)
        self.assertAlmostEqual(tl_x, 0.0, delta=0.5)
        self.assertAlmostEqual(tl_y, 0.0, delta=0.5)

        # Bottom-Right image pixel should map to (2341.88, 1168.40)
        br_x, br_y = self.homography.image_to_table(1820.0, 950.0)
        self.assertAlmostEqual(br_x, TABLE.INNER_LENGTH_MM, delta=0.5)
        self.assertAlmostEqual(br_y, TABLE.INNER_BREADTH_MM, delta=0.5)

    def test_homography_roundtrip(self):
        # Convert table coordinate to image and back to table
        test_x, test_y = 1170.94, 584.20
        u, v = self.homography.table_to_image(test_x, test_y)
        rx, ry = self.homography.image_to_table(u, v)
        self.assertAlmostEqual(rx, test_x, delta=0.1)
        self.assertAlmostEqual(ry, test_y, delta=0.1)

    def test_red_spot_detection(self):
        # Create a synthetic image with a bright red spot on a dark green background
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:, :] = (20, 80, 20)  # Green felt (BGR)

        # Put a bright red spot at (350, 220)
        cv2.circle(img, (350, 220), 5, (30, 40, 255), -1)  # BGR

        detected = detect_red_laser_spot(img)
        self.assertIsNotNone(detected)
        cx, cy, conf = detected
        self.assertAlmostEqual(cx, 350.0, delta=1.5)
        self.assertAlmostEqual(cy, 220.0, delta=1.5)
        self.assertGreater(conf, 0.5)


class TestSpotMeasurementAndErrors(unittest.TestCase):
    def test_record_error_calculation(self):
        # Target: (1000.0, 500.0)
        # Measured: (1003.0, 504.0)
        # ex = +3.0 mm, ey = +4.0 mm, e = sqrt(3^2 + 4^2) = 5.0 mm
        rec = SpotMeasurementRecord.create(
            target_x_mm=1000.0,
            target_y_mm=500.0,
            commanded_pan_deg=120.0,
            commanded_tilt_deg=80.0,
            measured_x_mm=1003.0,
            measured_y_mm=504.0,
        )
        self.assertAlmostEqual(rec.error_x_mm, 3.0, places=2)
        self.assertAlmostEqual(rec.error_y_mm, 4.0, places=2)
        self.assertAlmostEqual(rec.euclidean_error_mm, 5.0, places=2)

    def test_session_statistics_and_csv(self):
        session = MeasurementSession()

        # Add 3 records with errors: 5.0, 10.0, 15.0
        # Rec 1: dx=3, dy=4 -> e=5
        session.add_record(SpotMeasurementRecord.create(100, 100, 90, 90, 103, 104))
        # Rec 2: dx=6, dy=8 -> e=10
        session.add_record(SpotMeasurementRecord.create(200, 200, 90, 90, 206, 208))
        # Rec 3: dx=9, dy=12 -> e=15
        session.add_record(SpotMeasurementRecord.create(300, 300, 90, 90, 309, 312))

        stats = session.calculate_statistics()
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.mean_error_mm, 10.0, places=1)
        self.assertAlmostEqual(stats.median_error_mm, 10.0, places=1)
        self.assertAlmostEqual(stats.max_error_mm, 15.0, places=1)
        # RMS = sqrt((25 + 100 + 225)/3) = sqrt(350/3) = sqrt(116.6667) = 10.80 mm
        self.assertAlmostEqual(stats.rms_error_mm, 10.80, places=1)

        # CSV export & re-import test
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            csv_path = tf.name

        try:
            session.export_to_csv(csv_path)
            new_session = MeasurementSession()
            loaded_count = new_session.import_from_csv(csv_path)
            self.assertEqual(loaded_count, 3)
            new_stats = new_session.calculate_statistics()
            self.assertAlmostEqual(new_stats.mean_error_mm, 10.0, places=1)
            self.assertAlmostEqual(new_stats.rms_error_mm, 10.80, places=1)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
