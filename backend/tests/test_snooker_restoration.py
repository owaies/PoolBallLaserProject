import unittest
import math
from backend.app.services.restoration_service import restoration_service

class TestSnookerRestoration(unittest.TestCase):

    def test_01_single_displaced_ball(self):
        before = [
            {"id": "cue_0", "class_id": 0, "class_name": "cue_ball", "x": 100, "y": 100, "norm_x": 0.1, "norm_y": 0.1, "confidence": 0.95}
        ]
        after = [
            {"id": "cue_0", "class_id": 0, "class_name": "cue_ball", "x": 150, "y": 150, "norm_x": 0.15, "norm_y": 0.15, "confidence": 0.95}
        ]
        pairs = restoration_service.spatial_match_balls(before, after)
        displaced = restoration_service.detect_displaced_balls(pairs, threshold_norm=0.003)
        queue = restoration_service.create_restoration_queue(displaced)

        self.assertEqual(len(displaced), 1)
        self.assertEqual(queue["total_displaced"], 1)
        self.assertEqual(queue["displaced_balls"][0]["ball_class_name"], "cue_ball")

    def test_02_two_displaced_balls(self):
        before = [
            {"id": "red_1", "class_id": 1, "class_name": "red_ball", "x": 100, "y": 100, "norm_x": 0.10, "norm_y": 0.10, "confidence": 0.90},
            {"id": "blue_5", "class_id": 5, "class_name": "blue_ball", "x": 300, "y": 300, "norm_x": 0.30, "norm_y": 0.30, "confidence": 0.92}
        ]
        after = [
            {"id": "red_1", "class_id": 1, "class_name": "red_ball", "x": 120, "y": 120, "norm_x": 0.12, "norm_y": 0.12, "confidence": 0.90},
            {"id": "blue_5", "class_id": 5, "class_name": "blue_ball", "x": 350, "y": 350, "norm_x": 0.35, "norm_y": 0.35, "confidence": 0.92}
        ]
        pairs = restoration_service.spatial_match_balls(before, after)
        displaced = restoration_service.detect_displaced_balls(pairs, threshold_norm=0.003)
        queue = restoration_service.create_restoration_queue(displaced)

        self.assertEqual(len(displaced), 2)
        # Blue moved further (0.05 norm vs 0.02 norm), so Blue must be first in queue
        self.assertEqual(queue["displaced_balls"][0]["ball_class_name"], "blue_ball")
        self.assertEqual(queue["displaced_balls"][1]["ball_class_name"], "red_ball")

    def test_03_three_displaced_balls(self):
        before = [
            {"id": "red_1", "class_id": 1, "class_name": "red_ball", "x": 100, "y": 100, "norm_x": 0.10, "norm_y": 0.10, "confidence": 0.90},
            {"id": "blue_5", "class_id": 5, "class_name": "blue_ball", "x": 300, "y": 300, "norm_x": 0.30, "norm_y": 0.30, "confidence": 0.92},
            {"id": "pink_6", "class_id": 6, "class_name": "pink_ball", "x": 500, "y": 500, "norm_x": 0.50, "norm_y": 0.50, "confidence": 0.95}
        ]
        after = [
            {"id": "red_1", "class_id": 1, "class_name": "red_ball", "x": 120, "y": 120, "norm_x": 0.12, "norm_y": 0.12, "confidence": 0.90},
            {"id": "blue_5", "class_id": 5, "class_name": "blue_ball", "x": 310, "y": 310, "norm_x": 0.31, "norm_y": 0.31, "confidence": 0.92},
            {"id": "pink_6", "class_id": 6, "class_name": "pink_ball", "x": 580, "y": 580, "norm_x": 0.58, "norm_y": 0.58, "confidence": 0.95}
        ]
        pairs = restoration_service.spatial_match_balls(before, after)
        displaced = restoration_service.detect_displaced_balls(pairs, threshold_norm=0.003)
        queue = restoration_service.create_restoration_queue(displaced)

        self.assertEqual(len(displaced), 3)
        # Pink moved largest (0.08 norm), Red second (0.02 norm), Blue third (0.01 norm)
        self.assertEqual(queue["displaced_balls"][0]["ball_class_name"], "pink_ball")
        self.assertEqual(queue["displaced_balls"][1]["ball_class_name"], "red_ball")
        self.assertEqual(queue["displaced_balls"][2]["ball_class_name"], "blue_ball")

    def test_04_equal_displacement_tie_breaking(self):
        # Blue, Red, Pink all displaced by exact same distance (0.02 norm) & same confidence
        before = [
            {"id": "blue_5", "class_id": 5, "class_name": "blue_ball", "x": 300, "y": 300, "norm_x": 0.30, "norm_y": 0.30, "confidence": 0.90},
            {"id": "red_1", "class_id": 1, "class_name": "red_ball", "x": 100, "y": 100, "norm_x": 0.10, "norm_y": 0.10, "confidence": 0.90},
            {"id": "pink_6", "class_id": 6, "class_name": "pink_ball", "x": 500, "y": 500, "norm_x": 0.50, "norm_y": 0.50, "confidence": 0.90}
        ]
        after = [
            {"id": "blue_5", "class_id": 5, "class_name": "blue_ball", "x": 320, "y": 300, "norm_x": 0.32, "norm_y": 0.30, "confidence": 0.90},
            {"id": "red_1", "class_id": 1, "class_name": "red_ball", "x": 120, "y": 100, "norm_x": 0.12, "norm_y": 0.10, "confidence": 0.90},
            {"id": "pink_6", "class_id": 6, "class_name": "pink_ball", "x": 520, "y": 500, "norm_x": 0.52, "norm_y": 0.50, "confidence": 0.90}
        ]
        pairs = restoration_service.spatial_match_balls(before, after)
        displaced = restoration_service.detect_displaced_balls(pairs, threshold_norm=0.003)
        queue = restoration_service.create_restoration_queue(displaced)

        self.assertEqual(len(displaced), 3)
        # Class hierarchy tie-break: red_ball (1) < blue_ball (5) < pink_ball (6)
        self.assertEqual(queue["displaced_balls"][0]["ball_class_name"], "red_ball")
        self.assertEqual(queue["displaced_balls"][1]["ball_class_name"], "blue_ball")
        self.assertEqual(queue["displaced_balls"][2]["ball_class_name"], "pink_ball")

    def test_05_cue_ball_displacement(self):
        before = [{"id": "cue_0", "class_id": 0, "class_name": "cue_ball", "x": 200, "y": 200, "norm_x": 0.20, "norm_y": 0.20, "confidence": 0.99}]
        after = [{"id": "cue_0", "class_id": 0, "class_name": "cue_ball", "x": 250, "y": 250, "norm_x": 0.25, "norm_y": 0.25, "confidence": 0.99}]
        pairs = restoration_service.spatial_match_balls(before, after)
        displaced = restoration_service.detect_displaced_balls(pairs, threshold_norm=0.003)

        self.assertEqual(len(displaced), 1)
        self.assertEqual(displaced[0]["ball_class_id"], 0)

    def test_06_ball_not_displaced(self):
        # Ball moved less than threshold_norm (0.0001 norm change)
        before = [{"id": "black_7", "class_id": 7, "class_name": "black_ball", "x": 800, "y": 400, "norm_x": 0.80, "norm_y": 0.40, "confidence": 0.95}]
        after = [{"id": "black_7", "class_id": 7, "class_name": "black_ball", "x": 800.1, "y": 400.1, "norm_x": 0.8001, "norm_y": 0.4001, "confidence": 0.95}]
        pairs = restoration_service.spatial_match_balls(before, after)
        displaced = restoration_service.detect_displaced_balls(pairs, threshold_norm=0.003)

        self.assertEqual(len(displaced), 0)

    def test_07_multiple_red_balls_spatial_matching(self):
        # 3 red balls in different positions
        before = [
            {"id": "red_a", "class_id": 1, "class_name": "red_ball", "x": 100, "y": 100, "norm_x": 0.10, "norm_y": 0.10, "confidence": 0.90},
            {"id": "red_b", "class_id": 1, "class_name": "red_ball", "x": 500, "y": 500, "norm_x": 0.50, "norm_y": 0.50, "confidence": 0.90},
        ]
        after = [
            {"id": "red_a_moved", "class_id": 1, "class_name": "red_ball", "x": 110, "y": 110, "norm_x": 0.11, "norm_y": 0.11, "confidence": 0.90},
            {"id": "red_b_moved", "class_id": 1, "class_name": "red_ball", "x": 505, "y": 505, "norm_x": 0.505, "norm_y": 0.505, "confidence": 0.90},
        ]
        pairs = restoration_service.spatial_match_balls(before, after)
        self.assertEqual(len(pairs), 2)

    def test_08_laser_angle_math(self):
        b = {"norm_x": 0.5, "norm_y": 0.5} # Original
        a = {"norm_x": 0.4, "norm_y": 0.5} # Current (to left of original)
        dx = b["norm_x"] - a["norm_x"] # +0.1
        dy = b["norm_y"] - a["norm_y"] # 0.0
        angle = math.degrees(math.atan2(dy, dx))
        self.assertAlmostEqual(angle, 0.0, places=2) # 0 degrees = right direction

if __name__ == "__main__":
    unittest.main()
