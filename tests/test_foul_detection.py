"""
Unit Tests for Snooker Foul Detection Engine (Configured for Orange Object Balls)
"""
import unittest
from laser_calibration.foul_detector import FoulDetector, normalize_label, SNOOKER_POINTS

class TestFoulDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FoulDetector()
        self.initial_balls = [
            {"id": 1, "label": "white-ball", "table_x_mm": 584.2, "table_y_mm": 737.0},
            {"id": 2, "label": "orange ball", "table_x_mm": 584.2, "table_y_mm": 1800.0},
            {"id": 3, "label": "blue ball", "table_x_mm": 584.2, "table_y_mm": 1170.94},
            {"id": 4, "label": "pink ball", "table_x_mm": 584.2, "table_y_mm": 1756.4},
        ]
        self.detector.set_reference_state(self.initial_balls)

    def test_cue_ball_in_off(self):
        # White ball disappeared, orange moved
        current_balls = [
            {"id": 2, "label": "orange ball", "table_x_mm": 600.0, "table_y_mm": 1850.0},
            {"id": 3, "label": "blue ball", "table_x_mm": 584.2, "table_y_mm": 1170.94},
            {"id": 4, "label": "pink ball", "table_x_mm": 584.2, "table_y_mm": 1756.4},
        ]
        report = self.detector.analyze_shot(current_balls, nominated_target="orange")
        self.assertTrue(report.is_foul)
        self.assertEqual(report.foul_type, "CUE_BALL_IN_OFF")
        self.assertEqual(report.penalty_points, 4)
        self.assertTrue(report.cue_ball_in_off)

    def test_illegal_color_potted(self):
        # Player was on Orange, but Blue ball was potted
        current_balls = [
            {"id": 1, "label": "white-ball", "table_x_mm": 580.0, "table_y_mm": 900.0},
            {"id": 2, "label": "orange ball", "table_x_mm": 584.2, "table_y_mm": 1800.0},
            {"id": 4, "label": "pink ball", "table_x_mm": 584.2, "table_y_mm": 1756.4},
        ]
        report = self.detector.analyze_shot(current_balls, nominated_target="orange")
        self.assertTrue(report.is_foul)
        self.assertEqual(report.foul_type, "ILLEGAL_COLOR_POTTED")
        self.assertEqual(report.penalty_points, 5) # Blue ball value is 5
        self.assertIn("blue ball", report.potted_balls)

    def test_orange_potted_on_color(self):
        # Player was on Pink, but Orange was potted
        current_balls = [
            {"id": 1, "label": "white-ball", "table_x_mm": 580.0, "table_y_mm": 900.0},
            {"id": 3, "label": "blue ball", "table_x_mm": 584.2, "table_y_mm": 1170.94},
            {"id": 4, "label": "pink ball", "table_x_mm": 584.2, "table_y_mm": 1756.4},
        ]
        report = self.detector.analyze_shot(current_balls, nominated_target="pink")
        self.assertTrue(report.is_foul)
        self.assertEqual(report.foul_type, "ORANGE_POTTED_ON_COLOR")
        self.assertEqual(report.penalty_points, 4)

    def test_legal_orange_pot(self):
        # Player was on Orange, Orange is potted, White stays on table
        current_balls = [
            {"id": 1, "label": "white-ball", "table_x_mm": 580.0, "table_y_mm": 900.0},
            {"id": 3, "label": "blue ball", "table_x_mm": 584.2, "table_y_mm": 1170.94},
            {"id": 4, "label": "pink ball", "table_x_mm": 584.2, "table_y_mm": 1756.4},
        ]
        report = self.detector.analyze_shot(current_balls, nominated_target="orange")
        self.assertFalse(report.is_foul)
        self.assertIn("LEGAL SHOT", report.description)
        self.assertIn("Orange ball(s) potted", report.description)

    def test_displaced_balls_tracking(self):
        # Blue ball moved 50mm
        current_balls = [
            {"id": 1, "label": "white-ball", "table_x_mm": 584.2, "table_y_mm": 737.0},
            {"id": 2, "label": "orange ball", "table_x_mm": 584.2, "table_y_mm": 1800.0},
            {"id": 3, "label": "blue ball", "table_x_mm": 584.2 + 30.0, "table_y_mm": 1170.94 + 40.0},
            {"id": 4, "label": "pink ball", "table_x_mm": 584.2, "table_y_mm": 1756.4},
        ]
        report = self.detector.analyze_shot(current_balls, nominated_target="orange")
        self.assertFalse(report.is_foul)
        self.assertEqual(len(report.displaced_balls), 1)
        self.assertEqual(report.displaced_balls[0].label, "blue ball")
        self.assertAlmostEqual(report.displaced_balls[0].displacement_mm, 50.0, delta=1.0)

if __name__ == "__main__":
    unittest.main()
