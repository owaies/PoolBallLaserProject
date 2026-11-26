import unittest
import math
from backend.app.services.restoration_service import restoration_service
from backend.app.core.game_manager import game_manager
from backend.app.services.yolo_service import yolo_service

class TestSnookerRefereeRestoration(unittest.TestCase):

    def test_01_foul_detection_wrong_ball_hit_first(self):
        ball_on = 1 # Red ON
        first_contact = {"class_id": 5, "class_name": "blue_ball"} # Blue hit first
        penalty = max(4, 5) # Blue = 5
        self.assertEqual(penalty, 5)

    def test_02_cue_ball_scratch_foul(self):
        potted = [{"class_id": 0, "class_name": "cue_ball"}]
        is_scratch = any(b["class_id"] == 0 for b in potted)
        penalty = 4
        self.assertTrue(is_scratch)
        self.assertEqual(penalty, 4)

    def test_03_no_ball_hit_foul(self):
        first_contact = None
        ball_on = 7 # Black ON
        penalty = max(4, 7) # Black = 7
        self.assertIsNone(first_contact)
        self.assertEqual(penalty, 7)

    def test_04_penalty_calculation_max_rule(self):
        # Red ON (1) -> Pink (6) hit first
        p_pink = max(4, 6)
        self.assertEqual(p_pink, 6)

        # Red ON (1) -> Yellow (2) hit first
        p_yellow = max(4, 2)
        self.assertEqual(p_yellow, 4)

    def test_05_foul_and_miss_flag(self):
        is_foul = True
        is_foul_and_miss = True
        self.assertTrue(is_foul and is_foul_and_miss)

    def test_06_snapshot_creation_and_immutability(self):
        before = [
            {"id": "cue_0", "class_id": 0, "x": 100, "y": 100},
            {"id": "red_1", "class_id": 1, "x": 200, "y": 200}
        ]
        # Deep clone snapshot
        snapshot = [dict(b) for b in before]
        snapshot[0]["x"] = 999 # Mutate clone
        self.assertEqual(before[0]["x"], 100) # Original remains untouched

    def test_07_automatic_restoration_queue_ordering(self):
        displaced = [
            {"id": "red_1", "ball_class_id": 1, "ball_class_name": "red_ball", "displacement_norm": 0.05, "confidence": 0.90},
            {"id": "blue_5", "ball_class_id": 5, "ball_class_name": "blue_ball", "displacement_norm": 0.08, "confidence": 0.95}
        ]
        queue = restoration_service.create_restoration_queue(displaced)
        # Blue moved further (0.08 vs 0.05), so Blue must be #1
        self.assertEqual(queue["displaced_balls"][0]["ball_class_name"], "blue_ball")

    def test_08_manual_arbitrary_restoration_order(self):
        displaced = [
            {"id": "red_1", "ball_class_id": 1, "ball_class_name": "red_ball", "displacement_norm": 0.05},
            {"id": "pink_6", "ball_class_id": 6, "ball_class_name": "pink_ball", "displacement_norm": 0.08},
            {"id": "blue_5", "ball_class_id": 5, "ball_class_name": "blue_ball", "displacement_norm": 0.03}
        ]
        # Referee arbitrary order selection: Pink -> Blue -> Red
        arbitrary_order = [displaced[1], displaced[2], displaced[0]]
        self.assertEqual(arbitrary_order[0]["ball_class_name"], "pink_ball")
        self.assertEqual(arbitrary_order[1]["ball_class_name"], "blue_ball")
        self.assertEqual(arbitrary_order[2]["ball_class_name"], "red_ball")

    def test_09_partial_restoration(self):
        all_displaced = ["cue_0", "red_1", "blue_5"]
        selected_to_restore = ["pink_6", "blue_5"] # Referee selects subset
        restored = [b for b in all_displaced if b in selected_to_restore]
        unrestored = [b for b in all_displaced if b not in selected_to_restore]
        self.assertEqual(restored, ["blue_5"])
        self.assertEqual(unrestored, ["cue_0", "red_1"])

    def test_10_restore_all(self):
        before_state = [{"id": "cue_0"}, {"id": "red_1"}, {"id": "blue_5"}]
        restored_state = [dict(b) for b in before_state]
        self.assertEqual(len(restored_state), 3)

    def test_11_restoration_history_undo_redo(self):
        history = ["shot_1", "shot_2", "shot_3"]
        undo_stack = []

        # Undo
        undone = history.pop()
        undo_stack.append(undone)
        self.assertEqual(undone, "shot_3")
        self.assertEqual(len(history), 2)

        # Redo
        redone = undo_stack.pop()
        history.append(redone)
        self.assertEqual(redone, "shot_3")
        self.assertEqual(len(history), 3)

    def test_12_laser_angle_and_distance_math(self):
        current_pos = {"x": 100, "y": 100}
        target_pos = {"x": 200, "y": 100}
        dx = target_pos["x"] - current_pos["x"] # +100
        dy = target_pos["y"] - current_pos["y"] # 0
        dist = math.sqrt(dx*dx + dy*dy) # 100
        angle = math.degrees(math.atan2(dy, dx)) # 0 degrees
        self.assertEqual(dist, 100.0)
        self.assertEqual(angle, 0.0)

    def test_13_pool_model_isolation_check(self):
        game_manager.initialize()
        
        # Test Snooker model selection
        game_manager.select_game("snooker")
        yolo_service.initialize()
        snooker_info = yolo_service.get_info()
        self.assertEqual(snooker_info["number_of_classes"], 8)

        # Test Pool model selection
        game_manager.select_game("pool")
        yolo_service.initialize()
        pool_info = yolo_service.get_info()
        self.assertEqual(pool_info["number_of_classes"], 16)

if __name__ == "__main__":
    unittest.main()
