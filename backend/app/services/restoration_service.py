import math
import numpy as np
from typing import List, Dict, Any, Optional

SNOOKER_CLASS_NAMES = {
    0: "cue_ball",
    1: "red_ball",
    2: "yellow_ball",
    3: "green_ball",
    4: "brown_ball",
    5: "blue_ball",
    6: "pink_ball",
    7: "black_ball"
}

SNOOKER_BALL_COLORS = {
    0: "#FFFFFF",
    1: "#DC2626",
    2: "#FACC15",
    3: "#16A34A",
    4: "#92400E",
    5: "#2563EB",
    6: "#EC4899",
    7: "#18181B"
}

SNOOKER_CLASS_HIERARCHY = {
    0: 0, # cue_ball
    1: 1, # red_ball
    2: 2, # yellow_ball
    3: 3, # green_ball
    4: 4, # brown_ball
    5: 5, # blue_ball
    6: 6, # pink_ball
    7: 7  # black_ball
}

class RestorationService:
    @staticmethod
    def spatial_match_balls(before_balls: List[Dict[str, Any]], after_balls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched_pairs = []

        before_by_class = {}
        after_by_class = {}

        for b in before_balls:
            cid = b.get("class_id", 0)
            if cid not in before_by_class:
                before_by_class[cid] = []
            before_by_class[cid].append(b)

        for a in after_balls:
            cid = a.get("class_id", 0)
            if cid not in after_by_class:
                after_by_class[cid] = []
            after_by_class[cid].append(a)

        all_cids = set(list(before_by_class.keys()) + list(after_by_class.keys()))

        for cid in all_cids:
            b_list = before_by_class.get(cid, [])
            a_list = after_by_class.get(cid, [])

            if not b_list or not a_list:
                continue

            if cid != 1:
                # Unique colored balls
                min_len = min(len(b_list), len(a_list))
                for i in range(min_len):
                    b = b_list[i]
                    a = a_list[i]
                    dx_norm = a.get("norm_x", 0) - b.get("norm_x", 0)
                    dy_norm = a.get("norm_y", 0) - b.get("norm_y", 0)
                    dist_norm = math.sqrt(dx_norm * dx_norm + dy_norm * dy_norm)

                    dx_px = a.get("x", 0) - b.get("x", 0)
                    dy_px = a.get("y", 0) - b.get("y", 0)
                    dist_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)

                    matched_pairs.append({
                        "before_ball": b,
                        "after_ball": a,
                        "displacement_norm": dist_norm,
                        "displacement_px": dist_px,
                        "displacement_mm": dist_norm * 3569
                    })
            else:
                # Red balls (multiple instances) - Hungarian / Greedy Distance Matrix
                rem_b = list(b_list)
                rem_a = list(a_list)

                while rem_b and rem_a:
                    best_dist = float('inf')
                    best_b_idx = -1
                    best_a_idx = -1

                    for i, b in enumerate(rem_b):
                        for j, a in enumerate(rem_a):
                            dx = a.get("norm_x", 0) - b.get("norm_x", 0)
                            dy = a.get("norm_y", 0) - b.get("norm_y", 0)
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < best_dist:
                                best_dist = dist
                                best_b_idx = i
                                best_a_idx = j

                    if best_b_idx != -1 and best_a_idx != -1:
                        b = rem_b.pop(best_b_idx)
                        a = rem_a.pop(best_a_idx)
                        dx_px = a.get("x", 0) - b.get("x", 0)
                        dy_px = a.get("y", 0) - b.get("y", 0)
                        dist_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)

                        matched_pairs.append({
                            "before_ball": b,
                            "after_ball": a,
                            "displacement_norm": best_dist,
                            "displacement_px": dist_px,
                            "displacement_mm": best_dist * 3569
                        })
                    else:
                        break

        return matched_pairs

    @staticmethod
    def detect_displaced_balls(matched_pairs: List[Dict[str, Any]], threshold_norm: float = 0.003) -> List[Dict[str, Any]]:
        displaced_list = []

        for pair in matched_pairs:
            dist_norm = pair["displacement_norm"]
            dist_mm = pair["displacement_mm"]

            if dist_norm > threshold_norm:
                b = pair["before_ball"]
                a = pair["after_ball"]

                dx = b.get("norm_x", 0) - a.get("norm_x", 0)
                dy = b.get("norm_y", 0) - a.get("norm_y", 0)
                angle_deg = math.degrees(math.atan2(dy, dx))

                displaced_list.append({
                    "id": f"displaced_{b.get('id', 'ball')}",
                    "ball_class_id": b.get("class_id", 0),
                    "ball_class_name": b.get("class_name", SNOOKER_CLASS_NAMES.get(b.get("class_id", 0), "ball")),
                    "color": SNOOKER_BALL_COLORS.get(b.get("class_id", 0), "#FFFFFF"),
                    "original_pos": {
                        "x": b.get("x", 0),
                        "y": b.get("y", 0),
                        "norm_x": b.get("norm_x", 0),
                        "norm_y": b.get("norm_y", 0),
                        "table_x_mm": b.get("norm_x", 0) * 3569,
                        "table_y_mm": b.get("norm_y", 0) * 1778
                    },
                    "current_pos": {
                        "x": a.get("x", 0),
                        "y": a.get("y", 0),
                        "norm_x": a.get("norm_x", 0),
                        "norm_y": a.get("norm_y", 0),
                        "table_x_mm": a.get("norm_x", 0) * 3569,
                        "table_y_mm": a.get("norm_y", 0) * 1778
                    },
                    "displacement_px": pair["displacement_px"],
                    "displacement_norm": dist_norm,
                    "displacement_mm": dist_mm,
                    "confidence": min(b.get("confidence", 1.0), a.get("confidence", 1.0)),
                    "priority": 0,
                    "laser_angle_deg": angle_deg,
                    "laser_dx": dx,
                    "laser_dy": dy
                })

        return displaced_list

    @staticmethod
    def create_restoration_queue(displaced_balls: List[Dict[str, Any]]) -> Dict[str, Any]:
        def sort_key(ball):
            # 1. Largest distance first (-round(dist, 4))
            # 2. Highest confidence first (-round(conf, 3))
            # 3. Class hierarchy index
            dist = round(ball.get("displacement_norm", 0), 4)
            conf = round(ball.get("confidence", 0), 3)
            cid = ball.get("ball_class_id", 0)
            prio = SNOOKER_CLASS_HIERARCHY.get(cid, 99)
            return (-dist, -conf, prio)


        sorted_balls = sorted(displaced_balls, key=sort_key)
        for idx, ball in enumerate(sorted_balls):
            ball["priority"] = idx + 1

        return {
            "displaced_balls": sorted_balls,
            "total_displaced": len(sorted_balls),
            "current_index": 0,
            "is_complete": len(sorted_balls) == 0
        }

restoration_service = RestorationService()
