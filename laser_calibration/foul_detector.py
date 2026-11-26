"""
Snooker & Pool Foul Detection & Laser Referee Assistant Engine
--------------------------------------------------------------
Customized for User Setup:
- Orange balls used as 1-pt object balls (instead of Red)
- Active Ball Set: Orange, Yellow, Green, Blue, Pink, White (No Black/Brown)
- Detects:
  1. Cue Ball In-Off (Scratch / Potted)
  2. Illegal Color Potted when on Orange
  3. Orange Potted when on Color
  4. Wrong Color Potted
  5. Multiple Colors Potted
  6. Displaced Balls Tracking & Automated Physical Laser Repositioning
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import math
from laser_calibration.snooker_geometry import SNOOKER, get_snooker_pockets

SNOOKER_POINTS = {
    "orange": 1,
    "red": 1,
    "yellow": 2,
    "green": 3,
    "brown": 4,
    "blue": 5,
    "pink": 6,
    "black": 7,
    "white": 4 # standard scratch penalty
}

# User's active table regulation spots (No Black / No Brown)
REGULATION_SPOTS = {
    "pink": {"name": "Pink Spot", "x_mm": 584.2, "y_mm": 1756.4, "points": 6},
    "blue": {"name": "Blue Spot (Center)", "x_mm": 584.2, "y_mm": 1170.94, "points": 5},
    "green": {"name": "Green Spot (Baulk Left)", "x_mm": 876.3, "y_mm": 737.0, "points": 3},
    "yellow": {"name": "Yellow Spot (Baulk Right)", "x_mm": 292.1, "y_mm": 737.0, "points": 2},
    "cue": {"name": "Cue Ball In Hand ('D' Center)", "x_mm": 584.2, "y_mm": 737.0, "points": 4}
}

OBJECT_BALLS = ["orange", "red"]
COLOR_BALLS = ["yellow", "green", "blue", "pink"]

def normalize_label(label: str) -> str:
    """Normalize label to simple canonical key ('orange', 'yellow', 'blue', etc.)."""
    l = label.lower().replace("-ball", "").replace(" ball", "").strip()
    if l in ["red", "orange"]:
        return "orange" # User uses orange balls as object balls
    return l

@dataclass
class BallState:
    id: int
    label: str
    norm_label: str
    table_x_mm: float
    table_y_mm: float
    confidence: float = 1.0
    pixel_x: float = 0.0
    pixel_y: float = 0.0

@dataclass
class DisplacedBall:
    ball_id: int
    label: str
    norm_label: str
    pre_x: float
    pre_y: float
    post_x: Optional[float]
    post_y: Optional[float]
    displacement_mm: float
    status: str # "MOVED", "POTTED", "OFF_TABLE", "RESTORED"
    laser_pan: Optional[float] = None
    laser_tilt: Optional[float] = None

@dataclass
class FoulReport:
    is_foul: bool
    foul_type: Optional[str] = None
    penalty_points: int = 0
    description: str = ""
    nominated_target: str = "orange"
    potted_balls: List[str] = field(default_factory=list)
    displaced_balls: List[DisplacedBall] = field(default_factory=list)
    cue_ball_in_off: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

class FoulDetector:
    def __init__(self, interpolator=None):
        self.interpolator = interpolator
        self.pre_shot_state: List[BallState] = []
        self.last_foul_report: Optional[FoulReport] = None
        self.movement_threshold_mm = 20.0

    def set_reference_state(self, balls_data: List[Dict[str, Any]]):
        """
        Record the pre-shot snapshot of all balls on the table.
        """
        self.pre_shot_state = []
        for idx, b in enumerate(balls_data):
            tx = b.get("table_x_mm") or b.get("x")
            ty = b.get("table_y_mm") or b.get("y")
            if tx is None or ty is None:
                continue
            lbl = str(b.get("label", b.get("class", "unknown")))
            self.pre_shot_state.append(BallState(
                id=int(b.get("id", idx + 1)),
                label=lbl,
                norm_label=normalize_label(lbl),
                table_x_mm=float(tx),
                table_y_mm=float(ty),
                confidence=float(b.get("confidence", b.get("conf", 1.0))),
                pixel_x=float(b.get("pixel_x", b.get("cx", 0.0))),
                pixel_y=float(b.get("pixel_y", b.get("cy", 0.0)))
            ))
        return len(self.pre_shot_state)

    def rollover_to_next_preshot(self, current_balls_data: List[Dict[str, Any]]):
        """
        After a clean legal shot, automatically roll over the post-shot state into the new pre-shot baseline.
        """
        return self.set_reference_state(current_balls_data)

    def analyze_shot(self, current_balls_data: List[Dict[str, Any]], nominated_target: str = "orange") -> FoulReport:
        """
        Compare current post-shot state with pre_shot_state and evaluate snooker rules tailored to user ball set.
        """
        norm_target = normalize_label(nominated_target)
        if not self.pre_shot_state:
            self.set_reference_state(current_balls_data)
            rep = FoulReport(
                is_foul=False,
                description="Pre-shot reference baseline established. Ready for shot analysis.",
                nominated_target=norm_target
            )
            self.last_foul_report = rep
            return rep

        curr_state: List[BallState] = []
        for idx, b in enumerate(current_balls_data):
            tx = b.get("table_x_mm") or b.get("x")
            ty = b.get("table_y_mm") or b.get("y")
            if tx is None or ty is None:
                continue
            lbl = str(b.get("label", b.get("class", "unknown")))
            curr_state.append(BallState(
                id=int(b.get("id", idx + 1)),
                label=lbl,
                norm_label=normalize_label(lbl),
                table_x_mm=float(tx),
                table_y_mm=float(ty),
                confidence=float(b.get("confidence", b.get("conf", 1.0))),
                pixel_x=float(b.get("pixel_x", b.get("cx", 0.0))),
                pixel_y=float(b.get("pixel_y", b.get("cy", 0.0)))
            ))

        # 1. Match pre-shot balls to post-shot balls
        matched_post_indices = set()
        displaced_list: List[DisplacedBall] = []
        potted_list: List[BallState] = []

        for pre_b in self.pre_shot_state:
            best_curr_idx = -1
            best_dist = float('inf')

            for c_idx, curr_b in enumerate(curr_state):
                if c_idx in matched_post_indices:
                    continue
                if curr_b.norm_label == pre_b.norm_label:
                    d = math.hypot(curr_b.table_x_mm - pre_b.table_x_mm, curr_b.table_y_mm - pre_b.table_y_mm)
                    if d < best_dist:
                        best_dist = d
                        best_curr_idx = c_idx

            if best_curr_idx >= 0 and best_dist < 2500.0:
                matched_post_indices.add(best_curr_idx)
                curr_match = curr_state[best_curr_idx]
                if best_dist > self.movement_threshold_mm:
                    pan, tilt = self._calc_laser(pre_b.table_x_mm, pre_b.table_y_mm)
                    displaced_list.append(DisplacedBall(
                        ball_id=pre_b.id,
                        label=pre_b.label,
                        norm_label=pre_b.norm_label,
                        pre_x=round(pre_b.table_x_mm, 1),
                        pre_y=round(pre_b.table_y_mm, 1),
                        post_x=round(curr_match.table_x_mm, 1),
                        post_y=round(curr_match.table_y_mm, 1),
                        displacement_mm=round(best_dist, 1),
                        status="MOVED",
                        laser_pan=pan,
                        laser_tilt=tilt
                    ))
            else:
                potted_list.append(pre_b)
                pan, tilt = self._calc_laser(pre_b.table_x_mm, pre_b.table_y_mm)
                displaced_list.append(DisplacedBall(
                    ball_id=pre_b.id,
                    label=pre_b.label,
                    norm_label=pre_b.norm_label,
                    pre_x=round(pre_b.table_x_mm, 1),
                    pre_y=round(pre_b.table_y_mm, 1),
                    post_x=None,
                    post_y=None,
                    displacement_mm=9999.0,
                    status="POTTED",
                    laser_pan=pan,
                    laser_tilt=tilt
                ))

        # 2. Check for Rules & Fouls (Orange = 1 pt object ball; Yellow, Green, Blue, Pink = colors)
        cue_in_off = any(b.norm_label == "white" for b in potted_list)
        potted_colors = [b for b in potted_list if b.norm_label in COLOR_BALLS]
        potted_objects = [b for b in potted_list if b.norm_label in OBJECT_BALLS]

        is_foul = False
        foul_type = None
        penalty = 0
        desc = ""

        # FOUL 1: Cue ball potted (Scratch / In-off)
        if cue_in_off:
            is_foul = True
            foul_type = "CUE_BALL_IN_OFF"
            max_pt = 4
            for c in potted_colors:
                max_pt = max(max_pt, SNOOKER_POINTS.get(c.norm_label, 4))
            penalty = max_pt
            desc = f"FOUL: White Cue Ball potted (In-Off / Scratch)! Penalty: {penalty} points."

        # FOUL 2: Illegal color ball potted when player was on Orange
        elif norm_target in OBJECT_BALLS and len(potted_colors) > 0:
            is_foul = True
            foul_type = "ILLEGAL_COLOR_POTTED"
            highest_col = max(potted_colors, key=lambda c: SNOOKER_POINTS.get(c.norm_label, 4))
            penalty = max(4, SNOOKER_POINTS.get(highest_col.norm_label, 4))
            color_names = ", ".join([c.label for c in potted_colors])
            desc = f"FOUL: Illegal color ball potted ({color_names}) while player was on Orange! Penalty: {penalty} points."

        # FOUL 3: Orange potted when player was on Color
        elif norm_target in COLOR_BALLS:
            if len(potted_objects) > 0:
                is_foul = True
                foul_type = "ORANGE_POTTED_ON_COLOR"
                penalty = 4
                desc = f"FOUL: Orange ball potted while player was nominated on {norm_target.capitalize()}! Penalty: 4 points."
            elif any(c.norm_label != norm_target for c in potted_colors):
                wrong_cols = [c for c in potted_colors if c.norm_label != norm_target]
                is_foul = True
                foul_type = "WRONG_COLOR_POTTED"
                highest_col = max(wrong_cols, key=lambda c: SNOOKER_POINTS.get(c.norm_label, 4))
                penalty = max(4, SNOOKER_POINTS.get(highest_col.norm_label, 4), SNOOKER_POINTS.get(norm_target, 4))
                desc = f"FOUL: Wrong color potted ({wrong_cols[0].label})! Nominated was {norm_target.capitalize()}. Penalty: {penalty} points."

        # FOUL 4: Multiple colors potted
        if not is_foul and len(potted_colors) > 1:
            is_foul = True
            foul_type = "MULTIPLE_COLORS_POTTED"
            highest_pt = max([SNOOKER_POINTS.get(c.norm_label, 4) for c in potted_colors])
            penalty = max(4, highest_pt)
            desc = f"FOUL: Multiple colored balls potted ({len(potted_colors)} colors)! Penalty: {penalty} points."

        # LEGAL POT / SHOT
        if not is_foul:
            if norm_target in OBJECT_BALLS and len(potted_objects) > 0:
                pts = len(potted_objects)
                desc = f"LEGAL SHOT: {len(potted_objects)} Orange ball(s) potted! Score: +{pts} point(s)."
            elif norm_target in COLOR_BALLS and any(c.norm_label == norm_target for c in potted_colors):
                pts = SNOOKER_POINTS.get(norm_target, 4)
                desc = f"LEGAL SHOT: Nominated {norm_target.capitalize()} successfully potted! Score: +{pts} points."
            elif len(displaced_list) > 0:
                desc = f"Legal shot completed: {len(displaced_list)} balls moved. No balls potted."
            else:
                desc = "Shot complete. No foul detected."

        report = FoulReport(
            is_foul=is_foul,
            foul_type=foul_type,
            penalty_points=penalty,
            description=desc,
            nominated_target=norm_target,
            potted_balls=[b.label for b in potted_list],
            displaced_balls=displaced_list,
            cue_ball_in_off=cue_in_off,
            details={
                "pre_shot_ball_count": len(self.pre_shot_state),
                "post_shot_ball_count": len(curr_state),
                "potted_count": len(potted_list),
                "displaced_count": len(displaced_list)
            }
        )
        self.last_foul_report = report
        return report

    def _calc_laser(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        if self.interpolator is not None:
            try:
                pan, tilt, _ = self.interpolator.predict(x, y)
                return round(float(pan), 1), round(float(tilt), 1)
            except Exception:
                pass
        return None, None
