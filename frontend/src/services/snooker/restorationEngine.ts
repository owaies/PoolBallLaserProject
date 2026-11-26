import type {
  BallState,
  MatchedBallPair,
  DisplacedBall,
  RestorationQueue,
  TableCalibrationPoints,
  RestorationMode,
} from "../../types/snooker";

export const SNOOKER_CLASS_NAMES: Record<number, string> = {
  0: "cue_ball",
  1: "red_ball",
  2: "yellow_ball",
  3: "green_ball",
  4: "brown_ball",
  5: "blue_ball",
  6: "pink_ball",
  7: "black_ball",
};

export const SNOOKER_BALL_COLORS: Record<number, string> = {
  0: "#FFFFFF", // Cue
  1: "#DC2626", // Red
  2: "#FACC15", // Yellow
  3: "#16A34A", // Green
  4: "#92400E", // Brown
  5: "#2563EB", // Blue
  6: "#EC4899", // Pink
  7: "#18181B", // Black
};

export const SNOOKER_CLASS_HIERARCHY: Record<number, number> = {
  0: 0, // cue_ball
  1: 1, // red_ball
  2: 2, // yellow_ball
  3: 3, // green_ball
  4: 4, // brown_ball
  5: 5, // blue_ball
  6: 6, // pink_ball
  7: 7, // black_ball
};

export class RestorationEngine {
  /**
   * Spatial Class-Aware Ball Matching
   * Matches balls between 'Before Shot' and 'After Shot' states.
   * For uniquely colored balls (cue, yellow, green, brown, blue, pink, black), matches by class_id.
   * For red balls (multiple instances of class 1), performs greedy/nearest-neighbour distance matching in normalized (x,y) space.
   */
  static spatialMatchBalls(
    beforeBalls: BallState[],
    afterBalls: BallState[]
  ): MatchedBallPair[] {
    const matchedPairs: MatchedBallPair[] = [];

    // Group balls by class_id
    const beforeByClass: Record<number, BallState[]> = {};
    const afterByClass: Record<number, BallState[]> = {};

    for (const ball of beforeBalls) {
      if (!beforeByClass[ball.class_id]) beforeByClass[ball.class_id] = [];
      beforeByClass[ball.class_id].push(ball);
    }
    for (const ball of afterBalls) {
      if (!afterByClass[ball.class_id]) afterByClass[ball.class_id] = [];
      afterByClass[ball.class_id].push(ball);
    }

    const allClassIds = Array.from(
      new Set([...Object.keys(beforeByClass), ...Object.keys(afterByClass)])
    ).map(Number);

    for (const cid of allClassIds) {
      const bList = beforeByClass[cid] || [];
      const aList = afterByClass[cid] || [];

      if (bList.length === 0 || aList.length === 0) continue;

      if (cid !== 1) {
        // Unique colored balls or 1-to-1 match
        const minLen = Math.min(bList.length, aList.length);
        for (let i = 0; i < minLen; i++) {
          const b = bList[i];
          const a = aList[i];
          const dxNorm = a.norm_x - b.norm_x;
          const dyNorm = a.norm_y - b.norm_y;
          const distNorm = Math.sqrt(dxNorm * dxNorm + dyNorm * dyNorm);

          const dxPx = a.x - b.x;
          const dyPx = a.y - b.y;
          const distPx = Math.sqrt(dxPx * dxPx + dyPx * dyPx);

          matchedPairs.push({
            beforeBall: b,
            afterBall: a,
            displacement_norm: distNorm,
            displacement_px: distPx,
            displacement_mm: distNorm * 3569,
          });
        }
      } else {
        // Red balls (multiple instances) - Nearest-neighbour spatial matrix solver
        const remainingB = [...bList];
        const remainingA = [...aList];

        while (remainingB.length > 0 && remainingA.length > 0) {
          let bestDist = Infinity;
          let bestBIdx = -1;
          let bestAIdx = -1;

          for (let i = 0; i < remainingB.length; i++) {
            for (let j = 0; j < remainingA.length; j++) {
              const b = remainingB[i];
              const a = remainingA[j];
              const dx = a.norm_x - b.norm_x;
              const dy = a.norm_y - b.norm_y;
              const dist = Math.sqrt(dx * dx + dy * dy);

              if (dist < bestDist) {
                bestDist = dist;
                bestBIdx = i;
                bestAIdx = j;
              }
            }
          }

          if (bestBIdx !== -1 && bestAIdx !== -1) {
            const b = remainingB.splice(bestBIdx, 1)[0];
            const a = remainingA.splice(bestAIdx, 1)[0];

            const dxPx = a.x - b.x;
            const dyPx = a.y - b.y;
            const distPx = Math.sqrt(dxPx * dxPx + dyPx * dyPx);

            matchedPairs.push({
              beforeBall: b,
              afterBall: a,
              displacement_norm: bestDist,
              displacement_px: distPx,
              displacement_mm: bestDist * 3569,
            });
          } else {
            break;
          }
        }
      }
    }

    return matchedPairs;
  }

  /**
   * Filter pairs exceeding minimum displacement threshold.
   */
  static detectDisplacedBalls(
    matchedPairs: MatchedBallPair[],
    thresholdNorm: number = 0.003, // ~10px in 3000px space
    calibration?: TableCalibrationPoints
  ): DisplacedBall[] {
    const displacedList: DisplacedBall[] = [];

    for (const pair of matchedPairs) {
      let isDisplaced = false;
      let distMm = pair.displacement_mm;

      if (calibration && calibration.isCalibrated) {
        const dxMm = (pair.afterBall.norm_x - pair.beforeBall.norm_x) * calibration.tableWidthMm;
        const dyMm = (pair.afterBall.norm_y - pair.beforeBall.norm_y) * calibration.tableHeightMm;
        distMm = Math.sqrt(dxMm * dxMm + dyMm * dyMm);
        isDisplaced = distMm > 5.0; // 5 mm threshold
      } else {
        isDisplaced = pair.displacement_norm > thresholdNorm;
      }

      if (isDisplaced) {
        const dx = pair.beforeBall.norm_x - pair.afterBall.norm_x;
        const dy = pair.beforeBall.norm_y - pair.afterBall.norm_y;
        const angleRad = Math.atan2(dy, dx);
        const angleDeg = (angleRad * 180) / Math.PI;

        displacedList.push({
          id: `displaced_${pair.beforeBall.id}`,
          ball_class_id: pair.beforeBall.class_id,
          ball_class_name: pair.beforeBall.class_name,
          color: SNOOKER_BALL_COLORS[pair.beforeBall.class_id] || "#FFFFFF",
          original_pos: {
            x: pair.beforeBall.x,
            y: pair.beforeBall.y,
            norm_x: pair.beforeBall.norm_x,
            norm_y: pair.beforeBall.norm_y,
            table_x_mm: pair.beforeBall.norm_x * 3569,
            table_y_mm: pair.beforeBall.norm_y * 1778,
          },
          current_pos: {
            x: pair.afterBall.x,
            y: pair.afterBall.y,
            norm_x: pair.afterBall.norm_x,
            norm_y: pair.afterBall.norm_y,
            table_x_mm: pair.afterBall.norm_x * 3569,
            table_y_mm: pair.afterBall.norm_y * 1778,
          },
          displacement_px: pair.displacement_px,
          displacement_norm: pair.displacement_norm,
          displacement_mm: distMm,
          confidence: Math.min(pair.beforeBall.confidence, pair.afterBall.confidence),
          priority: 0,
          laser_angle_deg: angleDeg,
          laser_dx: dx,
          laser_dy: dy,
          isSelected: true,
          status: "PENDING",
        });
      }
    }

    return displacedList;
  }

  /**
   * Automatic Restoration Priority Queue Sorting Rule:
   * 1. Distance (largest displacement first)
   * 2. Confidence (higher confidence first)
   * 3. Deterministic Class Hierarchy priority (cue_ball -> red -> yellow -> green -> brown -> blue -> pink -> black)
   */
  static createRestorationQueue(
    displacedBalls: DisplacedBall[],
    mode: RestorationMode = "automatic"
  ): RestorationQueue {
    let sorted: DisplacedBall[] = [];

    if (mode === "automatic") {
      sorted = [...displacedBalls].sort((a, b) => {
        // 1. Compare displacement distance (round to 4 decimals for precision)
        const distA = Math.round(a.displacement_norm * 10000) / 10000;
        const distB = Math.round(b.displacement_norm * 10000) / 10000;
        if (distA !== distB) {
          return distB - distA;
        }

        // 2. Tie-break: Confidence
        const confA = Math.round(a.confidence * 1000) / 1000;
        const confB = Math.round(b.confidence * 1000) / 1000;
        if (confA !== confB) {
          return confB - confA;
        }

        // 3. Tie-break: Class Hierarchy
        const priorityA = SNOOKER_CLASS_HIERARCHY[a.ball_class_id] ?? 99;
        const priorityB = SNOOKER_CLASS_HIERARCHY[b.ball_class_id] ?? 99;
        return priorityA - priorityB;
      });
    } else {
      // Manual Mode preserves referee selection order
      sorted = [...displacedBalls];
    }

    // Assign priority indices
    sorted.forEach((ball, idx) => {
      ball.priority = idx + 1;
    });

    return {
      displaced_balls: sorted,
      total_displaced: sorted.length,
      current_index: 0,
      is_complete: sorted.length === 0,
      mode,
    };
  }

  /**
   * Calculates Laser Vector direction and distance.
   */
  static calculateLaserDirection(
    currentPos: { x: number; y: number },
    originalPos: { x: number; y: number }
  ) {
    const dx = originalPos.x - currentPos.x;
    const dy = originalPos.y - currentPos.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;

    return {
      dx,
      dy,
      distance,
      angleDeg,
    };
  }
}
