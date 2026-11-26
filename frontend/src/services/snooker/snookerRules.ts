import type { BallState, FoulResult } from "../../types/snooker";

export const SNOOKER_BALL_VALUES: Record<number, number> = {
  0: 4, // Cue ball (scratch penalty default = 4)
  1: 1, // Red = 1
  2: 2, // Yellow = 2
  3: 3, // Green = 3
  4: 4, // Brown = 4
  5: 5, // Blue = 5
  6: 6, // Pink = 6
  7: 7, // Black = 7
};

export class SnookerRulesEngine {
  public static readonly RULESET_NAME = "Simulation Rules (Implemented Rule Set)";

  /**
   * Evaluates if a foul occurred during the shot execution.
   */
  static evaluateShot(
    firstContactBall: BallState | null,
    pottedBalls: BallState[],
    ballOn: number = 1
  ): FoulResult {
    const cueBallPotted = pottedBalls.some((b) => b.class_id === 0);

    // 1. Cue Ball Pocketed (Scratch Foul)
    if (cueBallPotted) {
      return {
        isFoul: true,
        foulReason: "Cue Ball Pocketed (Scratch Foul)",
        penaltyPoints: 4,
        ballOn,
        firstContactBall,
        pottedBalls,
        isFoulAndMiss: true,
        displacedBallsCount: 0,
        ruleSetName: this.RULESET_NAME,
      };
    }

    // 2. No ball contacted (Miss Foul)
    if (!firstContactBall) {
      const penalty = Math.max(4, SNOOKER_BALL_VALUES[ballOn] || 4);
      return {
        isFoul: true,
        foulReason: "Foul: Cue Ball failed to contact any target ball",
        penaltyPoints: penalty,
        ballOn,
        firstContactBall: null,
        pottedBalls,
        isFoulAndMiss: true,
        displacedBallsCount: 0,
        ruleSetName: this.RULESET_NAME,
      };
    }

    // 3. Wrong ball contacted first
    if (firstContactBall.class_id !== ballOn) {
      // For Red ON (1), hitting any non-red is a foul
      const isWrongContact = ballOn === 1 ? firstContactBall.class_id !== 1 : firstContactBall.class_id !== ballOn;

      if (isWrongContact) {
        const contactBallName = firstContactBall.class_name.replace("_ball", "");
        const contactValue = SNOOKER_BALL_VALUES[firstContactBall.class_id] || 4;
        const ballOnValue = SNOOKER_BALL_VALUES[ballOn] || 4;
        const penalty = Math.max(4, contactValue, ballOnValue);

        return {
          isFoul: true,
          foulReason: `Foul: Hit ${contactBallName.toUpperCase()} first instead of Ball ON`,
          penaltyPoints: penalty,
          ballOn,
          firstContactBall,
          pottedBalls,
          isFoulAndMiss: true,
          displacedBallsCount: 0,
          ruleSetName: this.RULESET_NAME,
        };
      }
    }

    // 4. Illegal ball potted
    const illegalPotted = pottedBalls.find((b) => b.class_id !== ballOn);
    if (illegalPotted) {
      const illegalValue = SNOOKER_BALL_VALUES[illegalPotted.class_id] || 4;
      const penalty = Math.max(4, illegalValue);
      return {
        isFoul: true,
        foulReason: `Foul: Illegally potted ${illegalPotted.class_name.replace("_ball", "").toUpperCase()}`,
        penaltyPoints: penalty,
        ballOn,
        firstContactBall,
        pottedBalls,
        isFoulAndMiss: false,
        displacedBallsCount: 0,
        ruleSetName: this.RULESET_NAME,
      };
    }

    return {
      isFoul: false,
      foulReason: "Legal Shot",
      penaltyPoints: 0,
      ballOn,
      firstContactBall,
      pottedBalls,
      isFoulAndMiss: false,
      displacedBallsCount: 0,
      ruleSetName: this.RULESET_NAME,
    };
  }
}
