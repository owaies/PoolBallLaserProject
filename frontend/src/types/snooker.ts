export interface BallState {
  id: string;
  class_id: number;
  class_name: string;
  x: number;            // Pixel center x
  y: number;            // Pixel center y
  norm_x: number;       // Normalized x (0.0 to 1.0)
  norm_y: number;       // Normalized y (0.0 to 1.0)
  table_x_mm?: number;  // Calibrated table x in mm
  table_y_mm?: number;  // Calibrated table y in mm
  confidence: number;
  color: string;
  bbox?: [number, number, number, number]; // [xmin, ymin, xmax, ymax]
  isPocketed?: boolean;
}

export interface TableState {
  timestamp: number;
  shotNumber: number;
  frameNumber: number;
  currentPlayer: string;
  scores: { player1: number; player2: number };
  ballOn: number; // 1 = Red, 2 = Yellow, 3 = Green, 4 = Brown, 5 = Blue, 6 = Pink, 7 = Black
  cueBall: BallState;
  balls: BallState[];
}

export type RefereeDecisionType =
  | "confirmed"
  | "rejected"
  | "foul_and_miss"
  | "no_miss"
  | "override";

export type RestorationMode = "automatic" | "manual" | "none";

/** Explicit states for the one-ball-at-a-time restoration state machine */
export type RestorationState =
  | "IDLE"
  | "SELECT_BALL"
  | "SHOW_LASER"
  | "AWAITING_RESTORE_CONFIRMATION"
  | "RESTORING_BALL"
  | "BALL_RESTORED"
  | "NEXT_BALL"
  | "RESTORATION_COMPLETE";

/** Full wizard state carried by SnookerSimulation / SnookerPhoto */
export interface RestorationWizardState {
  mode: RestorationMode;
  currentState: RestorationState;
  /** Ordered queue: sorted by auto-priority for automatic, referee-insertion-order for manual */
  queue: DisplacedBall[];
  /** Index into queue of the ball currently being restored */
  currentIndex: number;
  restoredIds: string[];
  skippedIds: string[];
}

export interface ShotRecord {
  shotNumber: number;
  player: string;
  beforeState: TableState;
  afterState: TableState;
  ballOn: number;
  firstContact: BallState | null;
  pottedBalls: BallState[];
  foul: boolean;
  foulType: string;
  penalty: number;
  foulAndMiss: boolean;
  restorationRequired: boolean;
  restorationMethod: RestorationMode;
  restorationCompleted: boolean;
  refereeDecision: RefereeDecisionType;
  refereeOverrideNote?: string;
}

export interface MatchedBallPair {
  beforeBall: BallState;
  afterBall: BallState;
  displacement_px: number;
  displacement_norm: number;
  displacement_mm: number;
}

export interface DisplacedBall {
  id: string;
  ball_class_id: number;
  ball_class_name: string;
  /** Human-readable label, e.g. "Red #3" for the third red ball */
  label?: string;
  color: string;
  original_pos: {
    x: number;
    y: number;
    norm_x: number;
    norm_y: number;
    table_x_mm?: number;
    table_y_mm?: number;
  };
  current_pos: {
    x: number;
    y: number;
    norm_x: number;
    norm_y: number;
    table_x_mm?: number;
    table_y_mm?: number;
  };
  displacement_px: number;
  displacement_norm: number;
  displacement_mm: number;
  confidence: number;
  priority: number; // 1-based index in queue
  laser_angle_deg: number;
  laser_dx: number;
  laser_dy: number;
  isSelected?: boolean;
  /** Restoration status — always present once the wizard is active */
  status: "PENDING" | "RESTORED" | "SKIPPED";
}

export interface RestorationQueue {
  displaced_balls: DisplacedBall[];
  total_displaced: number;
  current_index: number; // For step-by-step restoration wizard
  is_complete: boolean;
  mode: RestorationMode;
}

export interface TableCalibrationPoints {
  topLeft: { x: number; y: number };
  topRight: { x: number; y: number };
  bottomRight: { x: number; y: number };
  bottomLeft: { x: number; y: number };
  isCalibrated: boolean;
  tableWidthMm: number;  // Official Snooker: 3569 mm
  tableHeightMm: number; // Official Snooker: 1778 mm
}

export interface FoulResult {
  isFoul: boolean;
  foulReason: string;
  penaltyPoints: number;
  ballOn: number;
  firstContactBall: BallState | null;
  pottedBalls: BallState[];
  isFoulAndMiss: boolean;
  displacedBallsCount: number;
  ruleSetName: string; // "Simulation Rules (Implemented Rule Set)"
}

export interface PhysicsBall {
  id: string;
  class_id: number;
  class_name: string;
  color: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  isPocketed: boolean;
  originalX: number;
  originalY: number;
}
