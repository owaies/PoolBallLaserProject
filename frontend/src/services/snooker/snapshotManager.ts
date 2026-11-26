import type { TableState, ShotRecord, BallState } from "../../types/snooker";

export class SnapshotManager {
  private history: ShotRecord[] = [];
  private undoStack: ShotRecord[] = [];

  /**
   * Captures an immutable snapshot of the table state.
   */
  static createSnapshot(
    balls: BallState[],
    cueBall: BallState,
    currentPlayer: string,
    scores: { player1: number; player2: number },
    ballOn: number,
    shotNumber: number,
    frameNumber: number = 1
  ): TableState {
    return {
      timestamp: Date.now(),
      shotNumber,
      frameNumber,
      currentPlayer,
      scores: { ...scores },
      ballOn,
      cueBall: JSON.parse(JSON.stringify(cueBall)),
      balls: JSON.parse(JSON.stringify(balls)),
    };
  }

  recordShot(shot: ShotRecord) {
    // Deep clone shot record to maintain immutability
    const clone: ShotRecord = JSON.parse(JSON.stringify(shot));
    this.history.push(clone);
    this.undoStack = []; // Clear redo stack on new shot
  }

  getHistory(): ShotRecord[] {
    return [...this.history];
  }

  getLastShot(): ShotRecord | null {
    if (this.history.length === 0) return null;
    return this.history[this.history.length - 1];
  }

  undoRestoration(): ShotRecord | null {
    if (this.history.length === 0) return null;
    const lastShot = this.history.pop()!;
    this.undoStack.push(lastShot);
    return lastShot;
  }

  redoRestoration(): ShotRecord | null {
    if (this.undoStack.length === 0) return null;
    const shot = this.undoStack.pop()!;
    this.history.push(shot);
    return shot;
  }

  clear() {
    this.history = [];
    this.undoStack = [];
  }
}
