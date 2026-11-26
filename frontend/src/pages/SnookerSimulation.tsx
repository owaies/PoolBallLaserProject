import { useState, useCallback } from "react";
import { SnookerPhysics } from "../services/snooker/snookerPhysics";
import { SnookerRulesEngine } from "../services/snooker/snookerRules";
import { RestorationEngine } from "../services/snooker/restorationEngine";
import { SnapshotManager } from "../services/snooker/snapshotManager";
import { SnookerTable } from "../components/snooker/SnookerTable";
import { RefereePanel } from "../components/snooker/RefereePanel";
import { RestorationWizard } from "../components/snooker/RestorationWizard";
import { ShotHistoryDrawer } from "../components/snooker/ShotHistoryDrawer";
import type {
  BallState,
  DisplacedBall,
  RestorationMode,
  RestorationState,
  RestorationWizardState,
  FoulResult,
  ShotRecord,
  RefereeDecisionType,
} from "../types/snooker";
import { Sparkles, RefreshCw } from "lucide-react";
import { toast } from "react-hot-toast";

// ── Helper: build wizard state from displaced balls ──────────────────────────
function buildWizard(
  displaced: DisplacedBall[],
  mode: RestorationMode
): RestorationWizardState {
  const queue = displaced.map((b) => ({ ...b, status: "PENDING" as const }));
  return {
    mode,
    currentState: mode === "manual" ? "SELECT_BALL" : "SHOW_LASER",
    queue,
    currentIndex: 0,
    restoredIds: [],
    skippedIds: [],
  };
}

// ── Helper: get current pending index ───────────────────────────────────────
function nextPendingIndex(queue: DisplacedBall[], from: number): number {
  for (let i = from; i < queue.length; i++) {
    if (queue[i].status === "PENDING") return i;
  }
  return queue.length; // none found → complete
}

export default function SnookerSimulation() {
  const [physics] = useState(() => new SnookerPhysics(900, 480));
  const [snapshotMgr] = useState(() => new SnapshotManager());

  const [beforeState, setBeforeState] = useState<BallState[]>([]);
  const [foulResult, setFoulResult] = useState<FoulResult | null>(null);
  const [showRefereePanel, setShowRefereePanel] = useState<boolean>(false);
  const [wizard, setWizard] = useState<RestorationWizardState | null>(null);
  const [shotHistory, setShotHistory] = useState<ShotRecord[]>([]);

  const [scores, setScores] = useState({ player1: 0, player2: 0 });
  const [currentPlayer, setCurrentPlayer] = useState<string>("Player 1");
  const [ballOn] = useState<number>(1);
  const [shotCount, setShotCount] = useState<number>(1);

  // Active displaced ball fed to the canvas for laser drawing
  const activeBall: DisplacedBall | null =
    wizard && wizard.currentIndex < wizard.queue.length && wizard.currentState !== "RESTORATION_COMPLETE"
      ? wizard.queue[wizard.currentIndex]
      : null;

  // ── Capture table state before cue strike ────────────────────────────────
  const captureBeforeState = useCallback(() => {
    const state: BallState[] = physics.balls
      .filter((b) => !b.isPocketed)
      .map((b) => ({
        id: b.id,
        class_id: b.class_id,
        class_name: b.class_name,
        x: b.x,
        y: b.y,
        norm_x: b.x / 900,
        norm_y: b.y / 480,
        confidence: 0.99,
        color: b.color,
      }));
    setBeforeState(state);
  }, [physics]);

  // ── Shot finish handler ───────────────────────────────────────────────────
  const handleShotFinish = useCallback(() => {
    const afterState: BallState[] = physics.balls
      .filter((b) => !b.isPocketed)
      .map((b) => ({
        id: b.id,
        class_id: b.class_id,
        class_name: b.class_name,
        x: b.x,
        y: b.y,
        norm_x: b.x / 900,
        norm_y: b.y / 480,
        confidence: 0.99,
        color: b.color,
      }));

    const potted = physics.pottedBalls.map((b) => ({
      id: b.id,
      class_id: b.class_id,
      class_name: b.class_name,
      x: b.x,
      y: b.y,
      norm_x: b.x / 900,
      norm_y: b.y / 480,
      confidence: 0.99,
      color: b.color,
    }));

    const firstContactBall = physics.firstContactBall
      ? {
          id: physics.firstContactBall.id,
          class_id: physics.firstContactBall.class_id,
          class_name: physics.firstContactBall.class_name,
          x: physics.firstContactBall.x,
          y: physics.firstContactBall.y,
          norm_x: physics.firstContactBall.x / 900,
          norm_y: physics.firstContactBall.y / 480,
          confidence: 0.99,
          color: physics.firstContactBall.color,
        }
      : null;

    const result = SnookerRulesEngine.evaluateShot(firstContactBall, potted, ballOn);

    if (result.isFoul) {
      setFoulResult(result);
      setShowRefereePanel(true);
      toast.error(`FOUL DETECTED: ${result.foulReason} (+${result.penaltyPoints} Penalty)`);
      setScores((prev) =>
        currentPlayer === "Player 1"
          ? { ...prev, player2: prev.player2 + result.penaltyPoints }
          : { ...prev, player1: prev.player1 + result.penaltyPoints }
      );
    } else {
      toast.success("Legal shot completed!");
    }

    const shotRec: ShotRecord = {
      shotNumber: shotCount,
      player: currentPlayer,
      beforeState: SnapshotManager.createSnapshot(beforeState, beforeState[0] || afterState[0], currentPlayer, scores, ballOn, shotCount),
      afterState: SnapshotManager.createSnapshot(afterState, afterState[0], currentPlayer, scores, ballOn, shotCount),
      ballOn,
      firstContact: firstContactBall,
      pottedBalls: potted,
      foul: result.isFoul,
      foulType: result.foulReason,
      penalty: result.penaltyPoints,
      foulAndMiss: result.isFoulAndMiss,
      restorationRequired: result.isFoul,
      restorationMethod: "none",
      restorationCompleted: false,
      refereeDecision: "confirmed",
    };
    snapshotMgr.recordShot(shotRec);
    setShotHistory(snapshotMgr.getHistory());
    setShotCount((prev) => prev + 1);
  }, [physics, beforeState, ballOn, currentPlayer, scores, shotCount, snapshotMgr]);

  // ── Build displaced ball queue from current physics state ─────────────────
  const buildDisplacedQueue = useCallback(
    (): DisplacedBall[] => {
      const afterState: BallState[] = physics.balls.map((b) => ({
        id: b.id,
        class_id: b.class_id,
        class_name: b.class_name,
        x: b.x,
        y: b.y,
        norm_x: b.x / 900,
        norm_y: b.y / 480,
        confidence: 0.99,
        color: b.color,
      }));
      const matched = RestorationEngine.spatialMatchBalls(beforeState, afterState);
      const displaced = RestorationEngine.detectDisplacedBalls(matched, 0.003);

      // Add human-readable labels (Red #1, Red #2, etc.)
      let redCount = 0;
      return displaced.map((b) => {
        if (b.ball_class_id === 1) {
          redCount++;
          return { ...b, label: `Red #${redCount}`, status: "PENDING" as const };
        }
        return {
          ...b,
          label: b.ball_class_name.replace("_ball", "").replace(/^\w/, (c) => c.toUpperCase()),
          status: "PENDING" as const,
        };
      });
    },
    [physics, beforeState]
  );

  // ── Referee triggers automatic restoration ────────────────────────────────
  const handleAutomaticRestore = useCallback(() => {
    setShowRefereePanel(false);
    const displaced = buildDisplacedQueue();
    if (displaced.length === 0) {
      toast("No displaced balls detected.");
      return;
    }
    const sorted = RestorationEngine.createRestorationQueue(
      displaced,
      "automatic"
    ).displaced_balls;
    const w = buildWizard(sorted, "automatic");
    setWizard(w);
    toast.success(`Automatic restoration: ${sorted.length} ball${sorted.length > 1 ? "s" : ""} queued`);
  }, [buildDisplacedQueue]);

  // ── Referee triggers manual restoration ──────────────────────────────────
  const handleManualRestore = useCallback(() => {
    setShowRefereePanel(false);
    const displaced = buildDisplacedQueue();
    if (displaced.length === 0) {
      toast("No displaced balls detected.");
      return;
    }
    const w = buildWizard(displaced, "manual");
    setWizard(w);
    toast.success(`Manual restoration: ${displaced.length} ball${displaced.length > 1 ? "s" : ""} queued`);
  }, [buildDisplacedQueue]);

  // ── Mode switch inside wizard ─────────────────────────────────────────────
  const handleWizardModeChange = useCallback(
    (newMode: RestorationMode) => {
      if (!wizard) return;
      // Re-sort existing pending balls when switching to automatic
      const pending = wizard.queue.filter((b) => b.status === "PENDING");
      const restored = wizard.queue.filter((b) => b.status !== "PENDING");
      let newQueue: DisplacedBall[];
      if (newMode === "automatic") {
        const sorted = RestorationEngine.createRestorationQueue(pending, "automatic").displaced_balls;
        newQueue = [...restored, ...sorted];
      } else {
        newQueue = [...restored, ...pending];
      }
      setWizard({
        mode: newMode,
        currentState: newMode === "manual" ? "SELECT_BALL" : "SHOW_LASER",
        queue: newQueue,
        currentIndex: nextPendingIndex(newQueue, 0),
        restoredIds: wizard.restoredIds,
        skippedIds: wizard.skippedIds,
      });
    },
    [wizard]
  );

  // ── Manual mode: referee selects a ball ──────────────────────────────────
  const handleSelectBall = useCallback(
    (id: string) => {
      if (!wizard) return;
      const idx = wizard.queue.findIndex((b) => b.id === id);
      if (idx === -1) return;
      setWizard((prev) =>
        prev
          ? {
              ...prev,
              currentIndex: idx,
              currentState: "SHOW_LASER",
            }
          : prev
      );
    },
    [wizard]
  );

  // ── Transition SHOW_LASER → AWAITING (canvas has drawn laser) ────────────
  // The SnookerTable calls this prop after the laser is rendered for the first time
  const handleLaserReady = useCallback(() => {
    setWizard((prev) => {
      if (!prev || prev.currentState !== "SHOW_LASER") return prev;
      return { ...prev, currentState: "AWAITING_RESTORE_CONFIRMATION" };
    });
  }, []);

  // ── "RESTORE THIS BALL" button pressed ───────────────────────────────────
  const handleRestoreThisBall = useCallback(() => {
    if (!wizard || !activeBall) return;
    // Transition to RESTORING_BALL — SnookerTable starts animation
    setWizard((prev) =>
      prev ? { ...prev, currentState: "RESTORING_BALL" } : prev
    );
  }, [wizard, activeBall]);

  // ── Canvas animation finished → commit restore ───────────────────────────
  const handleRestoreAnimationComplete = useCallback(() => {
    if (!wizard || !activeBall) return;
    // Physically move ball in physics engine
    physics.restoreBall(
      activeBall.ball_class_id,
      activeBall.original_pos.x,
      activeBall.original_pos.y
    );
    toast.success(`Restored: ${activeBall.label || activeBall.ball_class_name}`);

    setWizard((prev) => {
      if (!prev) return prev;
      const updatedQueue = prev.queue.map((b) =>
        b.id === activeBall.id ? { ...b, status: "RESTORED" as const } : b
      );
      const newRestored = [...prev.restoredIds, activeBall.id];
      const nextIdx = nextPendingIndex(updatedQueue, 0);
      const isComplete = nextIdx >= updatedQueue.length;
      let nextState: RestorationState = isComplete
        ? "RESTORATION_COMPLETE"
        : prev.mode === "manual"
        ? "SELECT_BALL"
        : "SHOW_LASER";

      return {
        ...prev,
        queue: updatedQueue,
        currentIndex: isComplete ? prev.currentIndex : nextIdx,
        restoredIds: newRestored,
        currentState: nextState,
      };
    });
  }, [wizard, activeBall, physics]);

  // ── Skip Ball ─────────────────────────────────────────────────────────────
  const handleSkipBall = useCallback(() => {
    if (!wizard || !activeBall) return;
    toast(`Skipped: ${activeBall.label || activeBall.ball_class_name}`, { icon: "⏭️" });

    setWizard((prev) => {
      if (!prev) return prev;
      const updatedQueue = prev.queue.map((b) =>
        b.id === activeBall.id ? { ...b, status: "SKIPPED" as const } : b
      );
      const newSkipped = [...prev.skippedIds, activeBall.id];
      const nextIdx = nextPendingIndex(updatedQueue, 0);
      const isComplete = nextIdx >= updatedQueue.length;
      let nextState: RestorationState = isComplete
        ? "RESTORATION_COMPLETE"
        : prev.mode === "manual"
        ? "SELECT_BALL"
        : "SHOW_LASER";

      return {
        ...prev,
        queue: updatedQueue,
        currentIndex: isComplete ? prev.currentIndex : nextIdx,
        skippedIds: newSkipped,
        currentState: nextState,
      };
    });
  }, [wizard, activeBall]);

  // ── Cancel wizard ─────────────────────────────────────────────────────────
  const handleCancelWizard = useCallback(() => {
    setWizard(null);
    setFoulResult(null);
  }, []);

  // ── Continue without restore ──────────────────────────────────────────────
  const handleContinueWithoutRestore = useCallback(() => {
    setShowRefereePanel(false);
    setWizard(null);
    setCurrentPlayer((prev) => (prev === "Player 1" ? "Player 2" : "Player 1"));
  }, []);

  // ── Restore & Replay (entire table to pre-shot) ───────────────────────────
  const handleRestoreAll = useCallback(() => {
    if (beforeState.length === 0) return;
    for (const ball of beforeState) {
      physics.restoreBall(ball.class_id, ball.x, ball.y);
    }
    setWizard(null);
    setShowRefereePanel(false);
    toast.success("Restored all balls to pre-shot coordinates!");
  }, [beforeState, physics]);

  const handleRefereeOverride = useCallback((decision: RefereeDecisionType) => {
    toast.success(`Referee decision: ${decision.toUpperCase()}`);
  }, []);

  // ── Undo / Redo ───────────────────────────────────────────────────────────
  const handleUndo = useCallback(() => {
    const undone = snapshotMgr.undoRestoration();
    if (undone) {
      setShotHistory(snapshotMgr.getHistory());
      toast.success(`Undone Shot #${undone.shotNumber}`);
    }
  }, [snapshotMgr]);

  const handleRedo = useCallback(() => {
    const redone = snapshotMgr.redoRestoration();
    if (redone) {
      setShotHistory(snapshotMgr.getHistory());
      toast.success(`Redone Shot #${redone.shotNumber}`);
    }
  }, [snapshotMgr]);

  // ── Demo Foul Scenario ────────────────────────────────────────────────────
  const handleDemoFoulScenario = useCallback(() => {
    captureBeforeState();
    physics.displaceBallForDemo(5, 490, 255); // Blue
    physics.displaceBallForDemo(1, 600, 260); // Red
    physics.displaceBallForDemo(6, 680, 270); // Pink

    const mockFoul: FoulResult = {
      isFoul: true,
      foulReason: "Demo Foul Scenario: Multi-Ball Displacement",
      penaltyPoints: 5,
      ballOn: 1,
      firstContactBall: null,
      pottedBalls: [],
      isFoulAndMiss: true,
      displacedBallsCount: 3,
      ruleSetName: SnookerRulesEngine.RULESET_NAME,
    };
    setFoulResult(mockFoul);
    setShowRefereePanel(true);
    toast.success("3-Ball Foul Demo Initialized — referee panel is open.");
  }, [captureBeforeState, physics]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto p-4 md:p-6 text-zinc-100 font-sans">
      {/* Header Bar */}
      <div className="p-5 rounded-2xl bg-zinc-950/80 border border-white/8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs text-red-400 font-bold uppercase tracking-wider">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            <span>AI-ASSISTED SNOOKER REFEREE</span>
            <span>•</span>
            <span>22 BALLS</span>
            <span>•</span>
            <span>ONE BALL AT A TIME</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white mt-1">
            Snooker Simulation & Referee Engine
          </h1>
          <p className="text-xs text-zinc-400 mt-1">
            Aim, shoot, evaluate fouls, and restore balls one-at-a-time with AI laser guidance.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleDemoFoulScenario}
            className="py-2.5 px-4 rounded-xl bg-amber-600 hover:bg-amber-700 font-bold text-xs text-white transition-all shadow-lg flex items-center gap-2"
          >
            <Sparkles size={15} /> Demo Foul Restoration
          </button>
          <button
            onClick={() => {
              physics.resetRack();
              setFoulResult(null);
              setShowRefereePanel(false);
              setWizard(null);
              toast.success("Reset 22-Ball Snooker Table Rack!");
            }}
            className="py-2.5 px-4 rounded-xl bg-zinc-900 border border-white/10 hover:bg-zinc-800 text-xs font-semibold text-zinc-300 transition-all flex items-center gap-2"
          >
            <RefreshCw size={14} /> Restart Game
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main 2D Canvas */}
        <div className="lg:col-span-2 space-y-4">
          <SnookerTable
            physics={physics}
            onShotStart={captureBeforeState}
            onShotFinish={handleShotFinish}
            activeDisplacedBall={activeBall}
            restorationState={wizard?.currentState ?? "IDLE"}
            onLaserReady={handleLaserReady}
            onRestoreAnimationComplete={handleRestoreAnimationComplete}
            currentPlayer={currentPlayer}
            scores={scores}
            ballOn={ballOn}
          />
        </div>

        {/* Referee Panels */}
        <div className="space-y-5">
          {/* Foul detected → show referee panel first */}
          {showRefereePanel && foulResult && (
            <RefereePanel
              foulResult={foulResult}
              onAutomaticRestore={handleAutomaticRestore}
              onManualRestore={handleManualRestore}
              onContinueWithoutRestore={handleContinueWithoutRestore}
              onRestorePreviousPosition={handleRestoreAll}
              onRefereeOverride={handleRefereeOverride}
            />
          )}

          {/* One-ball-at-a-time restoration wizard */}
          {wizard && !showRefereePanel && (
            <RestorationWizard
              wizardState={wizard}
              onModeChange={handleWizardModeChange}
              onRestoreThisBall={handleRestoreThisBall}
              onSkipBall={handleSkipBall}
              onCancel={handleCancelWizard}
              onSelectBall={handleSelectBall}
              onAnimationComplete={handleRestoreAnimationComplete}
            />
          )}

          <ShotHistoryDrawer
            history={shotHistory}
            onUndo={handleUndo}
            onRedo={handleRedo}
          />
        </div>
      </div>
    </div>
  );
}
