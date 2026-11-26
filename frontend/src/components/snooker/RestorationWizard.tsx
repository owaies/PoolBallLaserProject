import React, { useEffect } from "react";
import {
  Navigation,
  CheckCircle2,
  SkipForward,
  XCircle,
  Zap,
  MousePointerClick,
  ChevronRight,
  Target,
} from "lucide-react";
import type {
  DisplacedBall,
  RestorationMode,
  RestorationWizardState,
} from "../../types/snooker";

interface Props {
  wizardState: RestorationWizardState;
  onModeChange: (mode: RestorationMode) => void;
  /** Referee confirmed — move this ball to its original position */
  onRestoreThisBall: () => void;
  /** Skip the current ball (leave displaced, mark SKIPPED) */
  onSkipBall: () => void;
  /** Cancel entire restoration session */
  onCancel: () => void;
  /** Manual mode: referee picked a specific ball by id */
  onSelectBall: (id: string) => void;
  /** Called after animation completes — tells wizard to advance */
  onAnimationComplete: () => void;
}

function ballLabel(ball: DisplacedBall): string {
  return ball.label || ball.ball_class_name.replace("_ball", "").replace("_", " ").replace(/^\w/, (c) => c.toUpperCase());
}

function statusIcon(status: DisplacedBall["status"], isCurrent: boolean) {
  if (status === "RESTORED") return <CheckCircle2 size={14} className="text-emerald-400 flex-shrink-0" />;
  if (status === "SKIPPED") return <SkipForward size={14} className="text-zinc-500 flex-shrink-0" />;
  if (isCurrent) return <ChevronRight size={14} className="text-red-400 animate-pulse flex-shrink-0" />;
  return <span className="w-3.5 h-3.5 rounded-full border border-zinc-600 flex-shrink-0 inline-block" />;
}

export const RestorationWizard: React.FC<Props> = ({
  wizardState,
  onModeChange,
  onRestoreThisBall,
  onSkipBall,
  onCancel,
  onSelectBall,
  onAnimationComplete,
}) => {
  const { mode, currentState, queue, currentIndex, restoredIds, skippedIds } = wizardState;

  const currentBall = currentIndex < queue.length ? queue[currentIndex] : null;
  const isComplete = currentState === "RESTORATION_COMPLETE";
  const isManual = mode === "manual";
  const isAnimating = currentState === "RESTORING_BALL";

  // Auto-advance after BALL_RESTORED (signal from parent after animation completes)
  useEffect(() => {
    if (currentState === "BALL_RESTORED") {
      const t = setTimeout(() => onAnimationComplete(), 100);
      return () => clearTimeout(t);
    }
  }, [currentState, onAnimationComplete]);

  // ── RESTORATION COMPLETE SCREEN ────────────────────────────────────────────
  if (isComplete) {
    const restored = queue.filter((b) => restoredIds.includes(b.id));
    const skipped = queue.filter((b) => skippedIds.includes(b.id));
    return (
      <div className="bg-zinc-950/95 backdrop-blur-md border border-emerald-500/30 rounded-2xl p-5 shadow-2xl text-zinc-100 space-y-4 font-sans">
        <div className="text-center space-y-2 py-2">
          <CheckCircle2 className="mx-auto text-emerald-400" size={36} />
          <h3 className="font-extrabold text-emerald-400 text-base uppercase tracking-wider font-mono">
            ✓ RESTORATION COMPLETE
          </h3>
          <p className="text-xs text-zinc-400">All selected balls have been processed.</p>
        </div>

        <div className="space-y-1.5 text-xs font-mono">
          <span className="text-zinc-500 uppercase tracking-wider text-[10px] font-bold block">Balls Restored</span>
          {restored.map((b) => (
            <div key={b.id} className="flex items-center gap-2 text-emerald-400">
              <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: b.color }} />
              <span className="capitalize">{ballLabel(b)}</span>
              <span className="ml-auto">✓</span>
            </div>
          ))}
          {skipped.length > 0 && (
            <>
              <span className="text-zinc-500 uppercase tracking-wider text-[10px] font-bold block mt-2">Skipped</span>
              {skipped.map((b) => (
                <div key={b.id} className="flex items-center gap-2 text-zinc-500">
                  <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: b.color }} />
                  <span className="capitalize">{ballLabel(b)}</span>
                  <span className="ml-auto text-zinc-600">SKIPPED</span>
                </div>
              ))}
            </>
          )}
        </div>

        <button
          onClick={onCancel}
          className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-700 font-bold text-xs text-white transition-all"
        >
          CONTINUE GAME
        </button>
      </div>
    );
  }

  return (
    <div className="bg-zinc-950/95 backdrop-blur-md border border-red-500/30 rounded-2xl p-5 shadow-2xl text-zinc-100 space-y-4 font-sans">

      {/* ── HEADER ── */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div>
          <h3 className="font-extrabold text-sm text-red-400 tracking-wider uppercase font-mono">
            BALL RESTORATION
          </h3>
          <p className="text-[11px] text-zinc-400 mt-0.5">
            {queue.filter((b) => b.status === "PENDING").length} ball
            {queue.filter((b) => b.status === "PENDING").length !== 1 ? "s" : ""} requiring restoration
          </p>
        </div>
        <button onClick={onCancel} className="text-zinc-600 hover:text-zinc-300 transition-colors" title="Cancel Restoration">
          <XCircle size={18} />
        </button>
      </div>

      {/* ── MODE TOGGLE ── */}
      <div className="flex gap-2">
        <button
          onClick={() => onModeChange("manual")}
          className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold font-mono transition-all border ${
            isManual
              ? "bg-amber-600 text-white border-amber-500"
              : "bg-zinc-900 text-zinc-400 border-white/10 hover:text-white"
          }`}
        >
          <MousePointerClick size={11} className="inline mr-1" />
          MANUAL
        </button>
        <button
          onClick={() => onModeChange("automatic")}
          className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold font-mono transition-all border ${
            !isManual
              ? "bg-red-600 text-white border-red-500"
              : "bg-zinc-900 text-zinc-400 border-white/10 hover:text-white"
          }`}
        >
          <Zap size={11} className="inline mr-1" />
          AUTOMATIC
        </button>
      </div>

      {/* ── AUTO MODE DISCLAIMER ── */}
      {!isManual && (
        <div className="px-3 py-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-[10px] text-blue-300 font-mono">
          <strong>SYSTEM RESTORATION PRIORITY:</strong> Sorted by distance → confidence → class hierarchy. Not an official Snooker rule.
        </div>
      )}

      {/* ── RESTORATION QUEUE ── */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 font-bold block">Restoration Queue</span>
        <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
          {queue.map((ball, idx) => {
            const isCurrent = idx === currentIndex;
            const isClickable = isManual && ball.status === "PENDING" && !isAnimating;
            return (
              <div
                key={ball.id}
                onClick={() => isClickable ? onSelectBall(ball.id) : undefined}
                className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-mono transition-all ${
                  isCurrent && ball.status === "PENDING"
                    ? "bg-red-600/15 border border-red-500/40 text-white"
                    : ball.status === "RESTORED"
                    ? "bg-zinc-900/40 border border-white/5 text-emerald-500 line-through opacity-70"
                    : ball.status === "SKIPPED"
                    ? "bg-zinc-900/40 border border-white/5 text-zinc-600 opacity-50"
                    : isClickable
                    ? "bg-zinc-900/60 border border-white/5 text-zinc-300 hover:border-amber-500/40 cursor-pointer"
                    : "bg-zinc-900/60 border border-white/5 text-zinc-400"
                }`}
              >
                {statusIcon(ball.status, isCurrent)}
                <div className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: ball.color }} />
                <span className="flex-1 capitalize font-semibold">{ballLabel(ball)}</span>
                <span className={ball.status === "PENDING" ? "text-amber-400" : "text-zinc-600"}>
                  {ball.displacement_mm.toFixed(1)} mm
                </span>
                {isManual && ball.status === "PENDING" && !isCurrent && (
                  <span className="text-[9px] text-zinc-600 uppercase">click</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── MANUAL: WAITING FOR SELECTION ── */}
      {isManual && currentState === "SELECT_BALL" && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 font-mono flex items-center gap-2">
          <MousePointerClick size={14} className="animate-bounce" />
          <span>Select a ball from the queue above to begin restoration</span>
        </div>
      )}

      {/* ── CURRENT BALL DETAILS ── */}
      {currentBall && currentBall.status === "PENDING" && currentState !== "SELECT_BALL" && (
        <div className="space-y-3">
          {/* Ball Header */}
          <div className="p-3 rounded-xl bg-zinc-900/80 border border-white/8 space-y-2">
            <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 font-bold block">Current Ball</span>
            <div className="flex items-center gap-3">
              <div
                className="w-8 h-8 rounded-full border-2 border-white/20 shadow-lg flex-shrink-0"
                style={{
                  backgroundColor: currentBall.color,
                  boxShadow: `0 0 12px ${currentBall.color}66`,
                }}
              />
              <div>
                <span className="font-extrabold text-sm capitalize text-white uppercase font-mono">
                  {ballLabel(currentBall)}
                </span>
                <div className="text-[10px] text-amber-400 font-mono">
                  {currentState === "AWAITING_RESTORE_CONFIRMATION" || currentState === "SHOW_LASER"
                    ? "AWAITING RESTORATION"
                    : currentState === "RESTORING_BALL"
                    ? "RESTORING…"
                    : currentState === "BALL_RESTORED"
                    ? "✓ RESTORED"
                    : ""}
                </div>
              </div>
            </div>

            {/* Coordinate Grid */}
            <div className="grid grid-cols-2 gap-2 text-[10px] font-mono pt-1 border-t border-white/5 mt-1">
              <div>
                <span className="text-zinc-500 block">Current Position</span>
                <span className="text-zinc-200">
                  X = {currentBall.current_pos.x.toFixed(0)} &nbsp; Y = {currentBall.current_pos.y.toFixed(0)}
                </span>
              </div>
              <div>
                <span className="text-zinc-500 block">Original Position</span>
                <span className="text-emerald-400">
                  X = {currentBall.original_pos.x.toFixed(0)} &nbsp; Y = {currentBall.original_pos.y.toFixed(0)}
                </span>
              </div>
              <div>
                <span className="text-zinc-500 block">Displacement</span>
                <span className="text-amber-400">{currentBall.displacement_mm.toFixed(1)} mm</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Laser Angle</span>
                <span className="text-blue-400">{currentBall.laser_angle_deg.toFixed(1)}°</span>
              </div>
            </div>
          </div>

          {/* AI Virtual Laser Banner */}
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/25 text-[11px] text-red-300 flex items-center gap-2 font-mono">
            <Navigation className="animate-pulse text-red-400 flex-shrink-0" size={15} />
            <div>
              <strong>AI-GUIDED VIRTUAL LASER</strong>
              <div className="text-zinc-400 text-[10px]">
                From ({currentBall.current_pos.x.toFixed(0)}, {currentBall.current_pos.y.toFixed(0)}) →{" "}
                ({currentBall.original_pos.x.toFixed(0)}, {currentBall.original_pos.y.toFixed(0)}) &nbsp;|&nbsp;
                {currentBall.displacement_mm.toFixed(1)} mm &nbsp;|&nbsp; {currentBall.laser_angle_deg.toFixed(1)}°
              </div>
              <div className="text-[9px] text-zinc-600 mt-0.5">
                Browser laser is visualization only — does not directly control hardware.
              </div>
            </div>
          </div>

          {/* Queue Progress */}
          <div className="flex gap-1 items-center text-[10px] text-zinc-500 font-mono">
            <Target size={11} className="text-zinc-600" />
            <span>
              Ball {currentIndex + 1} of {queue.length} &nbsp;•&nbsp;{" "}
              {restoredIds.length} restored &nbsp;•&nbsp; {skippedIds.length} skipped
            </span>
          </div>

          {/* Action Buttons */}
          <div className="space-y-2">
            <button
              onClick={onRestoreThisBall}
              disabled={isAnimating || currentState === "SHOW_LASER"}
              className="w-full py-3 px-4 rounded-xl bg-red-600 hover:bg-red-700 disabled:opacity-40 disabled:cursor-not-allowed font-extrabold text-sm text-white transition-all shadow-lg flex items-center justify-center gap-2 font-mono uppercase tracking-wide"
            >
              <CheckCircle2 size={16} />
              {isAnimating ? "RESTORING…" : "RESTORE THIS BALL"}
            </button>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={onSkipBall}
                disabled={isAnimating}
                className="py-2 px-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-xs font-semibold text-zinc-300 transition-all flex items-center justify-center gap-1.5 font-mono"
              >
                <SkipForward size={13} />
                SKIP BALL
              </button>
              <button
                onClick={onCancel}
                disabled={isAnimating}
                className="py-2 px-3 rounded-xl bg-zinc-900 border border-white/10 hover:bg-zinc-800 disabled:opacity-40 text-xs font-semibold text-zinc-400 transition-all flex items-center justify-center gap-1.5 font-mono"
              >
                <XCircle size={13} />
                CANCEL
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
