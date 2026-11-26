import React, { useRef, useEffect, useState, useCallback } from "react";
import { SnookerPhysics } from "../../services/snooker/snookerPhysics";
import type { DisplacedBall, RestorationState } from "../../types/snooker";

interface Props {
  physics: SnookerPhysics;
  /** Called just before striking cue ball — captures pre-shot state */
  onShotStart?: () => void;
  onShotFinish?: () => void;
  activeDisplacedBall?: DisplacedBall | null;
  /** Current restoration wizard state, drives laser / animation rendering */
  restorationState?: RestorationState;
  /** Canvas notifies parent that laser is drawn (SHOW_LASER → AWAITING) */
  onLaserReady?: () => void;
  /** Canvas notifies parent that ball animation to target is complete */
  onRestoreAnimationComplete?: () => void;
  currentPlayer?: string;
  scores?: { player1: number; player2: number };
  ballOn?: number;
}

export const SnookerTable: React.FC<Props> = ({
  physics,
  onShotStart,
  onShotFinish,
  activeDisplacedBall,
  restorationState = "IDLE",
  onLaserReady,
  onRestoreAnimationComplete,
  currentPlayer = "Player 1",
  scores = { player1: 0, player2: 0 },
  ballOn = 1,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [aimAngle, setAimAngle] = useState<number>(0);
  const [power, setPower] = useState<number>(45);
  const [isStriking, setIsStriking] = useState<boolean>(false);

  // Animation state for smooth ball restoration
  const restoreAnimRef = useRef<{
    active: boolean;
    startX: number;
    startY: number;
    endX: number;
    endY: number;
    startTime: number;
    duration: number; // ms
    laserFadeStart: number;
    laserOpacity: number;
  } | null>(null);

  // Track whether we've fired onLaserReady for this ball
  const laserReadyFiredRef = useRef<string | null>(null);

  // Start restore animation when state transitions to RESTORING_BALL
  useEffect(() => {
    if (
      restorationState === "RESTORING_BALL" &&
      activeDisplacedBall &&
      !restoreAnimRef.current?.active
    ) {
      restoreAnimRef.current = {
        active: true,
        startX: activeDisplacedBall.current_pos.x,
        startY: activeDisplacedBall.current_pos.y,
        endX: activeDisplacedBall.original_pos.x,
        endY: activeDisplacedBall.original_pos.y,
        startTime: performance.now(),
        duration: 600,
        laserFadeStart: 0,
        laserOpacity: 1,
      };
    }
    if (restorationState === "IDLE" || restorationState === "RESTORATION_COMPLETE") {
      restoreAnimRef.current = null;
      laserReadyFiredRef.current = null;
    }
  }, [restorationState, activeDisplacedBall]);

  const easeInOut = (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

  // 60 FPS Canvas Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;

    const render = (timestamp: number) => {
      const w = canvas.width;
      const h = canvas.height;
      const padding = w * 0.08;

      ctx.clearRect(0, 0, w, h);

      // ── 1. Mahogany wood frame ────────────────────────────────────────────
      const frameGrad = ctx.createLinearGradient(0, 0, w, h);
      frameGrad.addColorStop(0, "#3d2314");
      frameGrad.addColorStop(0.35, "#2b1509");
      frameGrad.addColorStop(0.5, "#1a0d06");
      frameGrad.addColorStop(0.65, "#2b1509");
      frameGrad.addColorStop(1, "#3d2314");
      ctx.fillStyle = frameGrad;
      ctx.fillRect(0, 0, w, h);

      // Grain texture lines on rails
      ctx.save();
      ctx.globalAlpha = 0.06;
      for (let i = 0; i < 20; i++) {
        const xPos = (i * w) / 20;
        ctx.beginPath();
        ctx.moveTo(xPos, 0);
        ctx.lineTo(xPos + 10, h);
        ctx.strokeStyle = "#c08040";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.restore();

      // Cushion inner shadow
      ctx.fillStyle = "#163a1e";
      ctx.fillRect(padding * 0.65, padding * 0.65, w - padding * 1.3, h - padding * 1.3);

      // Inner cushion bevel (darker)
      ctx.fillStyle = "#0d2914";
      ctx.fillRect(padding * 0.8, padding * 0.8, w - padding * 1.6, h - padding * 1.6);

      // ── 2. Felt surface ───────────────────────────────────────────────────
      const feltGrad = ctx.createRadialGradient(w / 2, h / 2, 40, w / 2, h / 2, w * 0.65);
      feltGrad.addColorStop(0, "#0f6030");
      feltGrad.addColorStop(0.7, "#0b4a22");
      feltGrad.addColorStop(1, "#073315");
      ctx.fillStyle = feltGrad;
      ctx.fillRect(padding, padding, w - 2 * padding, h - 2 * padding);

      // Subtle cloth weave texture
      ctx.save();
      ctx.globalAlpha = 0.025;
      ctx.fillStyle = "#ffffff";
      for (let ix = padding; ix < w - padding; ix += 4) {
        for (let iy = padding; iy < h - padding; iy += 4) {
          if ((ix + iy) % 8 === 0) {
            ctx.fillRect(ix, iy, 1, 1);
          }
        }
      }
      ctx.restore();

      // ── 3. Table markings ─────────────────────────────────────────────────
      const playW = w - 2 * padding;
      const playH = h - 2 * padding;
      const cY = h / 2;
      const baulkX = padding + playW * 0.2;
      const baulkRadius = playH * 0.26;

      ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
      ctx.lineWidth = 1.5;

      // Baulk line
      ctx.beginPath();
      ctx.moveTo(baulkX, padding);
      ctx.lineTo(baulkX, h - padding);
      ctx.stroke();

      // D semi-circle
      ctx.beginPath();
      ctx.arc(baulkX, cY, baulkRadius, Math.PI * 0.5, Math.PI * 1.5);
      ctx.stroke();

      // Middle line (faint)
      ctx.save();
      ctx.globalAlpha = 0.08;
      ctx.beginPath();
      ctx.moveTo(w / 2, padding);
      ctx.lineTo(w / 2, h - padding);
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();

      // Spot markers
      const pinkX = padding + playW * 0.72;
      const blackX = padding + playW * 0.9;
      const spots = [
        { x: baulkX, y: cY },
        { x: baulkX, y: cY - baulkRadius },
        { x: baulkX, y: cY + baulkRadius },
        { x: w / 2, y: cY },
        { x: pinkX, y: cY },
        { x: blackX, y: cY },
      ];
      ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
      for (const s of spots) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // ── 4. Pockets ────────────────────────────────────────────────────────
      const pockets = [
        { x: padding, y: padding },
        { x: w / 2, y: padding - 2 },
        { x: w - padding, y: padding },
        { x: padding, y: h - padding },
        { x: w / 2, y: h - padding + 2 },
        { x: w - padding, y: h - padding },
      ];
      const pRad = physics.ballRadius * 1.75;

      for (const p of pockets) {
        // Outer rim shadow glow
        const pocketGlow = ctx.createRadialGradient(p.x, p.y, pRad, p.x, p.y, pRad + 8);
        pocketGlow.addColorStop(0, "rgba(0,0,0,0.6)");
        pocketGlow.addColorStop(1, "rgba(0,0,0,0)");
        ctx.beginPath();
        ctx.arc(p.x, p.y, pRad + 8, 0, Math.PI * 2);
        ctx.fillStyle = pocketGlow;
        ctx.fill();

        // Brass bezel ring
        ctx.beginPath();
        ctx.arc(p.x, p.y, pRad + 3.5, 0, Math.PI * 2);
        const brassGrad = ctx.createRadialGradient(p.x - 2, p.y - 2, 0, p.x, p.y, pRad + 4);
        brassGrad.addColorStop(0, "#d97706");
        brassGrad.addColorStop(1, "#78350f");
        ctx.fillStyle = brassGrad;
        ctx.fill();

        // Black hole
        ctx.beginPath();
        ctx.arc(p.x, p.y, pRad, 0, Math.PI * 2);
        const holeGrad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, pRad);
        holeGrad.addColorStop(0, "#111111");
        holeGrad.addColorStop(1, "#000000");
        ctx.fillStyle = holeGrad;
        ctx.fill();
      }

      // ── 5. Physics update ─────────────────────────────────────────────────
      const wasMoving = physics.isMoving();
      physics.update(w, h);
      const isNowMoving = physics.isMoving();

      if (wasMoving && !isNowMoving && onShotFinish) {
        onShotFinish();
      }

      // ── 6. Glossy balls ───────────────────────────────────────────────────
      for (const b of physics.balls) {
        if (b.isPocketed) continue;

        // Is this ball being animated toward its original position?
        let drawX = b.x;
        let drawY = b.y;
        const anim = restoreAnimRef.current;
        const isRestoringBall =
          anim?.active &&
          activeDisplacedBall &&
          b.class_id === activeDisplacedBall.ball_class_id;

        if (isRestoringBall && anim) {
          const elapsed = timestamp - anim.startTime;
          const t = Math.min(elapsed / anim.duration, 1);
          const ease = easeInOut(t);
          drawX = anim.startX + (anim.endX - anim.startX) * ease;
          drawY = anim.startY + (anim.endY - anim.startY) * ease;

          // Animation finished
          if (t >= 1 && onRestoreAnimationComplete) {
            restoreAnimRef.current = { ...anim, active: false, laserFadeStart: timestamp };
            onRestoreAnimationComplete();
          }
        }

        // Drop shadow
        ctx.beginPath();
        ctx.arc(drawX + 3, drawY + 4, b.radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
        ctx.fill();

        // 3D glossy sphere — multi-stop radial
        const grad = ctx.createRadialGradient(
          drawX - b.radius * 0.38,
          drawY - b.radius * 0.38,
          b.radius * 0.04,
          drawX,
          drawY,
          b.radius
        );
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.2, lighten(b.color, 0.4));
        grad.addColorStop(0.55, b.color);
        grad.addColorStop(1, darken(b.color, 0.6));

        ctx.beginPath();
        ctx.arc(drawX, drawY, b.radius, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        // Edge rim
        ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
        ctx.lineWidth = 0.8;
        ctx.stroke();

        // Secondary specular highlight (small bright dot)
        const specGrad = ctx.createRadialGradient(
          drawX - b.radius * 0.45, drawY - b.radius * 0.45, 0,
          drawX - b.radius * 0.45, drawY - b.radius * 0.45, b.radius * 0.35
        );
        specGrad.addColorStop(0, "rgba(255,255,255,0.75)");
        specGrad.addColorStop(1, "rgba(255,255,255,0)");
        ctx.beginPath();
        ctx.arc(drawX, drawY, b.radius, 0, Math.PI * 2);
        ctx.fillStyle = specGrad;
        ctx.fill();
      }

      // ── 7. Laser beam + target marker ─────────────────────────────────────
      const showLaser =
        activeDisplacedBall &&
        (restorationState === "SHOW_LASER" ||
          restorationState === "AWAITING_RESTORE_CONFIRMATION" ||
          restorationState === "RESTORING_BALL" ||
          restorationState === "BALL_RESTORED");

      if (showLaser && activeDisplacedBall) {
        const origX = activeDisplacedBall.original_pos.x;
        const origY = activeDisplacedBall.original_pos.y;
        const currX = activeDisplacedBall.current_pos.x;
        const currY = activeDisplacedBall.current_pos.y;

        // Determine laser opacity (fades after BALL_RESTORED)
        let laserAlpha = 1;
        const anim = restoreAnimRef.current;
        if (restorationState === "BALL_RESTORED" && anim && anim.laserFadeStart > 0) {
          const fadeDuration = 400;
          const fadeProg = Math.min((timestamp - anim.laserFadeStart) / fadeDuration, 1);
          laserAlpha = 1 - fadeProg;
        }

        ctx.save();
        ctx.globalAlpha = laserAlpha;

        // Target ring — pulsing
        const pulseScale = 1 + Math.sin(timestamp * 0.006) * 0.12;
        ctx.beginPath();
        ctx.arc(origX, origY, (physics.ballRadius + 5) * pulseScale, 0, Math.PI * 2);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Target crosshair
        const ch = 8;
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(origX - ch, origY); ctx.lineTo(origX + ch, origY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(origX, origY - ch); ctx.lineTo(origX, origY + ch); ctx.stroke();

        ctx.beginPath();
        ctx.arc(origX, origY, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "#38bdf8";
        ctx.fill();

        // Laser beam with glow
        const laserGlow = 2 + Math.sin(timestamp * 0.008) * 1.2;
        ctx.beginPath();
        ctx.moveTo(currX, currY);
        ctx.lineTo(origX, origY);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = laserGlow;
        ctx.shadowColor = "#ef4444";
        ctx.shadowBlur = 18;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Arrow head
        const angle = Math.atan2(origY - currY, origX - currX);
        const headLen = 13;
        ctx.beginPath();
        ctx.moveTo(origX, origY);
        ctx.lineTo(origX - headLen * Math.cos(angle - Math.PI / 5.5), origY - headLen * Math.sin(angle - Math.PI / 5.5));
        ctx.lineTo(origX - headLen * Math.cos(angle + Math.PI / 5.5), origY - headLen * Math.sin(angle + Math.PI / 5.5));
        ctx.fillStyle = "#ef4444";
        ctx.closePath();
        ctx.fill();

        // "ORIGINAL POSITION" label
        ctx.font = "bold 10px monospace";
        ctx.fillStyle = "rgba(56,189,248,0.9)";
        ctx.fillText("⭕ ORIGINAL", origX + 12, origY - 8);

        // "CURRENT" label
        ctx.fillStyle = "rgba(239,68,68,0.85)";
        ctx.fillText("● CURRENT", currX + 10, currY - 8);

        ctx.restore();

        // Fire onLaserReady once per ball
        if (
          restorationState === "SHOW_LASER" &&
          onLaserReady &&
          laserReadyFiredRef.current !== activeDisplacedBall.id
        ) {
          laserReadyFiredRef.current = activeDisplacedBall.id;
          // Defer so render loop doesn't re-trigger immediately
          setTimeout(onLaserReady, 100);
        }
      }

      // ── 8. Cue stick + aim guide (only when table is idle) ───────────────
      const cueBall = physics.balls.find((b) => b.class_id === 0 && !b.isPocketed);
      if (cueBall && !physics.isMoving() && !activeDisplacedBall) {
        const rad = (aimAngle * Math.PI) / 180;
        const rayLen = 240;
        const targetX = cueBall.x + Math.cos(rad) * rayLen;
        const targetY = cueBall.y + Math.sin(rad) * rayLen;

        // Trajectory guide
        ctx.beginPath();
        ctx.moveTo(cueBall.x, cueBall.y);
        ctx.lineTo(targetX, targetY);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Ghost target ball
        ctx.beginPath();
        ctx.arc(targetX, targetY, physics.ballRadius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Tapered cue stick (thick at back, thin at tip)
        const cueLength = 180;
        const pullbackOffset = isStriking ? 4 : (power / 100) * 28 + 14;
        const tipX = cueBall.x - Math.cos(rad) * pullbackOffset;
        const tipY = cueBall.y - Math.sin(rad) * pullbackOffset;
        const buttX = tipX - Math.cos(rad) * cueLength;
        const buttY = tipY - Math.sin(rad) * cueLength;

        // Cue shaft (maple wood gradient)
        const cueGrad = ctx.createLinearGradient(tipX, tipY, buttX, buttY);
        cueGrad.addColorStop(0, "#f0d090");
        cueGrad.addColorStop(0.4, "#d97706");
        cueGrad.addColorStop(0.8, "#92400e");
        cueGrad.addColorStop(1, "#3b1f08");

        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(buttX, buttY);
        ctx.strokeStyle = cueGrad as unknown as string;
        ctx.lineWidth = 5;
        ctx.lineCap = "round";
        ctx.stroke();

        // Cue tip (lighter)
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - Math.cos(rad) * 10, tipY - Math.sin(rad) * 10);
        ctx.strokeStyle = "#93c5fd";
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [physics, aimAngle, activeDisplacedBall, restorationState, onShotFinish, onLaserReady, onRestoreAnimationComplete, power, isStriking]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (physics.isMoving() || activeDisplacedBall) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const cueBall = physics.balls.find((b) => b.class_id === 0 && !b.isPocketed);
    if (!cueBall) return;
    setAimAngle((Math.atan2(mouseY - cueBall.y, mouseX - cueBall.x) * 180) / Math.PI);
  }, [physics, activeDisplacedBall]);

  const handleShoot = useCallback(() => {
    if (physics.isMoving() || activeDisplacedBall) return;
    if (onShotStart) onShotStart();
    setIsStriking(true);
    setTimeout(() => {
      physics.strikeCueBall(aimAngle, power);
      setIsStriking(false);
    }, 150);
  }, [physics, activeDisplacedBall, aimAngle, power, onShotStart]);

  return (
    <div className="space-y-4 font-sans select-none">
      {/* HUD Bar */}
      <div className="bg-zinc-950/80 backdrop-blur-md p-3 rounded-2xl border border-white/8 flex items-center justify-between font-mono text-xs text-zinc-300">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-zinc-400">Turn:</span>
            <strong className="text-white">{currentPlayer}</strong>
          </div>
          <div>
            <span className="text-zinc-400">Scores: </span>
            <span className="text-blue-400 font-bold">P1 {scores.player1}</span>
            <span className="text-zinc-600 px-1.5">—</span>
            <span className="text-amber-400 font-bold">P2 {scores.player2}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-zinc-400">Ball ON:</span>
          <span className="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 font-bold text-[10px] uppercase">
            {ballOn === 1 ? "RED" : `COLOR (${ballOn})`}
          </span>
          {activeDisplacedBall && (
            <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/25 font-bold text-[10px] animate-pulse">
              RESTORATION ACTIVE
            </span>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div className="relative rounded-2xl overflow-hidden shadow-2xl border border-white/8 bg-zinc-950">
        <canvas
          ref={canvasRef}
          width={900}
          height={480}
          onMouseMove={handleMouseMove}
          className="w-full h-auto cursor-crosshair block"
        />

        {/* Aim & Power overlay */}
        {!physics.isMoving() && !activeDisplacedBall && (
          <div className="absolute bottom-3 left-3 right-3 bg-zinc-950/88 backdrop-blur-md p-3 rounded-xl border border-white/8 flex items-center justify-between font-mono text-xs text-zinc-300">
            <div className="flex items-center gap-5">
              <span>Angle: <strong className="text-blue-400">{aimAngle.toFixed(0)}°</strong></span>
              <div className="flex items-center gap-2">
                <span>Power:</span>
                <input
                  type="range"
                  min={10}
                  max={100}
                  value={power}
                  onChange={(e) => setPower(Number(e.target.value))}
                  className="w-24 accent-red-500 cursor-pointer"
                />
                <strong className="text-emerald-400">{power}%</strong>
              </div>
            </div>
            <button
              onClick={handleShoot}
              className="px-5 py-2 rounded-xl bg-red-600 hover:bg-red-700 font-bold text-white text-xs transition-all shadow-lg"
            >
              STRIKE CUE BALL
            </button>
          </div>
        )}

        {/* Restoration active overlay hint */}
        {activeDisplacedBall && restorationState === "AWAITING_RESTORE_CONFIRMATION" && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-red-900/80 backdrop-blur-md px-4 py-1.5 rounded-full border border-red-500/40 text-red-300 text-[11px] font-mono font-bold flex items-center gap-2 animate-pulse">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            AI LASER TARGETING — AWAITING REFEREE CONFIRMATION
          </div>
        )}
      </div>
    </div>
  );
};

// ── Color helpers ────────────────────────────────────────────────────────────
function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  if (clean.length !== 6) return [128, 128, 128];
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16),
  ];
}

function lighten(hex: string, amount: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgb(${Math.min(255, r + (255 - r) * amount)},${Math.min(255, g + (255 - g) * amount)},${Math.min(255, b + (255 - b) * amount)})`;
}

function darken(hex: string, amount: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgb(${Math.max(0, r * (1 - amount))},${Math.max(0, g * (1 - amount))},${Math.max(0, b * (1 - amount))})`;
}
