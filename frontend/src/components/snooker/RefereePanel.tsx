import React, { useState } from "react";
import { AlertOctagon, CheckCircle2, XCircle, RotateCcw, Play, RefreshCw, ShieldAlert, Edit3 } from "lucide-react";
import type { FoulResult, RefereeDecisionType } from "../../types/snooker";

interface Props {
  foulResult: FoulResult;
  onAutomaticRestore: () => void;
  onManualRestore: () => void;
  onContinueWithoutRestore: () => void;
  onRestorePreviousPosition: () => void;
  onRefereeOverride: (decision: RefereeDecisionType) => void;
}

export const RefereePanel: React.FC<Props> = ({
  foulResult,
  onAutomaticRestore,
  onManualRestore,
  onContinueWithoutRestore,
  onRestorePreviousPosition,
  onRefereeOverride,
}) => {
  const [refereeOverrideActive, setRefereeOverrideActive] = useState<boolean>(false);
  const [currentDecision, setCurrentDecision] = useState<RefereeDecisionType>("confirmed");

  const handleDecision = (decision: RefereeDecisionType) => {
    setCurrentDecision(decision);
    setRefereeOverrideActive(decision !== "confirmed");
    onRefereeOverride(decision);
  };

  return (
    <div className="bg-zinc-950/95 backdrop-blur-md border border-red-500/40 rounded-2xl p-5 shadow-2xl space-y-5 text-zinc-100 font-sans">
      {/* Foul Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-red-600/20 border border-red-500/40 flex items-center justify-center text-red-400">
            <AlertOctagon size={22} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-extrabold text-base text-red-400 tracking-wider uppercase font-mono">
                FOUL DETECTED
              </h3>
              {refereeOverrideActive && (
                <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 text-[9px] font-mono font-bold uppercase">
                  REFEREE OVERRIDE
                </span>
              )}
            </div>
            <p className="text-xs text-zinc-400">{foulResult.foulReason}</p>
          </div>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-zinc-500 block uppercase font-mono">Penalty</span>
          <span className="text-2xl font-black text-red-500 font-mono">+{foulResult.penaltyPoints}</span>
        </div>
      </div>

      {/* Shot Details Grid */}
      <div className="grid grid-cols-3 gap-3 font-mono text-xs">
        <div className="p-3 rounded-xl bg-zinc-900/70 border border-white/5">
          <span className="text-[10px] text-zinc-500 block">Ball ON</span>
          <span className="font-bold text-white uppercase">
            {foulResult.ballOn === 1 ? "Red (1)" : `Color (${foulResult.ballOn})`}
          </span>
        </div>
        <div className="p-3 rounded-xl bg-zinc-900/70 border border-white/5">
          <span className="text-[10px] text-zinc-500 block">First Contact</span>
          <span className="font-bold text-amber-400 capitalize">
            {foulResult.firstContactBall ? foulResult.firstContactBall.class_name.replace("_ball", "") : "None"}
          </span>
        </div>
        <div className="p-3 rounded-xl bg-zinc-900/70 border border-white/5">
          <span className="text-[10px] text-zinc-500 block">Rule Set</span>
          <span className="text-[10px] font-semibold text-emerald-400 block truncate">
            Simulation Rules
          </span>
        </div>
      </div>

      {/* Referee Decision Controls */}
      <div className="space-y-2 pt-1 border-t border-white/5">
        <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 block font-bold">
          Referee Authority & Override
        </span>
        <div className="flex flex-wrap gap-2 text-xs">
          <button
            onClick={() => handleDecision("confirmed")}
            className={`px-3 py-1.5 rounded-lg border font-mono transition-all flex items-center gap-1 ${
              currentDecision === "confirmed"
                ? "bg-red-600 text-white border-red-500 shadow-md font-bold"
                : "bg-zinc-900 text-zinc-400 border-white/10 hover:text-white"
            }`}
          >
            <CheckCircle2 size={13} />
            FOUL CONFIRMED
          </button>
          <button
            onClick={() => handleDecision("foul_and_miss")}
            className={`px-3 py-1.5 rounded-lg border font-mono transition-all flex items-center gap-1 ${
              currentDecision === "foul_and_miss"
                ? "bg-amber-600 text-white border-amber-500 shadow-md font-bold"
                : "bg-zinc-900 text-zinc-400 border-white/10 hover:text-white"
            }`}
          >
            <ShieldAlert size={13} />
            FOUL & MISS
          </button>
          <button
            onClick={() => handleDecision("rejected")}
            className={`px-3 py-1.5 rounded-lg border font-mono transition-all flex items-center gap-1 ${
              currentDecision === "rejected"
                ? "bg-zinc-700 text-white border-zinc-500 shadow-md font-bold"
                : "bg-zinc-900 text-zinc-400 border-white/10 hover:text-white"
            }`}
          >
            <XCircle size={13} />
            REJECT FOUL
          </button>
        </div>
      </div>

      {/* Restoration Action Buttons */}
      <div className="space-y-2.5 pt-2">
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={onAutomaticRestore}
            className="py-3 px-4 rounded-xl bg-red-600 hover:bg-red-700 font-bold text-xs text-white transition-all shadow-lg flex items-center justify-center gap-2"
          >
            <RotateCcw size={15} />
            AUTOMATIC RESTORE
          </button>
          <button
            onClick={onManualRestore}
            className="py-3 px-4 rounded-xl bg-amber-600 hover:bg-amber-700 font-bold text-xs text-white transition-all shadow-lg flex items-center justify-center gap-2"
          >
            <Edit3 size={15} />
            MANUAL RESTORE
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={onRestorePreviousPosition}
            className="py-2.5 px-3 rounded-xl bg-zinc-900 border border-white/10 hover:bg-zinc-800 text-xs font-semibold text-zinc-300 transition-all flex items-center justify-center gap-2"
          >
            <RefreshCw size={14} />
            RESTORE & REPLAY
          </button>
          <button
            onClick={onContinueWithoutRestore}
            className="py-2.5 px-3 rounded-xl bg-zinc-900 border border-white/10 hover:bg-zinc-800 text-xs font-semibold text-zinc-300 transition-all flex items-center justify-center gap-2"
          >
            <Play size={14} />
            PLAY FROM CURRENT
          </button>
        </div>
      </div>
    </div>
  );
};
