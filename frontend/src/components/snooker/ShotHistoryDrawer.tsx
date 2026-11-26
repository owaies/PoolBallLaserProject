import React, { useState } from "react";
import { History, Undo2, Redo2, ChevronDown, ChevronUp, AlertTriangle, CheckCircle2 } from "lucide-react";
import type { ShotRecord } from "../../types/snooker";

interface Props {
  history: ShotRecord[];
  onUndo: () => void;
  onRedo: () => void;
}

export const ShotHistoryDrawer: React.FC<Props> = ({ history, onUndo, onRedo }) => {
  const [expandedShotNumber, setExpandedShotNumber] = useState<number | null>(null);

  const toggleExpand = (shotNum: number) => {
    setExpandedShotNumber((prev) => (prev === shotNum ? null : shotNum));
  };

  return (
    <div className="bg-zinc-950/80 backdrop-blur-md border border-white/10 rounded-2xl p-5 shadow-xl space-y-4 text-zinc-200 font-sans">
      <div className="flex items-center justify-between border-b border-white/10 pb-3 font-mono">
        <div className="flex items-center gap-2">
          <History size={18} className="text-blue-400" />
          <h3 className="font-bold text-sm text-white uppercase">Shot History Log</h3>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onUndo}
            disabled={history.length === 0}
            className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-white/10 hover:bg-zinc-800 disabled:opacity-40 text-xs font-semibold text-zinc-300 transition-all flex items-center gap-1"
          >
            <Undo2 size={12} /> Undo
          </button>
          <button
            onClick={onRedo}
            className="px-2.5 py-1 rounded-lg bg-zinc-900 border border-white/10 hover:bg-zinc-800 text-xs font-semibold text-zinc-300 transition-all flex items-center gap-1"
          >
            <Redo2 size={12} /> Redo
          </button>
        </div>
      </div>

      {history.length === 0 ? (
        <p className="text-xs text-zinc-500 font-mono text-center py-4">No shot history recorded yet.</p>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {history.map((shot) => {
            const isExpanded = expandedShotNumber === shot.shotNumber;
            return (
              <div
                key={shot.shotNumber}
                className={`rounded-xl border transition-all ${
                  shot.foul
                    ? "bg-red-950/20 border-red-500/30"
                    : "bg-zinc-900/60 border-white/5"
                }`}
              >
                <div
                  onClick={() => toggleExpand(shot.shotNumber)}
                  className="p-3 flex items-center justify-between cursor-pointer font-mono text-xs"
                >
                  <div className="flex items-center gap-3">
                    {shot.foul ? (
                      <AlertTriangle size={15} className="text-red-400" />
                    ) : (
                      <CheckCircle2 size={15} className="text-emerald-400" />
                    )}
                    <span className="font-bold text-white">Shot #{shot.shotNumber}</span>
                    <span className="text-zinc-400">({shot.player})</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      shot.foul ? "bg-red-500/20 text-red-400" : "bg-emerald-500/20 text-emerald-400"
                    }`}>
                      {shot.foul ? `Foul (+${shot.penalty})` : "Legal"}
                    </span>
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </div>
                </div>

                {isExpanded && (
                  <div className="p-3 pt-0 border-t border-white/5 font-mono text-[11px] text-zinc-400 space-y-1">
                    <div>Foul Type: <strong className="text-zinc-200">{shot.foulType}</strong></div>
                    <div>Restoration Method: <strong className="text-amber-400 capitalize">{shot.restorationMethod}</strong></div>
                    <div>Referee Decision: <strong className="text-emerald-400 capitalize">{shot.refereeDecision}</strong></div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
