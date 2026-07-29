import { useState, useEffect } from "react";
import { Compass, CheckCircle2, AlertTriangle, Cpu } from "lucide-react";
import { apiService } from "../services/api";
import type { CalibrationResponse } from "../types";

export default function Calibration() {
  const [calibration, setCalibration] = useState<CalibrationResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCalibration = async () => {
      try {
        const data = await apiService.getCalibration();
        setCalibration(data);
      } catch (err) {
        console.error("Failed to load calibration status", err);
      } finally {
        setLoading(false);
      }
    };

    fetchCalibration();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-zinc-500 text-xs font-mono tracking-widest uppercase">Fetching calibration...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      <div className="grid md:grid-cols-3 gap-6">
        
        {/* Status indicator Card */}
        <div className="md:col-span-1 p-6 rounded-3xl glass border border-white/5 space-y-6 h-fit">
          <div className="flex items-center gap-2 border-b border-white/5 pb-4">
            <Compass className="text-blue-500" size={18} />
            <h3 className="font-semibold text-white text-sm">Calibration Overview</h3>
          </div>

          <div className="space-y-4">
            <div>
              <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block mb-1">
                Calibration Status
              </span>
              {calibration?.is_calibrated ? (
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 font-semibold text-xs border border-emerald-500/20">
                  <CheckCircle2 size={14} />
                  Calibrated
                </div>
              ) : (
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-rose-500/10 text-rose-400 font-semibold text-xs border border-rose-500/20">
                  <AlertTriangle size={14} />
                  Uncalibrated (Fallback Active)
                </div>
              )}
            </div>

            <div className="space-y-2 text-xs text-zinc-400">
              <p>
                Calibration matrices allow the software pipeline to mathematically undistort image lenses so that targets align perfectly.
              </p>
              <p>
                If no calibration parameters exist, coordinates bypass lens correction and apply directly.
              </p>
            </div>
          </div>
        </div>

        {/* Matrix values table */}
        <div className="md:col-span-2 p-6 rounded-3xl glass border border-white/5 space-y-6">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Cpu size={16} className="text-blue-500" />
            Intrinsic Camera Parameters
          </h3>

          <div className="space-y-6">
            
            {/* Camera Matrix K */}
            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Camera Matrix (K) 3x3
              </span>
              {calibration?.camera_matrix ? (
                <div className="p-4 rounded-xl bg-zinc-950/40 border border-white/5 overflow-x-auto">
                  <table className="w-full text-center border-collapse font-mono text-xs text-white">
                    <tbody>
                      {calibration.camera_matrix.map((row, rIdx) => (
                        <tr key={rIdx} className="divide-x divide-white/5 border-b border-white/5 last:border-b-0">
                          {row.map((val, cIdx) => (
                            <td key={cIdx} className="py-2.5 px-4">
                              {val.toFixed(5)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs text-zinc-600 font-mono py-2">
                  No Camera Matrix loaded.
                </div>
              )}
            </div>

            {/* Distortion Coefficients D */}
            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Distortion Coefficients (D) [k1, k2, p1, p2, k3]
              </span>
              {calibration?.distortion_coefficients ? (
                <div className="p-4 rounded-xl bg-zinc-950/40 border border-white/5 font-mono text-xs text-white overflow-x-auto">
                  <div className="grid grid-cols-5 gap-2 text-center divide-x divide-white/5">
                    {calibration.distortion_coefficients.map((val, idx) => (
                      <div key={idx} className="px-2">
                        <span className="text-[9px] text-zinc-500 block mb-1">
                          {["k1", "k2", "p1", "p2", "k3"][idx]}
                        </span>
                        <span>{val.toFixed(5)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-zinc-600 font-mono py-2">
                  No Distortion Coefficients loaded.
                </div>
              )}
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}
