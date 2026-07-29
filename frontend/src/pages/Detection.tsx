import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  Download,
  FileSpreadsheet,
  FileJson,
  RefreshCw,
  Clock,
  CircleDot,
  Bug,
  CheckCircle2,
  XCircle,
  ShieldCheck,
  Layers,
} from "lucide-react";
import { apiService, getBackendUrl } from "../services/api";
import type { DetectionItem, DetectionResponse } from "../types";
import { toast } from "react-hot-toast";

// Per-class accent colors (Tailwind classes)
const CLASS_COLORS: Record<string, string> = {
  cue_ball:  "bg-zinc-300/20 text-zinc-200",
  "1_ball":  "bg-yellow-500/20 text-yellow-300",
  "2_ball":  "bg-blue-500/20 text-blue-300",
  "3_ball":  "bg-red-500/20 text-red-300",
  "4_ball":  "bg-purple-500/20 text-purple-300",
  "5_ball":  "bg-orange-500/20 text-orange-300",
  "6_ball":  "bg-green-500/20 text-green-300",
  "7_ball":  "bg-amber-900/30 text-amber-400",
  "8_ball":  "bg-zinc-800/60 text-zinc-300",
  "9_ball":  "bg-yellow-500/20 text-yellow-300",
  "10_ball": "bg-blue-500/20 text-blue-300",
  "11_ball": "bg-red-500/20 text-red-300",
  "12_ball": "bg-purple-500/20 text-purple-300",
  "13_ball": "bg-orange-500/20 text-orange-300",
  "14_ball": "bg-green-500/20 text-green-300",
  "15_ball": "bg-amber-900/30 text-amber-400",
};

function classColor(name: string) {
  return CLASS_COLORS[name] ?? "bg-blue-500/10 text-blue-400";
}

// Pipeline stage summary derived from all_detections
function buildStages(all: DetectionItem[]) {
  const total = all.length;
  const reasons: Record<string, number> = {};
  let accepted = 0;
  for (const d of all) {
    if (d.is_accepted) { accepted++; }
    else {
      const r = d.rejection_reason ?? "Unknown";
      reasons[r] = (reasons[r] ?? 0) + 1;
    }
  }
  return { total, accepted, rejected: total - accepted, reasons };
}

export default function Detection() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [devMode, setDevMode] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: false,
  });

  const runDetection = async () => {
    if (!file) return;
    setLoading(true);
    const toastId = toast.loading("Processing image with YOLOv8...");
    try {
      const data = await apiService.detectImage(file);
      setResult(data);
      toast.success("Detection complete!", { id: toastId });
    } catch {
      toast.error("Failed to run detection.", { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `detections-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadCsv = () => {
    const rows = result?.detections ?? [];
    if (!rows.length) return;
    const headers = "id,class,confidence,accepted,circularity,aspect_ratio,center_x,center_y,xmin,ymin,xmax,ymax\n";
    const body = rows
      .map((d) =>
        `"${d.detection_id}","${d.class_name}",${d.confidence},${d.is_accepted ?? true},${d.circularity?.toFixed(3) ?? ""},${d.aspect_ratio?.toFixed(3) ?? ""},${d.center_x.toFixed(1)},${d.center_y.toFixed(1)},${d.xmin.toFixed(1)},${d.ymin.toFixed(1)},${d.xmax.toFixed(1)},${d.ymax.toFixed(1)}`
      )
      .join("\n");
    const blob = new Blob([headers + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `detections-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const displayedImageUrl = devMode && result?.debug_annotated_image_url
    ? result.debug_annotated_image_url
    : result?.annotated_image_url;

  const allDets = result?.all_detections ?? [];
  const stages = allDets.length ? buildStages(allDets) : null;
  const rejectedDets = allDets.filter((d) => !d.is_accepted);

  return (
    <div className="space-y-6 pb-12">
      <div className="grid md:grid-cols-3 gap-6">

        {/* ─── Left: Upload ─── */}
        <div className="md:col-span-1 space-y-4">
          <div
            {...getRootProps()}
            className={`p-8 rounded-3xl border-2 border-dashed transition-all duration-200 text-center cursor-pointer flex flex-col items-center justify-center min-h-[260px] ${
              isDragActive
                ? "border-blue-500 bg-blue-500/5"
                : "border-white/5 hover:border-white/20 bg-zinc-950/20"
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="text-zinc-500 mb-4" size={32} />
            <p className="text-sm font-semibold text-white">Drag &amp; drop or Click to browse</p>
            <p className="text-xs text-zinc-500 mt-2">Supports JPG, PNG, WEBP, BMP up to 10MB</p>
          </div>

          {preview && (
            <div className="p-4 rounded-3xl glass space-y-4">
              <h4 className="font-semibold text-xs font-mono uppercase tracking-widest text-zinc-500">
                Selected Preview
              </h4>
              <img src={preview} alt="Preview" className="w-full rounded-2xl border border-white/5 max-h-48 object-cover" />
              <button
                onClick={runDetection}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-800 text-white font-medium text-sm transition-all"
              >
                {loading ? <RefreshCw className="animate-spin" size={16} /> : "Analyze Frame"}
              </button>
            </div>
          )}

          {/* Dev Mode toggle */}
          <div className="p-4 rounded-3xl glass border border-white/5 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bug size={15} className={devMode ? "text-amber-400" : "text-zinc-500"} />
              <span className="text-sm font-semibold text-zinc-300">Developer Mode</span>
            </div>
            <button
              id="dev-mode-toggle"
              onClick={() => setDevMode((v) => !v)}
              className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                devMode ? "bg-amber-500" : "bg-zinc-700"
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200 ${
                  devMode ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {devMode && (
            <p className="text-xs text-amber-400/70 font-mono px-1">
              Debug view enabled — rejected boxes shown in red with reasons.
            </p>
          )}
        </div>

        {/* ─── Right: Results ─── */}
        <div className="md:col-span-2 space-y-6">
          {result ? (
            <div className="p-6 rounded-3xl glass border border-white/5 space-y-6">

              {/* Header row */}
              <div className="flex items-center justify-between border-b border-white/5 pb-4 flex-wrap gap-3">
                <div className="flex items-center gap-4 text-xs font-mono text-zinc-400 flex-wrap">
                  <span className="flex items-center gap-1.5 text-blue-400">
                    <CircleDot size={14} />
                    {result.detections.length} Targets Detected
                  </span>
                  {stages && stages.rejected > 0 && (
                    <span className="flex items-center gap-1.5 text-red-400">
                      <XCircle size={14} />
                      {stages.rejected} Filtered
                    </span>
                  )}
                  <span className="flex items-center gap-1.5">
                    <Clock size={14} />
                    {(result.processing_time * 1000).toFixed(0)}ms latency
                  </span>
                </div>

                <div className="flex gap-2">
                  <a
                    href={`${getBackendUrl()}${displayedImageUrl}`}
                    download
                    target="_blank"
                    rel="noreferrer"
                    className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-white/5 text-zinc-400 hover:text-white transition"
                    title="Download Image"
                  >
                    <Download size={14} />
                  </a>
                  <button
                    onClick={downloadCsv}
                    className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-white/5 text-zinc-400 hover:text-white transition"
                    title="Download CSV"
                  >
                    <FileSpreadsheet size={14} />
                  </button>
                  <button
                    onClick={downloadJson}
                    className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-white/5 text-zinc-400 hover:text-white transition"
                    title="Download JSON"
                  >
                    <FileJson size={14} />
                  </button>
                </div>
              </div>

              {/* Pipeline Stage Summary (Dev Mode only) */}
              {devMode && stages && (
                <div className="rounded-2xl bg-zinc-900/60 border border-amber-500/20 p-4 space-y-3">
                  <div className="flex items-center gap-2 text-amber-400 text-xs font-mono uppercase tracking-widest">
                    <Layers size={13} />
                    Post-Processing Pipeline Audit
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                    <div className="p-2 rounded-lg bg-zinc-800/80 text-center">
                      <div className="text-zinc-400 mb-1">Raw Detections</div>
                      <div className="text-white text-lg font-bold">{stages.total}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-green-500/10 border border-green-500/20 text-center">
                      <div className="text-green-400 mb-1">Accepted</div>
                      <div className="text-green-300 text-lg font-bold">{stages.accepted}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/20 text-center">
                      <div className="text-red-400 mb-1">Filtered</div>
                      <div className="text-red-300 text-lg font-bold">{stages.rejected}</div>
                    </div>
                  </div>
                  {Object.keys(stages.reasons).length > 0 && (
                    <div className="space-y-1">
                      <div className="text-zinc-500 text-xs mb-2">Rejection Breakdown:</div>
                      {Object.entries(stages.reasons).map(([reason, count]) => (
                        <div key={reason} className="flex justify-between items-center text-xs text-zinc-400 font-mono">
                          <span className="text-red-400/80">{reason}</span>
                          <span className="px-1.5 py-0.5 rounded bg-red-500/10 text-red-300">{count}×</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Annotated Image */}
              <div className="relative rounded-2xl overflow-hidden border border-white/5 bg-zinc-950/80 flex items-center justify-center">
                {devMode && (
                  <span className="absolute top-2 left-2 z-10 text-xs font-mono px-2 py-1 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    DEBUG VIEW
                  </span>
                )}
                <img
                  src={`${getBackendUrl()}${displayedImageUrl}`}
                  alt="Detections annotated"
                  className="max-h-[500px] w-auto object-contain"
                />
              </div>

              {/* Accepted Detections Table */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <ShieldCheck size={14} className="text-green-400" />
                  <h4 className="font-semibold text-xs font-mono uppercase tracking-widest text-zinc-500">
                    Detection Summary
                  </h4>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="border-b border-white/5 text-zinc-500 font-mono">
                        <th className="py-2">ID</th>
                        <th className="py-2">Class</th>
                        <th className="py-2">Confidence</th>
                        {devMode && <th className="py-2">Circularity</th>}
                        {devMode && <th className="py-2">A.Ratio</th>}
                        <th className="py-2 text-right">Center (X, Y) px</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 font-mono text-zinc-300">
                      {result.detections.map((det) => (
                        <tr key={det.detection_id}>
                          <td className="py-2 text-zinc-500">{det.detection_id}</td>
                          <td className="py-2">
                            <span className={`px-2 py-0.5 rounded-md text-xs font-semibold ${classColor(det.class_name)}`}>
                              {det.class_name}
                            </span>
                          </td>
                          <td className="py-2">
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 rounded-full bg-zinc-700/60 flex-1 max-w-[60px]">
                                <div
                                  className="h-full rounded-full bg-blue-500"
                                  style={{ width: `${(det.confidence * 100).toFixed(0)}%` }}
                                />
                              </div>
                              <span className="text-zinc-300">{(det.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                          {devMode && (
                            <td className="py-2 text-zinc-400">
                              {det.circularity != null ? det.circularity.toFixed(2) : "—"}
                            </td>
                          )}
                          {devMode && (
                            <td className="py-2 text-zinc-400">
                              {det.aspect_ratio != null ? det.aspect_ratio.toFixed(2) : "—"}
                            </td>
                          )}
                          <td className="py-2 text-right text-zinc-400">
                            ({det.center_x.toFixed(0)}, {det.center_y.toFixed(0)})
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Rejected Detections Log (Dev Mode only) */}
              {devMode && rejectedDets.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <XCircle size={14} className="text-red-400" />
                    <h4 className="font-semibold text-xs font-mono uppercase tracking-widest text-red-500/70">
                      Filtered / Rejected Detections
                    </h4>
                  </div>
                  <div className="overflow-x-auto rounded-xl border border-red-500/10 bg-red-500/5">
                    <table className="w-full text-xs text-left border-collapse">
                      <thead>
                        <tr className="border-b border-red-500/10 text-zinc-500 font-mono">
                          <th className="py-2 px-3">ID</th>
                          <th className="py-2 px-3">Class</th>
                          <th className="py-2 px-3">Conf</th>
                          <th className="py-2 px-3">Circularity</th>
                          <th className="py-2 px-3">Reason</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-red-500/5 font-mono">
                        {rejectedDets.map((det) => (
                          <tr key={det.detection_id} className="text-zinc-400">
                            <td className="py-2 px-3 text-zinc-500">{det.detection_id}</td>
                            <td className="py-2 px-3 text-red-300/80">{det.class_name}</td>
                            <td className="py-2 px-3">{(det.confidence * 100).toFixed(0)}%</td>
                            <td className="py-2 px-3">
                              {det.circularity != null ? det.circularity.toFixed(2) : "—"}
                            </td>
                            <td className="py-2 px-3">
                              <span className="px-2 py-0.5 rounded bg-red-500/10 text-red-300 border border-red-500/20">
                                {det.rejection_reason ?? "Unknown"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Accepted count badge */}
              {!devMode && (
                <div className="flex items-center gap-2 text-xs text-green-400/70 font-mono">
                  <CheckCircle2 size={13} />
                  Post-processing verified — false positives automatically filtered
                </div>
              )}
            </div>
          ) : (
            <div className="h-[400px] rounded-3xl border border-white/5 bg-zinc-950/10 flex items-center justify-center text-zinc-500 text-sm font-sans">
              Perform an analysis to display results
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
