import { useState, useCallback } from "react";
import {
  Upload,
  Play,
  Crosshair,
  X,
  Compass,
  Info,
} from "lucide-react";
import { apiService, getBackendUrl } from "../services/api";
import { RestorationEngine, SNOOKER_BALL_COLORS } from "../services/snooker/restorationEngine";
import { RestorationWizard } from "../components/snooker/RestorationWizard";
import type {
  BallState,
  DisplacedBall,
  RestorationMode,
  RestorationState,
  RestorationWizardState,
  TableCalibrationPoints,
} from "../types/snooker";
import { toast } from "react-hot-toast";

// ── Helper: next pending index ───────────────────────────────────────────────
function nextPendingIndex(queue: DisplacedBall[], from: number): number {
  for (let i = from; i < queue.length; i++) {
    if (queue[i].status === "PENDING") return i;
  }
  return queue.length;
}


export default function SnookerPhoto() {
  const [activeTab, setActiveTab] = useState<"single" | "before_after" | "calibration">("single");

  // Single Photo State
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [singlePreview, setSinglePreview] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0.25);
  const [iou, setIou] = useState<number>(0.45);
  const [loading, setLoading] = useState<boolean>(false);
  const [detectionResult, setDetectionResult] = useState<any | null>(null);
  const [detections, setDetections] = useState<any[]>([]);

  // Before / After State
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [beforePreview, setBeforePreview] = useState<string | null>(null);
  const [beforeDetections, setBeforeDetections] = useState<BallState[]>([]);

  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [afterPreview, setAfterPreview] = useState<string | null>(null);
  const [afterDetections, setAfterDetections] = useState<BallState[]>([]);

  const [wizard, setWizard] = useState<RestorationWizardState | null>(null);

  // Table Calibration State
  const [calibration] = useState<TableCalibrationPoints>({
    topLeft: { x: 50, y: 50 },
    topRight: { x: 850, y: 50 },
    bottomRight: { x: 850, y: 430 },
    bottomLeft: { x: 50, y: 430 },
    isCalibrated: true,
    tableWidthMm: 3569,
    tableHeightMm: 1778,
  });


  const handleSingleFileChange = (file: File) => {
    setSingleFile(file);
    setSinglePreview(URL.createObjectURL(file));
    setDetectionResult(null);
    setDetections([]);
  };

  const handleDetectSingle = async () => {
    if (!singleFile) return;
    setLoading(true);
    try {
      const res = await apiService.detectSnookerPhoto(singleFile, confidence, iou);
      setDetectionResult(res);
      setDetections(res.detections || []);
      toast.success(`Detected ${res.detections?.length || 0} Snooker balls!`);
    } catch (err) {
      toast.error("Failed to process Snooker photo detection.");
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFalsePositive = (idx: number) => {
    setDetections((prev) => prev.filter((_, i) => i !== idx));
    toast.success("Removed false positive detection.");
  };

  const handleBeforeFileChange = async (file: File) => {
    setBeforeFile(file);
    setBeforePreview(URL.createObjectURL(file));
    try {
      const res = await apiService.detectSnookerPhoto(file, confidence, iou);
      const balls: BallState[] = (res.detections || []).map((d: any, idx: number) => ({
        id: `before_${idx}`,
        class_id: d.class_id,
        class_name: d.class_name,
        x: d.center_x,
        y: d.center_y,
        norm_x: d.center_x / (res.width || 1000),
        norm_y: d.center_y / (res.height || 600),
        confidence: d.confidence,
        color: SNOOKER_BALL_COLORS[d.class_id] || "#FFFFFF",
      }));
      setBeforeDetections(balls);
    } catch (err) {
      toast.error("Failed to detect balls on Before Shot photo.");
    }
  };

  const handleAfterFileChange = async (file: File) => {
    setAfterFile(file);
    setAfterPreview(URL.createObjectURL(file));
    try {
      const res = await apiService.detectSnookerPhoto(file, confidence, iou);
      const balls: BallState[] = (res.detections || []).map((d: any, idx: number) => ({
        id: `after_${idx}`,
        class_id: d.class_id,
        class_name: d.class_name,
        x: d.center_x,
        y: d.center_y,
        norm_x: d.center_x / (res.width || 1000),
        norm_y: d.center_y / (res.height || 600),
        confidence: d.confidence,
        color: SNOOKER_BALL_COLORS[d.class_id] || "#FFFFFF",
      }));
      setAfterDetections(balls);
    } catch (err) {
      toast.error("Failed to detect balls on After Shot photo.");
    }
  };

  const handleCompareStates = useCallback(() => {
    if (beforeDetections.length === 0 || afterDetections.length === 0) {
      toast.error("Both 'Before' and 'After' shot photos are required for comparison.");
      return;
    }
    const matched = RestorationEngine.spatialMatchBalls(beforeDetections, afterDetections);
    const rawDisplaced = RestorationEngine.detectDisplacedBalls(matched, 0.003, calibration);
    const sorted = RestorationEngine.createRestorationQueue(rawDisplaced, "automatic").displaced_balls;

    // Add human-readable labels (Red #1, Red #2, ...)
    let redCount = 0;
    const displaced: DisplacedBall[] = sorted.map((b) => {
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

    setWizard({
      mode: "automatic",
      currentState: displaced.length > 0 ? "SHOW_LASER" : "RESTORATION_COMPLETE",
      queue: displaced,
      currentIndex: 0,
      restoredIds: [],
      skippedIds: [],
    });
    toast.success(`Found ${displaced.length} displaced ball${displaced.length !== 1 ? "s" : ""}!`);
  }, [beforeDetections, afterDetections, calibration]);

  // ── Wizard action handlers (photo mode — no canvas animation) ────────────
  const handleWizardModeChange = useCallback((newMode: RestorationMode) => {
    if (!wizard) return;
    const pending = wizard.queue.filter((b) => b.status === "PENDING");
    let newQueue: DisplacedBall[];
    if (newMode === "automatic") {
      newQueue = RestorationEngine.createRestorationQueue(pending, "automatic").displaced_balls;
    } else {
      newQueue = [...pending];
    }
    setWizard({
      mode: newMode,
      currentState: newMode === "manual" ? "SELECT_BALL" : "SHOW_LASER",
      queue: newQueue,
      currentIndex: nextPendingIndex(newQueue, 0),
      restoredIds: wizard.restoredIds,
      skippedIds: wizard.skippedIds,
    });
  }, [wizard]);

  const handleSelectBall = useCallback((id: string) => {
    if (!wizard) return;
    const idx = wizard.queue.findIndex((b) => b.id === id);
    if (idx === -1) return;
    setWizard((prev) => prev ? { ...prev, currentIndex: idx, currentState: "SHOW_LASER" } : prev);
  }, [wizard]);


  // Photo mode: no animation, just instantly mark restored
  const handleRestoreThisBall = useCallback(() => {
    if (!wizard) return;
    const activeBall = wizard.queue[wizard.currentIndex];
    if (!activeBall || activeBall.status !== "PENDING") return;
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
        : prev.mode === "manual" ? "SELECT_BALL" : "SHOW_LASER";
      return { ...prev, queue: updatedQueue, currentIndex: isComplete ? prev.currentIndex : nextIdx, restoredIds: newRestored, currentState: nextState };
    });
  }, [wizard]);

  const handleSkipBall = useCallback(() => {
    if (!wizard) return;
    const activeBall = wizard.queue[wizard.currentIndex];
    if (!activeBall) return;
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
        : prev.mode === "manual" ? "SELECT_BALL" : "SHOW_LASER";
      return { ...prev, queue: updatedQueue, currentIndex: isComplete ? prev.currentIndex : nextIdx, skippedIds: newSkipped, currentState: nextState };
    });
  }, [wizard]);

  // Photo: animation complete is instant (no canvas)
  const handleAnimationComplete = useCallback(() => {
    handleRestoreThisBall();
  }, [handleRestoreThisBall]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto p-4 md:p-6 text-zinc-100 font-sans">
      {/* Header Banner */}
      <div className="p-6 rounded-3xl bg-zinc-950/80 border border-white/10 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs text-red-500 font-bold uppercase tracking-wider">
            <span>🔴 Snooker AI Vision Suite</span>
            <span>•</span>
            <span>YOLOv8 8-Class Model</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-white mt-1">Snooker Photo Analysis & AI Calibration</h1>
          <p className="text-xs text-zinc-400 mt-1">
            Detect balls from real table photographs, calibrate 4-point homography, and generate AI-guided virtual laser restoration directions.
          </p>
        </div>

        {/* Tab Navigation Controls */}
        <div className="flex bg-zinc-900/90 p-1.5 rounded-2xl border border-white/5 font-mono text-xs">
          <button
            onClick={() => setActiveTab("single")}
            className={`px-4 py-2 rounded-xl transition-all font-semibold ${
              activeTab === "single"
                ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            📷 Single Detection
          </button>
          <button
            onClick={() => setActiveTab("before_after")}
            className={`px-4 py-2 rounded-xl transition-all font-semibold ${
              activeTab === "before_after"
                ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            🔄 Before / After Compare
          </button>
          <button
            onClick={() => setActiveTab("calibration")}
            className={`px-4 py-2 rounded-xl transition-all font-semibold ${
              activeTab === "calibration"
                ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            📐 Calibration
          </button>
        </div>
      </div>

      {/* SINGLE PHOTO DETECTION TAB */}
      {activeTab === "single" && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Controls Panel */}
          <div className="p-6 rounded-2xl bg-zinc-950/80 border border-white/10 space-y-6">
            <h3 className="font-bold text-base text-white flex items-center gap-2">
              <Upload size={18} className="text-red-400" />
              Upload Snooker Photo
            </h3>

            <label className="border-2 border-dashed border-white/10 hover:border-red-500/50 rounded-2xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all bg-zinc-900/40 min-h-[220px]">
              {singlePreview ? (
                <img src={singlePreview} alt="Preview" className="max-h-48 object-contain rounded-lg" />
              ) : (
                <>
                  <Upload size={40} className="text-zinc-500 mb-2" />
                  <span className="text-xs text-zinc-300 font-medium">Drop Snooker image or click to browse</span>
                  <span className="text-[10px] text-zinc-500 mt-1">JPG, PNG, JPEG supported</span>
                </>
              )}
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleSingleFileChange(e.target.files[0])}
              />
            </label>

            {/* Single Photo Fallback Warning Banner */}
            <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 font-mono space-y-1">
              <div className="flex items-center gap-1.5 font-bold">
                <Info size={14} />
                <span>Single Photo Mode Active</span>
              </div>
              <p className="text-[11px] text-amber-400/80 leading-relaxed">
                Previous table state unavailable. Please upload pre-shot image in "Before/After Compare" or use Manual Restore.
              </p>
            </div>

            {/* Threshold Sliders */}
            <div className="space-y-4 font-mono text-xs pt-2">
              <div>
                <div className="flex justify-between text-zinc-400 mb-1">
                  <span>Confidence Threshold:</span>
                  <strong className="text-red-400">{(confidence * 100).toFixed(0)}%</strong>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                  className="w-full accent-red-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-zinc-400 mb-1">
                  <span>IoU NMS Threshold:</span>
                  <strong className="text-blue-400">{(iou * 100).toFixed(0)}%</strong>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={iou}
                  onChange={(e) => setIou(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
              </div>
            </div>

            <button
              onClick={handleDetectSingle}
              disabled={!singleFile || loading}
              className="w-full py-3 rounded-xl bg-red-600 hover:bg-red-700 disabled:opacity-50 font-bold text-xs text-white transition-all shadow-lg shadow-red-600/30 flex items-center justify-center gap-2"
            >
              {loading ? "Processing Model Inference..." : "Run Snooker Detection"}
            </button>
          </div>

          {/* Detection Preview Overlay */}
          <div className="lg:col-span-2 p-6 rounded-2xl bg-zinc-950/80 border border-white/10 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-base text-white flex items-center gap-2">
                <Crosshair size={18} className="text-red-400" />
                Detection Results & Annotations
              </h3>
              <span className="text-xs font-mono text-zinc-400">
                {detections.length} Balls Found
              </span>
            </div>

            <div className="relative rounded-xl overflow-hidden bg-black border border-white/10 min-h-[400px] flex items-center justify-center">
              {detectionResult?.annotated_image_url ? (
                <img
                  src={`${getBackendUrl()}${detectionResult.annotated_image_url}`}
                  alt="Annotated Detection"
                  className="w-full h-auto object-contain block"
                />
              ) : singlePreview ? (
                <img src={singlePreview} alt="Preview" className="w-full h-auto object-contain block opacity-70" />
              ) : (
                <div className="text-center text-zinc-600 space-y-2 font-mono">
                  <Upload className="mx-auto" size={40} />
                  <p className="text-xs">No image loaded yet.</p>
                </div>
              )}
            </div>

            {/* Detections List & Low Confidence Warnings */}
            {detections.length > 0 && (
              <div className="space-y-2 pt-2 font-mono text-xs">
                <span className="text-zinc-400 block font-bold">Detected Snooker Objects (Manual False Positive Removal):</span>
                <div className="grid sm:grid-cols-2 gap-2 max-h-48 overflow-y-auto pr-1">
                  {detections.map((d: any, idx: number) => {
                    const isLowConf = d.confidence < 0.65;
                    return (
                      <div
                        key={idx}
                        className={`p-2.5 rounded-xl border flex items-center justify-between transition-all ${
                          isLowConf
                            ? "bg-amber-950/20 border-amber-500/30 text-amber-200"
                            : "bg-zinc-900 border-white/5 text-zinc-200"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className="w-3.5 h-3.5 rounded-full border border-white/20"
                            style={{ backgroundColor: SNOOKER_BALL_COLORS[d.class_id] || "#FFF" }}
                          />
                          <span className="font-bold capitalize">{d.class_name.replace("_", " ")}</span>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isLowConf ? "bg-amber-500/20 text-amber-400" : "bg-emerald-500/20 text-emerald-400"
                          }`}>
                            {(d.confidence * 100).toFixed(0)}%
                          </span>
                          <button
                            onClick={() => handleRemoveFalsePositive(idx)}
                            className="text-zinc-500 hover:text-red-400 transition-all p-0.5"
                            title="Remove False Detection"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* BEFORE / AFTER SHOT COMPARE TAB */}
      {activeTab === "before_after" && (
        <div className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Before Shot Card */}
            <div className="p-6 rounded-2xl bg-zinc-950/80 border border-white/10 space-y-4">
              <h3 className="font-bold text-sm text-white flex items-center gap-2">
                <span className="w-6 h-6 rounded-md bg-blue-500/20 text-blue-400 flex items-center justify-center font-mono text-xs">1</span>
                "Before Shot" State
              </h3>

              <label className="border-2 border-dashed border-white/10 hover:border-blue-500/50 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all bg-zinc-900/40 min-h-[220px]">
                {beforePreview ? (
                  <img src={beforePreview} alt="Before Shot" className="max-h-48 object-contain rounded-lg" />
                ) : (
                  <>
                    <Upload size={32} className="text-zinc-500 mb-2" />
                    <span className="text-xs text-zinc-300 font-medium">Upload "Before Shot" Image</span>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleBeforeFileChange(e.target.files[0])}
                />
              </label>

              <div className="text-xs font-mono text-zinc-400 flex justify-between">
                <span>File: {beforeFile ? beforeFile.name : "None"} | Balls Detected:</span>
                <strong className="text-blue-400">{beforeDetections.length}</strong>
              </div>
            </div>

            {/* After Shot Card */}
            <div className="p-6 rounded-2xl bg-zinc-950/80 border border-white/10 space-y-4">
              <h3 className="font-bold text-sm text-white flex items-center gap-2">
                <span className="w-6 h-6 rounded-md bg-amber-500/20 text-amber-400 flex items-center justify-center font-mono text-xs">2</span>
                "After Shot" State
              </h3>

              <label className="border-2 border-dashed border-white/10 hover:border-amber-500/50 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all bg-zinc-900/40 min-h-[220px]">
                {afterPreview ? (
                  <img src={afterPreview} alt="After Shot" className="max-h-48 object-contain rounded-lg" />
                ) : (
                  <>
                    <Upload size={32} className="text-zinc-500 mb-2" />
                    <span className="text-xs text-zinc-300 font-medium">Upload "After Shot" Image</span>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleAfterFileChange(e.target.files[0])}
                />
              </label>

              <div className="text-xs font-mono text-zinc-400 flex justify-between">
                <span>File: {afterFile ? afterFile.name : "None"} | Balls Detected:</span>
                <strong className="text-amber-400">{afterDetections.length}</strong>
              </div>
            </div>
          </div>

          <div className="text-center">
            <button
              onClick={handleCompareStates}
              className="py-3.5 px-8 rounded-2xl bg-red-600 hover:bg-red-700 font-bold text-sm text-white transition-all shadow-xl shadow-red-600/30 flex items-center gap-2 mx-auto"
            >
              <Play size={16} /> Run Table State Comparison & Foul Restoration Analysis
            </button>
          </div>

          {/* One-Ball-at-a-Time Restoration Wizard */}
          {wizard && (
            <RestorationWizard
              wizardState={wizard}
              onModeChange={handleWizardModeChange}
              onRestoreThisBall={handleRestoreThisBall}
              onSkipBall={handleSkipBall}
              onCancel={() => setWizard(null)}
              onSelectBall={handleSelectBall}
              onAnimationComplete={handleAnimationComplete}
            />
          )}
        </div>
      )}

      {/* TABLE CALIBRATION TAB */}
      {activeTab === "calibration" && (
        <div className="p-6 rounded-2xl bg-zinc-950/80 border border-white/10 space-y-6">
          <div className="flex items-center gap-3 border-b border-white/10 pb-4">
            <Compass size={24} className="text-red-400" />
            <div>
              <h3 className="font-bold text-base text-white">4-Point Table Perspective Calibration</h3>
              <p className="text-xs text-zinc-400">
                Maps 2D camera pixels to official Snooker table physical coordinates (3569 mm × 1778 mm).
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 text-xs font-mono">
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-white/5 space-y-3">
              <span className="text-zinc-400 font-bold block">Table Corner Coordinates (Pixels):</span>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-white/5">
                  Top-Left: ({calibration.topLeft.x}, {calibration.topLeft.y})
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-white/5">
                  Top-Right: ({calibration.topRight.x}, {calibration.topRight.y})
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-white/5">
                  Bottom-Left: ({calibration.bottomLeft.x}, {calibration.bottomLeft.y})
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-white/5">
                  Bottom-Right: ({calibration.bottomRight.x}, {calibration.bottomRight.y})
                </div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-zinc-900/60 border border-white/5 space-y-3">
              <span className="text-zinc-400 font-bold block">Physical Dimensions:</span>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Table Width:</span>
                  <strong className="text-emerald-400">{calibration.tableWidthMm} mm</strong>
                </div>
                <div className="flex justify-between">
                  <span>Table Length:</span>
                  <strong className="text-emerald-400">{calibration.tableHeightMm} mm</strong>
                </div>
                <div className="flex justify-between">
                  <span>Calibration Status:</span>
                  <strong className="text-blue-400">ACTIVE (100% Calibrated)</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
