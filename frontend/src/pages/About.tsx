import { Info, BookOpen, Code } from "lucide-react";

export default function About() {
  return (
    <div className="space-y-6 pb-12 max-w-3xl">
      <div className="p-6 rounded-3xl glass border border-white/5 space-y-6">
        <div className="flex items-center gap-2 border-b border-white/5 pb-4">
          <Info className="text-blue-500" size={18} />
          <h3 className="font-semibold text-white text-sm font-sans">Project Technical Overview</h3>
        </div>

        <div className="space-y-4 text-xs text-zinc-400 leading-relaxed font-sans">
          <p>
            <strong>Project Name:</strong> AI-Based Pool Ball Identification and Laser Positioning System
          </p>
          <p>
            This system integrates state-of-the-art computer vision models (Ultralytics YOLOv8) with advanced mathematical camera calibration (checkerboard calibration) and geometric mapping (perspective homography) to detect targets on a billiards table and translate them into physical real-world millimeter coordinates.
          </p>
          <p>
            The final objective is to feed these computed coordinates into a physical stepper motor control pan-tilt setup using ESP32 controllers to point a targeting laser directly at selected balls.
          </p>
        </div>

        {/* Timeline block */}
        <div className="space-y-3">
          <h4 className="font-semibold text-xs font-mono uppercase tracking-widest text-zinc-500">
            Development Phase Index
          </h4>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/40 border border-white/5">
              <span>Phase 1-3: Dataset prep & YOLO Training</span>
              <span className="text-emerald-500 font-semibold">✓ Completed</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/40 border border-white/5">
              <span>Phase 4: Offline Detection Pipeline</span>
              <span className="text-emerald-500 font-semibold">✓ Completed</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/40 border border-white/5">
              <span>Phase 5: Camera Checkerboard Calibration</span>
              <span className="text-emerald-500 font-semibold">✓ Completed</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/40 border border-white/5">
              <span>Phase 6: Perspective Homography Mapping</span>
              <span className="text-emerald-500 font-semibold">✓ Completed</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/40 border border-white/5">
              <span>Phase 7: FastAPI REST API Engine</span>
              <span className="text-emerald-500 font-semibold">✓ Completed</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/40 border border-white/5">
              <span>Phase 8: Apple-style Frontend Interface</span>
              <span className="text-blue-500 font-semibold">✓ Active</span>
            </div>
          </div>
        </div>

        {/* Dependencies links */}
        <div className="flex gap-4 border-t border-white/5 pt-6">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-zinc-500 hover:text-white transition text-xs font-mono"
          >
            <Code size={14} />
            Source Repository
          </a>
          <a
            href="https://docs.ultralytics.com"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-zinc-500 hover:text-white transition text-xs font-mono"
          >
            <BookOpen size={14} />
            YOLOv8 Documentation
          </a>
        </div>
      </div>
    </div>
  );
}
