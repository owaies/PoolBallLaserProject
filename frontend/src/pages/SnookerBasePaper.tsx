import { useState } from "react";
import {
  FileText,
  Download,
  ExternalLink,
  Maximize2,
  Check,
  ArrowRight,
  TrendingUp,
  Cpu,
  Layers,
  BookOpen,
} from "lucide-react";

export default function SnookerBasePaper() {
  const [fullscreen, setFullscreen] = useState(false);

  const handleFullscreen = () => {
    const elem = document.getElementById("pdf-viewer-container");
    if (elem) {
      if (!document.fullscreenElement) {
        elem.requestFullscreen().then(() => setFullscreen(true)).catch(() => {});
      } else {
        document.exitFullscreen().then(() => setFullscreen(false)).catch(() => {});
      }
    }
  };

  const comparisonFeatures = [
    { name: "Ball detection", base: true, ours: true },
    { name: "Deep-learning vision", base: true, ours: true },
    { name: "Snooker-specific classes", base: false, ours: true, note: "Base paper detects general pool balls (16 classes); ours detects snooker balls (8 classes)" },
    { name: "Multiple-ball detection", base: true, ours: true },
    { name: "Table calibration", base: true, ours: true, note: "Base paper uses static bounds; ours uses checkerboard camera calibration & homography" },
    { name: "Before/after state comparison", base: false, ours: true },
    { name: "Foul detection", base: false, ours: true },
    { name: "Foul & Miss workflow", base: false, ours: true },
    { name: "Ball displacement calculation", base: false, ours: true },
    { name: "Multiple displaced-ball handling", base: false, ours: true },
    { name: "Multiple red-ball matching", base: false, ours: true, note: "Ours uses Hungarian spatial matching" },
    { name: "Referee decision support", base: false, ours: true },
    { name: "Automatic restoration guidance", base: false, ours: true },
    { name: "Manual restoration controls", base: false, ours: true },
    { name: "Virtual laser overlay", base: false, ours: true },
    { name: "Physical laser integration", base: false, ours: true, note: "Base paper controls a 6-axis robotic arm; ours controls an ESP32 pan-tilt laser module" },
    { name: "Playable simulation dashboard", base: false, ours: true },
  ];

  return (
    <div className="space-y-10 pb-16 font-sans">
      {/* ── HEADER ── */}
      <div className="border-b border-white/5 pb-6">
        <p className="text-[10px] uppercase tracking-widest text-emerald-500 font-mono font-semibold mb-2">
          Academic Research Foundation
        </p>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          BASE PAPER
        </h1>
        <p className="text-lg text-zinc-300 font-semibold mt-1">
          "Evaluation of Pool Balls Detection Using Deep Learning Segmentation"
        </p>
        <p className="text-xs text-zinc-500 font-mono mt-1">
          Research Foundation of the AI-Assisted Snooker Referee &amp; Laser Restoration System
        </p>
      </div>

      {/* ── METADATA CARDS ── */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl glass space-y-1">
          <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Authors</span>
          <span className="text-xs text-white font-medium block">
            Y.-H. Wang, M.-T. Lin, Y.-Z. Li, G.-Y. Lin, &amp; Y.-C. Chen
          </span>
        </div>
        <div className="p-4 rounded-2xl glass space-y-1">
          <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Year</span>
          <span className="text-xs text-white font-medium block">2023</span>
        </div>
        <div className="p-4 rounded-2xl glass space-y-1">
          <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Research Area</span>
          <span className="text-xs text-white font-medium block">Computer Vision &amp; Pool Robotics</span>
        </div>
        <div className="p-4 rounded-2xl glass space-y-1">
          <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Primary Contribution</span>
          <span className="text-xs text-white font-medium block">Deep learning segmentation for pool robots</span>
        </div>
      </div>

      {/* ── TWO-COLUMN VIEW: PDF VIEWER & BASE PAPER SUMMARY ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* PDF Viewer Block */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono flex items-center gap-2">
              <FileText size={16} className="text-red-500" />
              Paper Document
            </h3>
            <div className="flex gap-2">
              <button
                onClick={handleFullscreen}
                className="p-1.5 rounded-lg bg-zinc-900 border border-white/5 text-zinc-400 hover:text-white hover:bg-zinc-800 transition text-xs flex items-center gap-1.5"
                title="Fullscreen Preview"
              >
                <Maximize2 size={13} />
                <span className="hidden sm:inline">Fullscreen</span>
              </button>
              <a
                href="/base-paper.pdf"
                target="_blank"
                rel="noreferrer"
                className="p-1.5 rounded-lg bg-zinc-900 border border-white/5 text-zinc-400 hover:text-white hover:bg-zinc-800 transition text-xs flex items-center gap-1.5"
              >
                <ExternalLink size={13} />
                <span className="hidden sm:inline">Open PDF</span>
              </a>
              <a
                href="/base-paper.pdf"
                download="Evaluation_of_Pool_Balls_Detection_Wang_2023.pdf"
                className="p-1.5 rounded-lg bg-zinc-900 border border-white/5 text-zinc-400 hover:text-white hover:bg-zinc-800 transition text-xs flex items-center gap-1.5"
              >
                <Download size={13} />
                <span className="hidden sm:inline">Download</span>
              </a>
            </div>
          </div>

          <div
            id="pdf-viewer-container"
            className={`w-full rounded-2xl overflow-hidden border border-white/5 bg-zinc-950/40 relative ${
              fullscreen ? "h-screen p-4" : "h-[620px]"
            }`}
          >
            <iframe
              src="/base-paper.pdf#toolbar=1&navpanes=1"
              className="w-full h-full border-0"
              title="Base Paper PDF Viewer"
            />
          </div>
          <p className="text-[10px] text-zinc-500 font-mono text-right">
            Note: Use the embedded PDF reader interface to adjust zoom, navigate pages, or print the document.
          </p>
        </div>

        {/* Base Paper Summary Block */}
        <div className="lg:col-span-5 space-y-6">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono flex items-center gap-2">
            <BookOpen size={16} className="text-emerald-500" />
            Base Paper Summary
          </h3>

          <div className="p-6 rounded-2xl glass space-y-5 text-xs text-zinc-400 leading-relaxed">
            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono font-bold text-white block">Problem Addressed</span>
              <p>
                The paper addresses the challenge of identifying and locating pool balls under variable table conditions, specifically tackling visual distractors like light glare, table shadows, and background noise that disrupt standard robotic vision.
              </p>
            </div>

            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono font-bold text-white block">Computer-Vision Approach</span>
              <p>
                The authors compare two deep learning architectures: U-Net and Mask R-CNN. U-Net is utilized for fine-grained semantic segmentation, while Mask R-CNN is used for instance segmentation and bounding-box classification.
              </p>
            </div>

            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono font-bold text-white block">Detection &amp; Segmentation Workflow</span>
              <p>
                A top-down camera captures images which are processed through the trained networks. For the U-Net path, a median filter and a Hough Circle Transform are applied post-segmentation to locate the ball centers and radii. Mask R-CNN directly yields object coordinates via predicted bounding boxes.
              </p>
            </div>

            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono font-bold text-white block">Reported Performance (Table I Metrics)</span>
              <div className="overflow-x-auto border border-white/5 rounded-lg bg-zinc-950/20">
                <table className="min-w-full divide-y divide-white/5 text-[10px]">
                  <thead>
                    <tr className="bg-zinc-950/40 text-white font-mono">
                      <th className="px-3 py-1.5 text-left">Metric</th>
                      <th className="px-3 py-1.5 text-left">U-Net</th>
                      <th className="px-3 py-1.5 text-left">Mask R-CNN</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-zinc-400">
                    <tr>
                      <td className="px-3 py-1.5 font-mono">IoU</td>
                      <td className="px-3 py-1.5">0.953</td>
                      <td className="px-3 py-1.5">0.956</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-1.5 font-mono">mAP</td>
                      <td className="px-3 py-1.5">0.998</td>
                      <td className="px-3 py-1.5">0.957</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-1.5 font-mono">mAR</td>
                      <td className="px-3 py-1.5">0.975</td>
                      <td className="px-3 py-1.5">0.951</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-1.5 font-mono">Precision</td>
                      <td className="px-3 py-1.5">0.973</td>
                      <td className="px-3 py-1.5">0.930</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-1.5 font-mono">Inference</td>
                      <td className="px-3 py-1.5 text-emerald-500 font-semibold">6.1 ms</td>
                      <td className="px-3 py-1.5">110 ms</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-1.5 font-mono">Memory</td>
                      <td className="px-3 py-1.5">121.3 MB</td>
                      <td className="px-3 py-1.5">172.1 MB</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-zinc-500">
                Robotic execution metrics: Average 20 strokes per game, potting rate exceeding 80%, average time of 15s per stroke.
              </p>
            </div>

            <div className="space-y-2">
              <span className="text-[10px] uppercase font-mono font-bold text-white block">Explicit Limitations</span>
              <ul className="list-disc pl-4 space-y-1">
                <li>Extremely limited dataset size of only 100 images.</li>
                <li>U-Net coordinates require post-processing (Hough circles) to determine centers, which introduces geometric approximation error.</li>
                <li>Restricted to single-stroke execution paths without tactical referee logic.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* ── BASE PAPER APPROACH DIAGRAM ── */}
      <div className="p-6 rounded-2xl glass space-y-4">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono">
          Base Paper Approach
        </h3>
        <div className="flex flex-col md:flex-row items-center justify-center gap-3 py-6 bg-zinc-950/20 border border-white/5 rounded-xl text-xs text-white">
          <div className="px-4 py-2 bg-zinc-900 border border-white/5 rounded-lg font-mono">Camera / Image</div>
          <ArrowRight size={14} className="text-zinc-500 rotate-90 md:rotate-0" />
          <div className="px-4 py-2 bg-zinc-900 border border-white/5 rounded-lg font-mono">Image Preprocessing</div>
          <ArrowRight size={14} className="text-zinc-500 rotate-90 md:rotate-0" />
          <div className="px-4 py-2 bg-zinc-900 border border-white/5 rounded-lg font-mono">Deep Learning Segmentation</div>
          <ArrowRight size={14} className="text-zinc-500 rotate-90 md:rotate-0" />
          <div className="px-4 py-2 bg-zinc-900 border border-white/5 rounded-lg font-mono">Pool Ball Detection</div>
          <ArrowRight size={14} className="text-zinc-500 rotate-90 md:rotate-0" />
          <div className="px-4 py-2 bg-emerald-950/50 border border-emerald-500/20 text-emerald-400 rounded-lg font-mono">Ball Localization / ID</div>
        </div>
      </div>

      {/* ── IDENTIFIED RESEARCH GAPS ── */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono">
          Identified Research Gaps (Scope not addressed by the base paper)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-zinc-950/30 border border-white/5 text-xs text-zinc-400 space-y-2">
            <span className="text-red-400 font-semibold block">1. Lack of Shot-State Comparison</span>
            <p>The paper only analyzes single-image states and lacks a temporal before/after comparator logic to notice displaced balls.</p>
          </div>
          <div className="p-4 rounded-xl bg-zinc-950/30 border border-white/5 text-xs text-zinc-400 space-y-2">
            <span className="text-red-400 font-semibold block">2. No Foul-Aware Restoration Workflow</span>
            <p>No mechanism is designed to handle fouls, track displacements, or generate targets to restore table configurations.</p>
          </div>
          <div className="p-4 rounded-xl bg-zinc-950/30 border border-white/5 text-xs text-zinc-400 space-y-2">
            <span className="text-red-400 font-semibold block">3. No Multiple-Red Ball Spatial Matching</span>
            <p>In snooker, multiple red balls are identical. The paper provides no identity matching algorithms to resolve red ball swapping during displacement.</p>
          </div>
          <div className="p-4 rounded-xl bg-zinc-950/30 border border-white/5 text-xs text-zinc-400 space-y-2">
            <span className="text-red-400 font-semibold block">4. No Integrated Physical Laser Positioning</span>
            <p>While the paper outputs coordinate data to actuate a massive 6-axis robotic arm, it does not explore portable, low-cost laser pointers for referee guide projection.</p>
          </div>
        </div>
      </div>

      {/* ── PIPELINE COMPARISON (EXTENSION DIAGRAM) ── */}
      <div className="p-6 rounded-2xl glass space-y-6">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono">
          How Our Project Extends the Base Paper
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
          {/* Base Paper column */}
          <div className="p-5 rounded-xl bg-zinc-900/40 border border-white/5 space-y-4">
            <span className="text-xs font-bold text-zinc-400 uppercase font-mono tracking-wider block">Base Paper Pipeline</span>
            <div className="flex flex-col gap-2 text-xs">
              <div className="p-2.5 rounded bg-zinc-950 border border-white/5 font-mono text-center">Deep-Learning Ball Detection</div>
              <div className="flex justify-center text-zinc-600">↓</div>
              <div className="p-2.5 rounded bg-zinc-950 border border-white/5 font-mono text-center">Ball Identification / Localization</div>
            </div>
          </div>

          {/* Our Project column */}
          <div className="p-5 rounded-xl bg-red-950/10 border border-red-500/10 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-red-400 uppercase font-mono tracking-wider block">Our Project Extended Pipeline</span>
              <span className="px-2 py-0.5 rounded-full bg-red-500/10 text-[9px] text-red-400 border border-red-500/20 font-mono uppercase font-semibold">Active Snooker Engine</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">YOLOv8 Snooker Detection</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">8-Class Ball Identification</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">Table Calibration</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">Before/After State Comparison</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">Foul Analysis</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">Displaced Ball Detection</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">Referee Decision Support</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">Restoration Queue</div>
              <div className="p-2 rounded bg-zinc-950 border border-white/5 font-mono text-center">AI-Guided Virtual Laser</div>
              <div className="p-2 rounded bg-red-900/30 border border-red-500/30 text-red-300 font-mono text-center font-semibold">Physical Laser / Servo Output</div>
            </div>
          </div>
        </div>
      </div>

      {/* ── ACADEMIC COMPARISON TABLE ── */}
      <div className="p-6 rounded-2xl glass space-y-4">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono">
          System Comparison Matrix
        </h3>
        <div className="overflow-x-auto border border-white/5 rounded-xl bg-zinc-950/20">
          <table className="min-w-full divide-y divide-white/5 text-xs">
            <thead>
              <tr className="bg-zinc-950/40 text-white font-mono">
                <th className="px-4 py-3 text-left">Feature</th>
                <th className="px-4 py-3 text-center">Base Paper</th>
                <th className="px-4 py-3 text-center">Our Project</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-zinc-400">
              {comparisonFeatures.map((item, idx) => (
                <tr key={idx} className="hover:bg-white/[0.01]">
                  <td className="px-4 py-3 font-medium">
                    <div>{item.name}</div>
                    {item.note && <div className="text-[10px] text-zinc-500 font-normal mt-0.5">{item.note}</div>}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {item.base ? (
                      <Check size={16} className="text-emerald-500 mx-auto" />
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {item.ours ? (
                      <Check size={16} className="text-emerald-500 mx-auto" />
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] text-zinc-500 font-mono">
          Note: ✓ = Supported / Reported. — = Not addressed / outside reported scope.
        </p>
      </div>

      {/* ── RESEARCH GAP -> SOLUTION ── */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono">
          Research Gap → Proposed Solution
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-2xl glass border-l-4 border-red-500 space-y-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-red-400 font-mono block">Gap A</span>
            <p className="text-xs text-zinc-300 font-medium">Ball detection without temporal state comparison or restoration</p>
            <div className="text-[10px] text-zinc-500 font-mono">↓</div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono block">Proposed Solution</span>
            <p className="text-xs text-emerald-400 font-medium">Before/After State Comparator &amp; Hungarian Matching</p>
          </div>

          <div className="p-5 rounded-2xl glass border-l-4 border-red-500 space-y-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-red-400 font-mono block">Gap B</span>
            <p className="text-xs text-zinc-300 font-medium">No physical coordinates projection onto the billiard table surface</p>
            <div className="text-[10px] text-zinc-500 font-mono">↓</div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono block">Proposed Solution</span>
            <p className="text-xs text-emerald-400 font-medium">Checkerboard Homography Mapping + Laser Target Coordinates Generator</p>
          </div>

          <div className="p-5 rounded-2xl glass border-l-4 border-red-500 space-y-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-red-400 font-mono block">Gap C</span>
            <p className="text-xs text-zinc-300 font-medium">Lack of interactive referee-in-the-loop decision controls</p>
            <div className="text-[10px] text-zinc-500 font-mono">↓</div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono block">Proposed Solution</span>
            <p className="text-xs text-emerald-400 font-medium">Interactive Referee Dashboard &amp; Multi-ball Restoration Queue</p>
          </div>
        </div>
      </div>

      {/* ── PROJECT NOVELTY ── */}
      <div className="p-6 rounded-2xl glass border-l-2 border-emerald-500 space-y-3">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-emerald-400 font-mono">
          Project Novelty Statement
        </h3>
        <p className="text-xs text-zinc-300 leading-relaxed font-sans">
          "The proposed system extends vision-based billiards analysis into an AI-assisted snooker referee and restoration framework by integrating calibrated multi-ball detection, shot-state comparison, foul analysis, displacement estimation, referee-controlled restoration, and laser-guided repositioning within a unified system."
        </p>
        <p className="text-[10px] text-zinc-500 font-mono">
          Novelty lies in the integration and coordination of the complete cyber-physical loop (Computer Vision → Decision Framework → Mechatronic Output) tailored for competitive snooker.
        </p>
      </div>

      {/* ── CURRENT SYSTEM EVIDENCE (OUR MODEL METRICS) ── */}
      <div className="p-6 rounded-2xl glass space-y-6">
        <div className="flex items-center justify-between border-b border-white/5 pb-4">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono flex items-center gap-2">
            <TrendingUp size={16} className="text-red-500" />
            Our Model / Experimental Results
          </h3>
          <span className="px-2.5 py-1 rounded-full bg-red-500/10 text-[10px] text-red-400 border border-red-500/20 font-mono font-semibold uppercase">
            Verified Dataset v2.0
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div className="space-y-1">
            <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Snooker Dataset</span>
            <span className="text-xl font-bold text-white font-mono block">10,002</span>
            <span className="text-[10px] text-zinc-500 font-mono block">Images</span>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Objects Annotated</span>
            <span className="text-xl font-bold text-white font-mono block">60,582</span>
            <span className="text-[10px] text-zinc-500 font-mono block">Labeled Bounding Boxes</span>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Classes Trained</span>
            <span className="text-xl font-bold text-white font-mono block">8</span>
            <span className="text-[10px] text-zinc-500 font-mono block">Snooker Ball Colors</span>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] uppercase text-zinc-500 font-mono font-bold block">Model Architecture</span>
            <span className="text-xl font-bold text-red-400 font-mono block">YOLOv8</span>
            <span className="text-[10px] text-zinc-500 font-mono block">Nano Model</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-4 border-t border-white/5 text-center">
          <div className="p-3 bg-zinc-950/40 rounded-xl border border-white/5 space-y-0.5">
            <span className="text-[9px] uppercase text-zinc-500 font-mono block">Test Precision</span>
            <span className="text-base font-bold text-white font-mono">50.39%</span>
          </div>
          <div className="p-3 bg-zinc-950/40 rounded-xl border border-white/5 space-y-0.5">
            <span className="text-[9px] uppercase text-zinc-500 font-mono block">Test Recall</span>
            <span className="text-base font-bold text-white font-mono">94.87%</span>
          </div>
          <div className="p-3 bg-zinc-950/40 rounded-xl border border-white/5 space-y-0.5">
            <span className="text-[9px] uppercase text-zinc-500 font-mono block">Test mAP50</span>
            <span className="text-base font-bold text-white font-mono">62.53%</span>
          </div>
          <div className="p-3 bg-zinc-950/40 rounded-xl border border-white/5 space-y-0.5">
            <span className="text-[9px] uppercase text-zinc-500 font-mono block">Test mAP50-95</span>
            <span className="text-base font-bold text-white font-mono">46.63%</span>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-zinc-950/40 border border-white/5 flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-2">
            <Cpu size={14} className="text-emerald-500" />
            <span>Average Inference Latency:</span>
          </div>
          <div className="space-x-3">
            <span className="text-emerald-400 font-bold">18.21 ms / image</span>
            <span className="text-zinc-600">|</span>
            <span className="text-emerald-400 font-bold">54.9 FPS</span>
          </div>
        </div>
      </div>

      {/* ── PHYSICAL SYSTEM SETUP ── */}
      <div className="p-6 rounded-2xl glass space-y-6">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-zinc-400 font-mono flex items-center gap-2">
          <Layers size={16} className="text-red-500" />
          Proposed Physical System
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          {/* Hardware block diagram */}
          <div className="flex flex-col gap-2.5 text-xs text-white max-w-sm mx-auto w-full">
            <div className="p-2 rounded bg-zinc-900 border border-white/5 text-center font-mono">Overhead wall-mounted camera</div>
            <div className="text-center text-zinc-600">↓</div>
            <div className="p-2 rounded bg-zinc-900 border border-white/5 text-center font-mono">Snooker Table Play Area</div>
            <div className="text-center text-zinc-600">↓</div>
            <div className="p-2 rounded bg-zinc-900 border border-white/5 text-center font-mono">FastAPI Ball Detection &amp; Homography</div>
            <div className="text-center text-zinc-600">↓</div>
            <div className="p-2 rounded bg-zinc-900 border border-white/5 text-center font-mono">Restoration Target Coordinates Generation</div>
            <div className="text-center text-zinc-600">↓</div>
            <div className="p-2 rounded bg-zinc-900 border border-white/5 text-center font-mono">ESP32 Controller Inverse Kinematics</div>
            <div className="text-center text-zinc-600">↓</div>
            <div className="p-2 rounded bg-red-950/40 border border-red-500/20 text-red-400 text-center font-mono font-semibold">Dual Servo (Yaw + Pitch) &amp; Laser Diodes</div>
          </div>

          <div className="space-y-4 text-xs text-zinc-400 leading-relaxed">
            <span className="text-[10px] uppercase font-mono font-bold text-white block">Calibration &amp; Target Mapping</span>
            <p>
              The wall-mounted USB camera captures a full view of the snooker table. The checkerboard camera calibration removes lens barrel distortion, and the planar homography maps bounding box centers to real-world table coordinates in millimeters.
            </p>
            <p>
              When a ball's coordinates require physical placement, the target millimeter coordinates are sent over serial/Wi-Fi to the ESP32. The ESP32 calculates motor angles via inverse kinematics and drives the mechatronic module to align the laser spot, guiding the referee with sub-centimeter accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
