import { Link } from "react-router-dom";
import { ArrowRight, LayoutDashboard } from "lucide-react";
import AppleImageSequence from "../components/AppleImageSequence";
import { useEffect, useRef } from "react";
import gsap from "gsap";

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);

  // Trigger floating hero entry animations
  useEffect(() => {
    if (!heroRef.current) return;
    
    gsap.fromTo(
      heroRef.current.querySelectorAll(".hero-animate"),
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, stagger: 0.2, ease: "power3.out" }
    );
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0c] select-none text-zinc-300">
      {/* 1. Apple-style Scroll-Driven Image Sequence */}
      <div className="relative">
        <AppleImageSequence
          framePath="/videoframes/ezgif-frame-"
          frameCount={240}
          fileExtension="jpg"
          padLength={3}
          scrollerSelector="#main-scroll-container"
          scrollMultiplier={5}
        />
        
        {/* Absolute floating Title overlay */}
        <div className="absolute top-1/4 left-10 md:left-24 max-w-lg z-10 pointer-events-none">
          <span className="text-blue-500 font-mono text-xs uppercase tracking-widest block mb-2 animate-pulse">
            Phase 6 Production Build
          </span>
          <h2 className="text-4xl md:text-6xl font-bold tracking-tight text-white leading-none uppercase font-mono">
            Pool Ball <br />
            <span className="text-zinc-500">Targeting</span>
          </h2>
          <p className="text-sm text-zinc-400 mt-4 leading-relaxed font-sans max-w-sm">
            Harnessing real-time computer vision and homography transformation to map pixel space directly to table coordinates.
          </p>
        </div>
      </div>

      {/* 2. Content Sections Grid */}
      <section ref={heroRef} className="py-24 px-6 md:px-12 max-w-6xl mx-auto space-y-24">
        
        {/* Intro Grid */}
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <h3 className="text-3xl font-bold text-white tracking-tight hero-animate">
              A Complete Artificial Intelligence Pipeline
            </h3>
            <p className="text-zinc-400 leading-relaxed font-sans text-sm hero-animate">
              The AI-Based Pool Ball Identification and Laser Positioning System uses a overhead camera feed to locate pool balls, computes sub-pixel coordinates, undistorts perspective offsets, and prepares physical outputs.
            </p>
            <div className="flex flex-wrap gap-4 hero-animate">
              <Link
                to="/detection"
                className="flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm transition-all duration-200"
              >
                Start Detection
                <ArrowRight size={16} />
              </Link>
              <Link
                to="/dashboard"
                className="flex items-center gap-2 px-6 py-3 rounded-lg bg-zinc-900 border border-white/5 hover:bg-zinc-800 text-zinc-200 font-medium text-sm transition-all duration-200"
              >
                View Dashboard
                <LayoutDashboard size={16} />
              </Link>
            </div>
          </div>
          
          {/* Card Features List */}
          <div className="grid sm:grid-cols-2 gap-4 hero-animate">
            <div className="p-6 rounded-2xl glass space-y-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500 font-bold font-mono text-xs">01</div>
              <h4 className="font-semibold text-white text-sm">YOLOv8 Detection</h4>
              <p className="text-xs text-zinc-500 leading-relaxed">Runs real-time inference on 16 ball classes (solids, stripes, cue) with 96% mAP.</p>
            </div>
            <div className="p-6 rounded-2xl glass space-y-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500 font-bold font-mono text-xs">02</div>
              <h4 className="font-semibold text-white text-sm">Geometric Undistortion</h4>
              <p className="text-xs text-zinc-500 leading-relaxed">Undistorts lens skew using calibration matrices to achieve absolute accuracy.</p>
            </div>
            <div className="p-6 rounded-2xl glass space-y-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500 font-bold font-mono text-xs">01</div>
              <h4 className="font-semibold text-white text-sm">Homography Transform</h4>
              <p className="text-xs text-zinc-500 leading-relaxed">Transforms pixel values to real table coordinates (mm) via perspective mapping.</p>
            </div>
            <div className="p-6 rounded-2xl glass space-y-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500 font-bold font-mono text-xs">04</div>
              <h4 className="font-semibold text-white text-sm">REST API Integration</h4>
              <p className="text-xs text-zinc-500 leading-relaxed">Provides high-performance, validated HTTP endpoints built on FastAPI.</p>
            </div>
          </div>
        </div>

        {/* Pipeline Diagram */}
        <div className="p-8 rounded-3xl glass border border-white/5 space-y-8 hero-animate">
          <div className="text-center max-w-md mx-auto space-y-2">
            <span className="text-[10px] uppercase font-mono text-blue-500 font-bold tracking-widest">Architectural Flow</span>
            <h3 className="text-2xl font-bold text-white">System Data Pipeline</h3>
            <p className="text-xs text-zinc-500">Tracing how raw light coordinates are mapped into millimetric targeting positions.</p>
          </div>
          
          <div className="grid sm:grid-cols-4 gap-4 items-center">
            <div className="p-4 rounded-xl bg-zinc-950/50 border border-white/5 text-center space-y-2">
              <span className="text-xs font-mono text-zinc-500">1. Image Acquisition</span>
              <p className="text-[11px] text-zinc-600">Overhead camera grabs the pool table frame.</p>
            </div>
            <div className="hidden sm:block text-center text-zinc-700">&#10132;</div>
            <div className="p-4 rounded-xl bg-zinc-950/50 border border-white/5 text-center space-y-2">
              <span className="text-xs font-mono text-zinc-500">2. YOLOv8 Detections</span>
              <p className="text-[11px] text-zinc-600">AI locates the balls and extracts centers.</p>
            </div>
            <div className="hidden sm:block text-center text-zinc-700">&#10132;</div>
            <div className="p-4 rounded-xl bg-zinc-950/50 border border-white/5 text-center space-y-2">
              <span className="text-xs font-mono text-zinc-500">3. Camera Undistort</span>
              <p className="text-[11px] text-zinc-600">Removes lens curvature distortion.</p>
            </div>
            <div className="hidden sm:block text-center text-zinc-700">&#10132;</div>
            <div className="p-4 rounded-xl bg-zinc-950/50 border border-white/5 text-center space-y-2">
              <span className="text-xs font-mono text-zinc-500">4. Homography Map</span>
              <p className="text-[11px] text-zinc-600">Converts pixel coordinates to millimeters.</p>
            </div>
          </div>
        </div>

      </section>
    </div>
  );
}
