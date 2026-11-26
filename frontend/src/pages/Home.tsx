import { Link } from "react-router-dom";
import { ArrowRight, Camera, Gamepad2, Shield, Zap, Target, CheckCircle } from "lucide-react";
import { useEffect, useRef } from "react";
import gsap from "gsap";

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!heroRef.current) return;
    gsap.fromTo(
      heroRef.current.querySelectorAll(".hero-animate"),
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, stagger: 0.15, ease: "power3.out" }
    );
  }, []);

  const workflow = [
    { step: "01", label: "Photo / Simulation", desc: "Capture real table photo or run virtual simulation" },
    { step: "02", label: "AI Detection", desc: "YOLOv8 identifies all balls with class & position" },
    { step: "03", label: "Foul Analysis", desc: "Rules engine evaluates first contact, potted balls, penalties" },
    { step: "04", label: "Referee Decision", desc: "Referee confirms, rejects, or applies override" },
    { step: "05", label: "Laser Restoration", desc: "AI-guided virtual laser targets each displaced ball" },
    { step: "06", label: "One Ball at a Time", desc: "Referee confirms each ball individually before movement" },
  ];

  const features = [
    {
      icon: Camera,
      title: "AI Photo Analysis",
      desc: "Upload real snooker table photos — YOLOv8 detects all 8 ball classes with spatial red-ball matching.",
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/20",
    },
    {
      icon: Gamepad2,
      title: "Playable Simulation",
      desc: "Full 22-ball 2D physics simulation with cue control, pocket detection, and 60 FPS rendering.",
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
    },
    {
      icon: Shield,
      title: "Referee Authority",
      desc: "Referee confirms or overrides AI foul decisions. Foul & Miss, continue, or request restoration.",
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20",
    },
    {
      icon: Target,
      title: "Laser Guidance",
      desc: "AI-guided virtual laser shows exact path from displaced ball to original position for physical referees.",
      color: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/20",
    },
    {
      icon: Zap,
      title: "Sequential Restoration",
      desc: "One ball at a time — laser shown, referee confirms, ball restored. Never moves multiple balls simultaneously.",
      color: "text-purple-400",
      bg: "bg-purple-500/10",
      border: "border-purple-500/20",
    },
    {
      icon: CheckCircle,
      title: "Shot History & Undo",
      desc: "Every shot is snapshotted. Full undo/redo of restoration decisions with complete shot history log.",
      color: "text-cyan-400",
      bg: "bg-cyan-500/10",
      border: "border-cyan-500/20",
    },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0c] select-none text-zinc-300 overflow-x-hidden">

      {/* ── HERO SECTION ── */}
      <div className="relative min-h-[80vh] flex items-center justify-center overflow-hidden">
        {/* Animated background radial glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-red-800/10 blur-3xl" />
          <div className="absolute top-2/3 left-1/4 w-[400px] h-[400px] rounded-full bg-emerald-900/8 blur-3xl" />
          <div className="absolute top-1/4 right-1/4 w-[300px] h-[300px] rounded-full bg-red-900/8 blur-3xl" />
        </div>

        {/* Grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: "linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />

        <div className="relative z-10 text-center px-6 max-w-4xl mx-auto space-y-8">
          <div className="space-y-3">
            <span className="inline-flex items-center gap-2 text-[10px] uppercase tracking-[0.3em] font-mono text-red-500 font-bold bg-red-500/10 border border-red-500/20 px-4 py-1.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
              Final Year Project — AI-Assisted Sports Officiating
            </span>
            <h1 className="text-5xl md:text-7xl font-black text-white tracking-tight leading-none">
              Snooker
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-red-700"> Referee</span>
              <br />
              <span className="text-3xl md:text-5xl text-zinc-400 font-bold">AI Restoration System</span>
            </h1>
            <p className="text-base text-zinc-400 max-w-2xl mx-auto leading-relaxed font-sans">
              AI-powered ball detection, foul analysis, and laser-guided one-ball-at-a-time restoration
              for professional snooker officiating. Built on YOLOv8 with a real-time 2D physics simulation.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link
              to="/snooker/photo"
              className="flex items-center gap-2.5 px-8 py-3.5 rounded-xl bg-red-600 hover:bg-red-700 text-white font-bold text-sm transition-all duration-200 shadow-lg shadow-red-900/40 hover:shadow-red-900/60"
            >
              <Camera size={18} />
              Photo Analysis
              <ArrowRight size={16} />
            </Link>
            <Link
              to="/snooker/simulation"
              className="flex items-center gap-2.5 px-8 py-3.5 rounded-xl bg-zinc-900 border border-white/10 hover:bg-zinc-800 text-zinc-200 font-bold text-sm transition-all duration-200"
            >
              <Gamepad2 size={18} />
              Snooker Simulation
            </Link>
          </div>
        </div>
      </div>

      {/* ── WORKFLOW SECTION ── */}
      <section ref={heroRef} className="py-20 px-6 md:px-12 max-w-6xl mx-auto space-y-20">

        <div className="space-y-10 hero-animate">
          <div className="text-center space-y-2">
            <span className="text-[10px] uppercase font-mono text-red-500 font-bold tracking-widest">Primary Workflow</span>
            <h2 className="text-3xl font-bold text-white">How It Works</h2>
            <p className="text-sm text-zinc-500 max-w-md mx-auto">
              From photo upload to laser-guided ball restoration — a complete AI referee workflow.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {workflow.map((w) => (
              <div key={w.step} className="p-5 rounded-2xl bg-zinc-950/60 border border-white/5 space-y-2 hover:border-red-500/20 transition-all">
                <div className="flex items-center gap-3">
                  <span className="text-2xl font-black text-red-700/40 font-mono">{w.step}</span>
                  <h3 className="font-bold text-white text-sm">{w.label}</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed pl-9">{w.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── FEATURES GRID ── */}
        <div className="space-y-10 hero-animate">
          <div className="text-center space-y-2">
            <span className="text-[10px] uppercase font-mono text-red-500 font-bold tracking-widest">System Capabilities</span>
            <h2 className="text-3xl font-bold text-white">What It Does</h2>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f) => {
              const Icon = f.icon;
              return (
                <div
                  key={f.title}
                  className={`p-6 rounded-2xl bg-zinc-950/60 border ${f.border} space-y-3 hover:bg-zinc-900/40 transition-all`}
                >
                  <div className={`w-9 h-9 rounded-xl ${f.bg} border ${f.border} flex items-center justify-center ${f.color}`}>
                    <Icon size={18} />
                  </div>
                  <h3 className="font-bold text-white text-sm">{f.title}</h3>
                  <p className="text-xs text-zinc-500 leading-relaxed">{f.desc}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── CTA ── */}
        <div className="hero-animate">
          <div className="p-10 rounded-3xl bg-gradient-to-br from-red-950/60 to-zinc-950/80 border border-red-900/30 text-center space-y-6">
            <div className="text-5xl">🎱</div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-white">Ready to referee?</h2>
              <p className="text-sm text-zinc-400">
                Upload a snooker photo or open the simulation to begin.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Link
                to="/snooker/photo"
                className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-red-600 hover:bg-red-700 text-white font-bold text-sm transition-all"
              >
                <Camera size={16} />
                📷 Photo Analysis
              </Link>
              <Link
                to="/snooker/simulation"
                className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-bold text-sm transition-all"
              >
                <Gamepad2 size={16} />
                🎱 Play Simulation
              </Link>
            </div>
          </div>
        </div>

      </section>
    </div>
  );
}
