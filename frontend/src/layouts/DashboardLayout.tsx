import { useState, useEffect } from "react";
import { Link, useLocation, Outlet } from "react-router-dom";
import {
  Camera,
  Gamepad2,
  FileText,
  Compass,
  Settings,
  Info,
  Menu,
  X,
  Home,
  Cpu,
} from "lucide-react";
import { apiService, getBackendUrl } from "../services/api";
import type { HealthResponse } from "../types";

export default function DashboardLayout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [backendHealth, setBackendHealth] = useState<HealthResponse | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  // Poll backend health status
  useEffect(() => {
    const checkHealth = async () => {
      const startTime = performance.now();
      try {
        const health = await apiService.getHealth();
        const endTime = performance.now();
        setBackendHealth(health);
        setLatency(Math.round(endTime - startTime));
      } catch (err) {
        setBackendHealth(null);
        setLatency(null);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, [location.pathname]);

  // Snooker-only navigation items (Pool items hidden from nav but routes still exist)
  const menuItems = [
    { name: "Home", path: "/", icon: Home },
    { name: "Snooker Photo Analysis", path: "/snooker/photo", icon: Camera },
    { name: "Snooker Simulation", path: "/snooker/simulation", icon: Gamepad2 },
    { name: "📄 Base Paper", path: "/snooker/base-paper", icon: FileText },
    { name: "Reports", path: "/reports", icon: FileText },
    { name: "Calibration", path: "/calibration", icon: Compass },
    { name: "Settings", path: "/settings", icon: Settings },
    { name: "About", path: "/about", icon: Info },
  ];

  return (
    <div className="flex h-screen bg-[#0a0a0c] overflow-hidden text-gray-200">
      {/* 1. Mobile Sidebar Backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* 2. Sidebar Component */}
      <aside
        className={`fixed md:relative inset-y-0 left-0 z-50 flex flex-col w-64 bg-zinc-950/85 backdrop-blur-xl border-r border-white/5 transition-transform duration-300 md:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Sidebar Header — Snooker Branding */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-white/5">
          <Link to="/" className="flex items-center gap-2.5" onClick={() => setSidebarOpen(false)}>
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-red-700 to-red-900 flex items-center justify-center shadow-lg shadow-red-900/40 text-sm select-none">
              🎱
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-xs tracking-wider text-white uppercase font-mono leading-none">
                Snooker Referee
              </span>
              <span className="text-[9px] text-zinc-500 font-mono">AI Restoration System</span>
            </div>
          </Link>
          <button className="md:hidden p-1 text-zinc-400 hover:text-white" onClick={() => setSidebarOpen(false)}>
            <X size={20} />
          </button>
        </div>

        {/* Sidebar Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-0.5 overflow-y-auto">
          {/* Snooker section label */}
          <p className="text-[9px] uppercase tracking-widest text-zinc-600 font-mono font-bold px-4 pb-2 pt-1">
            SNOOKER
          </p>
          {menuItems.slice(1, 4).map((item) => {
            const isActive = location.pathname === item.path || location.pathname.startsWith(item.path.split("/").slice(0, 3).join("/"));
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-red-600/15 text-red-400 border-l-2 border-red-500 font-semibold"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-900/50"
                }`}
              >
                <Icon size={16} />
                <span className="text-[13px]">{item.name}</span>
              </Link>
            );
          })}

          {/* System section label */}
          <p className="text-[9px] uppercase tracking-widest text-zinc-600 font-mono font-bold px-4 pb-2 pt-4">
            SYSTEM
          </p>
          {[menuItems[0], ...menuItems.slice(4)].map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-red-600/15 text-red-400 border-l-2 border-red-500 font-semibold"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-900/50"
                }`}
              >
                <Icon size={16} />
                <span className="text-[13px]">{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-white/5 bg-zinc-900/10 flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>API Status</span>
            {backendHealth?.status === "healthy" ? (
              <span className="flex items-center gap-1.5 text-emerald-500 font-semibold">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                Online
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-rose-500 font-semibold">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
                Offline
              </span>
            )}
          </div>
          {backendHealth?.status === "healthy" && (
            <div className="text-[10px] text-zinc-600 space-y-0.5">
              <div className="flex justify-between">
                <span>Latency:</span>
                <span className="text-zinc-400 font-mono">{latency}ms</span>
              </div>
              <div className="flex justify-between truncate">
                <span>Model:</span>
                <span className="text-zinc-400 font-mono" title={backendHealth.current_model || "None"}>
                  {backendHealth.current_model ? backendHealth.current_model.split("/").pop() : "None"}
                </span>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* 3. Main Frame Content Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Top Navbar */}
        <header className="flex items-center justify-between h-14 px-6 bg-zinc-950/40 backdrop-blur-md border-b border-white/5 z-30">
          <div className="flex items-center gap-4">
            <button className="md:hidden p-2 text-zinc-400 hover:text-white" onClick={() => setSidebarOpen(true)}>
              <Menu size={22} />
            </button>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold text-white tracking-tight">
                {menuItems.find((item) => location.pathname === item.path || location.pathname.startsWith(item.path + "/"))?.name || "Snooker Referee System"}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            {/* GPU Status Info */}
            {backendHealth?.gpu_available && (
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20 font-semibold">
                <Cpu size={12} />
                CUDA Acceleration Active
              </div>
            )}

            {/* Target Address Info */}
            <div className="hidden md:block text-zinc-500">
              Host: <span className="text-zinc-400 font-mono">{getBackendUrl()}</span>
            </div>
          </div>
        </header>

        {/* Dynamic Route Outlet */}
        <main id="main-scroll-container" className={`flex-1 overflow-y-auto bg-zinc-950/20 ${location.pathname === '/' ? '' : 'p-6'}`}>
          {location.pathname === '/' ? (
            <Outlet />
          ) : (
            <div className="max-w-6xl mx-auto w-full">
              <Outlet />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
