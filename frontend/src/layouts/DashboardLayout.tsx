import { useState, useEffect } from "react";
import { Link, useLocation, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  Image,
  FolderOpen,
  Map,
  Compass,
  FileText,
  FileCode,
  Settings,
  Info,
  Menu,
  X,
  Home,
  Cpu
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
    const interval = setInterval(checkHealth, 5000); // Check every 5s
    return () => clearInterval(interval);
  }, [location.pathname]); // Recheck on route change to capture reload triggers

  const menuItems = [
    { name: "Home", path: "/", icon: Home },
    { name: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
    { name: "Image Detection", path: "/detection", icon: Image },
    { name: "Batch Detection", path: "/batch", icon: FolderOpen },
    { name: "Coordinate Mapping", path: "/mapping", icon: Map },
    { name: "Calibration", path: "/calibration", icon: Compass },
    { name: "Reports", path: "/reports", icon: FileText },
    { name: "Logs", path: "/logs", icon: FileCode },
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
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-white/5">
          <Link to="/" className="flex items-center gap-2" onClick={() => setSidebarOpen(false)}>
            <div className="w-6 h-6 rounded-md bg-blue-600 flex items-center justify-center font-bold text-xs text-white">
              P
            </div>
            <span className="font-semibold text-sm tracking-wide text-white uppercase font-mono">
              PoolBall Laser
            </span>
          </Link>
          <button className="md:hidden p-1 text-zinc-400 hover:text-white" onClick={() => setSidebarOpen(false)}>
            <X size={20} />
          </button>
        </div>

        {/* Sidebar Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-blue-600/10 text-blue-400 border-l-2 border-blue-500 font-semibold"
                    : "text-zinc-400 hover:text-white hover:bg-zinc-900/50"
                }`}
              >
                <Icon size={18} />
                {item.name}
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
        <header className="flex items-center justify-between h-16 px-6 bg-zinc-950/40 backdrop-blur-md border-b border-white/5 z-30">
          <div className="flex items-center gap-4">
            <button className="md:hidden p-2 text-zinc-400 hover:text-white" onClick={() => setSidebarOpen(true)}>
              <Menu size={22} />
            </button>
            <h1 className="text-base font-semibold text-white tracking-tight">
              {menuItems.find((item) => item.path === location.pathname)?.name || "System"}
            </h1>
          </div>

          <div className="flex items-center gap-4 text-xs">
            {/* GPU Status Info */}
            {backendHealth?.gpu_available && (
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold">
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
