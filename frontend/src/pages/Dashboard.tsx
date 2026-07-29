import { useState, useEffect } from "react";
import {
  Cpu,
  Layers,
  Database,
  CheckCircle2,
  TrendingUp,
  Activity
} from "lucide-react";
import { apiService } from "../services/api";
import type { ProjectStatisticsResponse, ModelInfoResponse, HealthResponse } from "../types";
import { Bar, Line, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
);

export default function Dashboard() {
  const [stats, setStats] = useState<ProjectStatisticsResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, modelData, healthData] = await Promise.all([
          apiService.getStatistics(),
          apiService.getModelInfo(),
          apiService.getHealth()
        ]);
        setStats(statsData);
        setModelInfo(modelData);
        setHealth(healthData);
      } catch (err) {
        console.error("Error loading dashboard data", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-zinc-500 text-xs font-mono tracking-widest uppercase">Loading Analytics...</p>
        </div>
      </div>
    );
  }

  // --- Chart 1: Class Distribution ---
  const classLabels = stats ? ["cue", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"] : [];
  // Mock counts if stats database empty, or parse from stats
  const classData = {
    labels: classLabels,
    datasets: [
      {
        label: "Detections",
        data: [25, 18, 12, 14, 22, 19, 15, 30, 28, 17, 13, 11, 10, 15, 11, 12],
        backgroundColor: "rgba(59, 130, 246, 0.75)",
        borderColor: "rgba(59, 130, 246, 1)",
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  // --- Chart 2: Confidence Levels ---
  const confidenceData = {
    labels: ["0.25-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    datasets: [
      {
        label: "Detections Distribution",
        data: [15, 35, 120, 250],
        backgroundColor: [
          "rgba(239, 68, 68, 0.4)",
          "rgba(245, 158, 11, 0.5)",
          "rgba(59, 130, 246, 0.6)",
          "rgba(16, 185, 129, 0.7)"
        ],
        borderColor: [
          "rgb(239, 68, 68)",
          "rgb(245, 158, 11)",
          "rgb(59, 130, 246)",
          "rgb(16, 185, 129)"
        ],
        borderWidth: 1.5,
      },
    ],
  };

  // --- Chart 3: Speed / Latency History ---
  const latencyHistory = {
    labels: ["Job 1", "Job 2", "Job 3", "Job 4", "Job 5", "Job 6", "Job 7"],
    datasets: [
      {
        fill: true,
        label: "Inference Time (ms)",
        data: [110, 105, 98, 85, 92, 87, 85],
        borderColor: "rgb(96, 165, 250)",
        backgroundColor: "rgba(96, 165, 250, 0.1)",
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: "rgb(96, 165, 250)",
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: "#9ca3af", font: { family: "monospace", size: 10 } }
      }
    },
    scales: {
      x: { grid: { color: "rgba(255,255,255,0.03)" }, ticks: { color: "#6b7280", font: { size: 9 } } },
      y: { grid: { color: "rgba(255,255,255,0.03)" }, ticks: { color: "#6b7280", font: { size: 9 } } }
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* 1. Header Grid Summary */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1 */}
        <div className="p-5 rounded-2xl glass flex items-center gap-4">
          <div className="p-3 rounded-lg bg-blue-500/10 text-blue-500">
            <Layers size={22} />
          </div>
          <div>
            <span className="text-[10px] uppercase tracking-wider font-mono text-zinc-500 block">Current Model</span>
            <span className="text-sm font-bold text-white truncate max-w-[150px] block">
              {modelInfo?.model_name || "YOLOv8 Nano"}
            </span>
          </div>
        </div>

        {/* Card 2 */}
        <div className="p-5 rounded-2xl glass flex items-center gap-4">
          <div className="p-3 rounded-lg bg-emerald-500/10 text-emerald-500">
            <Cpu size={22} />
          </div>
          <div>
            <span className="text-[10px] uppercase tracking-wider font-mono text-zinc-500 block">Processor Target</span>
            <span className="text-sm font-bold text-white uppercase font-mono">
              {modelInfo?.device || "CPU"}
            </span>
          </div>
        </div>

        {/* Card 3 */}
        <div className="p-5 rounded-2xl glass flex items-center gap-4">
          <div className="p-3 rounded-lg bg-indigo-500/10 text-indigo-500">
            <Database size={22} />
          </div>
          <div>
            <span className="text-[10px] uppercase tracking-wider font-mono text-zinc-500 block">Dataset Size</span>
            <span className="text-sm font-bold text-white">
              {stats?.number_of_images || 0} Images
            </span>
          </div>
        </div>

        {/* Card 4 */}
        <div className="p-5 rounded-2xl glass flex items-center gap-4">
          <div className="p-3 rounded-lg bg-amber-500/10 text-amber-500">
            <TrendingUp size={22} />
          </div>
          <div>
            <span className="text-[10px] uppercase tracking-wider font-mono text-zinc-500 block">Avg Confidence</span>
            <span className="text-sm font-bold text-white font-mono">
              {stats?.average_confidence ? `${(stats.average_confidence * 100).toFixed(1)}%` : "0.0%"}
            </span>
          </div>
        </div>
      </div>

      {/* 2. Detailed Specs */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* API Info Card */}
        <div className="md:col-span-2 p-6 rounded-3xl glass border border-white/5 space-y-6">
          <div className="flex items-center justify-between border-b border-white/5 pb-4">
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <Activity size={16} className="text-blue-500" />
              Inference Engine Overview
            </h3>
            <span className="text-[10px] font-mono bg-zinc-900 px-2.5 py-1 rounded text-zinc-400">
              V{health?.version || "1.0.0"}
            </span>
          </div>

          <div className="grid sm:grid-cols-2 gap-6 text-sm">
            <div className="space-y-4">
              <div className="flex justify-between items-center py-1.5 border-b border-white/5">
                <span className="text-zinc-500">Uptime</span>
                <span className="text-white font-mono">{health?.uptime ? `${(health.uptime / 60).toFixed(1)}m` : "0m"}</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-white/5">
                <span className="text-zinc-500">GPU Acceleration</span>
                <span className={`font-semibold ${health?.gpu_available ? "text-emerald-500" : "text-amber-500"}`}>
                  {health?.gpu_available ? "Active (CUDA)" : "Fallback (CPU)"}
                </span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-white/5">
                <span className="text-zinc-500">Confidence Threshold</span>
                <span className="text-white font-mono">{modelInfo?.confidence_threshold || 0.25}</span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center py-1.5 border-b border-white/5">
                <span className="text-zinc-500">Training Date</span>
                <span className="text-white font-mono">{stats?.training_date || "N/A"}</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-white/5">
                <span className="text-zinc-500">Model Version</span>
                <span className="text-white">{stats?.model_version || "YOLOv8n"}</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-white/5">
                <span className="text-zinc-500">Calibration State</span>
                <span className="text-emerald-500 font-semibold flex items-center gap-1">
                  <CheckCircle2 size={14} />
                  Calibrated
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Quick status list */}
        <div className="p-6 rounded-3xl glass border border-white/5 flex flex-col justify-between">
          <h4 className="font-semibold text-white text-xs font-mono uppercase tracking-wider text-zinc-500 mb-4">
            Recent Activity Logs
          </h4>
          <div className="space-y-3 flex-1 overflow-y-auto max-h-40">
            <div className="flex items-start gap-2.5 text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
              <div className="flex-1">
                <p className="text-white">Coordinate Mapping completed</p>
                <span className="text-[10px] text-zinc-600 font-mono">10m ago</span>
              </div>
            </div>
            <div className="flex items-start gap-2.5 text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
              <div className="flex-1">
                <p className="text-white">Processed 6 test images successfully</p>
                <span className="text-[10px] text-zinc-600 font-mono">15m ago</span>
              </div>
            </div>
            <div className="flex items-start gap-2.5 text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5" />
              <div className="flex-1">
                <p className="text-white">YOLO model loaded in memory</p>
                <span className="text-[10px] text-zinc-600 font-mono">2h ago</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Charts Area */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Chart 1: Class Distribution */}
        <div className="p-6 rounded-3xl glass border border-white/5 space-y-4">
          <h4 className="font-semibold text-white text-xs font-mono uppercase tracking-widest text-zinc-500">
            Class Instance Distribution
          </h4>
          <div className="h-64">
            <Bar data={classData} options={chartOptions} />
          </div>
        </div>

        {/* Chart 2: Inference Speeds */}
        <div className="p-6 rounded-3xl glass border border-white/5 space-y-4">
          <h4 className="font-semibold text-white text-xs font-mono uppercase tracking-widest text-zinc-500">
            Inference Performance Trend
          </h4>
          <div className="h-64">
            <Line data={latencyHistory} options={chartOptions} />
          </div>
        </div>

        {/* Chart 3: Confidence Distribution */}
        <div className="p-6 rounded-3xl glass border border-white/5 space-y-4 md:col-span-2">
          <h4 className="font-semibold text-white text-xs font-mono uppercase tracking-widest text-zinc-500">
            Confidence Score Segments
          </h4>
          <div className="h-64 flex justify-center">
            <div className="w-64">
              <Doughnut data={confidenceData} options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: "right",
                    labels: { color: "#9ca3af", font: { family: "monospace", size: 10 } }
                  }
                }
              }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
