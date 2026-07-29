import { useState, useEffect } from "react";
import { Terminal, RefreshCw, Search, Filter } from "lucide-react";
import { apiService } from "../services/api";

export default function Logs() {
  const [logs, setLogs] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState("ALL");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);

  const fetchLogs = async () => {
    try {
      const data = await apiService.getLogs(150); // Get latest 150 lines
      setLogs(data);
    } catch (err) {
      console.error("Failed to load logs", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  // Handle auto-refresh interval
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 3000); // refresh every 3s
    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Filter & Search log matching
  const filteredLogs = logs.filter((log) => {
    const matchesSearch = log.toLowerCase().includes(search.toLowerCase());
    
    if (levelFilter === "ALL") return matchesSearch;
    
    // Check level tags
    if (levelFilter === "ERROR" && log.toUpperCase().includes("ERROR")) return matchesSearch;
    if (levelFilter === "WARNING" && log.toUpperCase().includes("WARNING")) return matchesSearch;
    if (levelFilter === "INFO" && log.toUpperCase().includes("INFO")) return matchesSearch;
    
    return false;
  });

  return (
    <div className="space-y-6 pb-12">
      {/* Search and Filters toolbar */}
      <div className="p-4 rounded-2xl glass border border-white/5 flex flex-wrap gap-4 items-center justify-between">
        <div className="flex flex-wrap gap-3 flex-1">
          {/* Search bar */}
          <div className="relative max-w-xs flex-1">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search logs..."
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500"
            />
            <Search className="absolute left-3 top-3 text-zinc-600" size={13} />
          </div>

          {/* Log level Filter */}
          <div className="relative">
            <select
              value={levelFilter}
              onChange={(e) => setLevelFilter(e.target.value)}
              className="pl-9 pr-8 py-2 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 appearance-none font-mono"
            >
              <option value="ALL">ALL LEVELS</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
            </select>
            <Filter className="absolute left-3 top-3.5 text-zinc-600" size={13} />
          </div>
        </div>

        {/* Auto Refresh switch */}
        <div className="flex items-center gap-4 text-xs">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-white/5 bg-zinc-950 text-blue-500 focus:ring-0 focus:ring-offset-0"
            />
            <span className="text-zinc-500">Auto Refresh (3s)</span>
          </label>

          <button
            onClick={fetchLogs}
            disabled={loading}
            className="p-2 rounded-lg bg-zinc-900 border border-white/5 hover:bg-zinc-800 text-zinc-400 hover:text-white transition"
          >
            <RefreshCw className={loading ? "animate-spin" : ""} size={14} />
          </button>
        </div>
      </div>

      {/* Console output display box */}
      <div className="p-6 rounded-3xl bg-zinc-950/80 border border-white/5 font-mono text-[11px] leading-relaxed text-zinc-400 overflow-hidden flex flex-col h-[500px]">
        <div className="flex items-center justify-between border-b border-white/5 pb-3 mb-4">
          <span className="text-zinc-500 flex items-center gap-1.5 uppercase tracking-wider text-[10px]">
            <Terminal size={12} />
            Terminal Console
          </span>
          <span className="text-zinc-600 text-[10px]">
            Showing {filteredLogs.length} / {logs.length} lines
          </span>
        </div>

        {/* Inner logs scrolling list */}
        <div className="flex-1 overflow-y-auto space-y-1.5 pr-2">
          {filteredLogs.length > 0 ? (
            filteredLogs.map((log, idx) => {
              let color = "text-zinc-400";
              if (log.toUpperCase().includes("ERROR")) color = "text-rose-500 font-semibold";
              else if (log.toUpperCase().includes("WARNING")) color = "text-amber-500";
              else if (log.toUpperCase().includes("INFO")) color = "text-zinc-300";

              return (
                <div key={idx} className={`${color} whitespace-pre-wrap`}>
                  {log}
                </div>
              );
            })
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
              No matching log records found
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
