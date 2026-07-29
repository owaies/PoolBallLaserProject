import { useState } from "react";
import { Folder, Play, Download, CheckCircle, FileText } from "lucide-react";
import { apiService, getBackendUrl } from "../services/api";
import { toast } from "react-hot-toast";
import type { FolderDetectionResponse } from "../types";

export default function Batch() {
  const [folderPath, setFolderPath] = useState("datasets/raw");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FolderDetectionResponse | null>(null);

  const handleRunBatch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!folderPath.trim()) return;

    setLoading(true);
    setResult(null);
    const toastId = toast.loading("Analyzing folder...");
    try {
      const data = await apiService.detectFolder(folderPath);
      setResult(data);
      toast.success("Batch detection completed successfully!", { id: toastId });
    } catch (err) {
      toast.error("Failed to run batch detection.", { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 pb-12">
      <div className="grid md:grid-cols-3 gap-6">
        
        {/* Settings Panel */}
        <div className="md:col-span-1 p-6 rounded-3xl glass border border-white/5 space-y-4 h-fit">
          <h3 className="font-semibold text-white text-sm">Configure Batch Run</h3>
          <form onSubmit={handleRunBatch} className="space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Target Folder Path
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder="e.g. datasets/raw"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
                />
                <Folder className="absolute left-3.5 top-3.5 text-zinc-600" size={14} />
              </div>
              <span className="text-[9px] text-zinc-600 block leading-normal">
                Absolute path or path relative to project root. For security, directory traversals (`../`) are blocked.
              </span>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-800 text-white font-medium text-sm transition-all"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <Play size={14} />
                  Run Batch Job
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="md:col-span-2 space-y-6">
          {result ? (
            <div className="p-6 rounded-3xl glass border border-white/5 space-y-6">
              
              {/* Header */}
              <div className="flex items-center gap-3 text-emerald-500 border-b border-white/5 pb-4">
                <CheckCircle size={20} />
                <h3 className="font-semibold text-white text-sm">Job Run Complete</h3>
              </div>

              {/* Counts metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-zinc-950/50 border border-white/5 text-center">
                  <span className="text-[10px] uppercase tracking-wider font-mono text-zinc-500 block">Images Found</span>
                  <span className="text-xl font-bold text-white">{result.total_images}</span>
                </div>
                <div className="p-4 rounded-xl bg-zinc-950/50 border border-white/5 text-center">
                  <span className="text-[10px] uppercase tracking-wider font-mono text-zinc-500 block">Detections</span>
                  <span className="text-xl font-bold text-white">{result.total_detections}</span>
                </div>
              </div>

              {/* Exports downloads */}
              <div className="space-y-3">
                <h4 className="font-semibold text-xs font-mono uppercase tracking-widest text-zinc-500">
                  Export Files
                </h4>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-3.5 rounded-xl bg-zinc-900/40 border border-white/5 text-xs">
                    <div className="flex items-center gap-3">
                      <FileText className="text-blue-500" size={16} />
                      <div>
                        <p className="text-white font-medium">Batch Summary CSV</p>
                        <span className="text-[10px] text-zinc-500 font-mono">{result.csv_path.split("/").pop()}</span>
                      </div>
                    </div>
                    <a
                      href={`${getBackendUrl()}/static/exports/${result.csv_path.split("/").pop()}`}
                      download
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 font-medium transition text-[11px]"
                    >
                      <Download size={12} />
                      CSV
                    </a>
                  </div>

                  <div className="flex items-center justify-between p-3.5 rounded-xl bg-zinc-900/40 border border-white/5 text-xs">
                    <div className="flex items-center gap-3">
                      <FileText className="text-indigo-500" size={16} />
                      <div>
                        <p className="text-white font-medium">Batch Details JSON</p>
                        <span className="text-[10px] text-zinc-500 font-mono">{result.json_path.split("/").pop()}</span>
                      </div>
                    </div>
                    <a
                      href={`${getBackendUrl()}/static/exports/${result.json_path.split("/").pop()}`}
                      download
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 font-medium transition text-[11px]"
                    >
                      <Download size={12} />
                      JSON
                    </a>
                  </div>
                </div>
              </div>

            </div>
          ) : (
            <div className="h-[250px] rounded-3xl border border-white/5 bg-zinc-950/10 flex items-center justify-center text-zinc-500 text-sm font-sans">
              Enter path and execute batch detection to view progress
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
