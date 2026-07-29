import { useState } from "react";
import { Settings as SettingsIcon, Save } from "lucide-react";
import { toast } from "react-hot-toast";

export default function Settings() {
  const [backendUrl, setBackendUrl] = useState(
    localStorage.getItem("backend_url") || "http://127.0.0.1:8000"
  );
  const [confThreshold, setConfThreshold] = useState(
    localStorage.getItem("confidence_threshold") || "0.25"
  );
  const [iouThreshold, setIouThreshold] = useState(
    localStorage.getItem("iou_threshold") || "0.45"
  );
  const [theme, setTheme] = useState(
    localStorage.getItem("theme") || "dark"
  );
  const [language, setLanguage] = useState(
    localStorage.getItem("language") || "en"
  );

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();

    localStorage.setItem("backend_url", backendUrl);
    localStorage.setItem("confidence_threshold", confThreshold);
    localStorage.setItem("iou_threshold", iouThreshold);
    localStorage.setItem("theme", theme);
    localStorage.setItem("language", language);

    toast.success("Preferences updated and saved locally!", { id: "settings-save" });
  };

  return (
    <div className="space-y-6 pb-12 max-w-xl">
      <div className="p-6 rounded-3xl glass border border-white/5 space-y-6">
        <div className="flex items-center gap-2 border-b border-white/5 pb-4">
          <SettingsIcon className="text-blue-500" size={18} />
          <h3 className="font-semibold text-white text-sm">System Configuration</h3>
        </div>

        <form onSubmit={handleSave} className="space-y-5">
          {/* Backend API Endpoint */}
          <div className="space-y-2">
            <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
              Backend REST Server URL
            </label>
            <input
              type="url"
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Confidence Threshold */}
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Confidence Threshold (0.01 - 1.0)
              </label>
              <input
                type="number"
                step="0.05"
                min="0.01"
                max="1.0"
                value={confThreshold}
                onChange={(e) => setConfThreshold(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>

            {/* IoU Threshold */}
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                IoU NMS Threshold
              </label>
              <input
                type="number"
                step="0.05"
                min="0.01"
                max="1.0"
                value={iouThreshold}
                onChange={(e) => setIouThreshold(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Visual Theme Selection */}
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Color theme
              </label>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500"
              >
                <option value="dark">Professional Dark (Recommended)</option>
                <option value="light">Carbon Light</option>
              </select>
            </div>

            {/* Local Language */}
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Language
              </label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500"
              >
                <option value="en">English (US)</option>
                <option value="de">Deutsch</option>
                <option value="es">Español</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            className="w-full flex items-center justify-center gap-2 py-3 mt-4 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm transition-all"
          >
            <Save size={14} />
            Save Preferences
          </button>
        </form>
      </div>
    </div>
  );
}
