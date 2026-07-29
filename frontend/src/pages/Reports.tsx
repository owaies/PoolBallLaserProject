import { Download } from "lucide-react";
import { toast } from "react-hot-toast";

export default function Reports() {
  const reports = [
    {
      title: "Model Training Summary Report",
      description: "Compiles training metrics across 50 epochs, showing loss convergences, precision, recall (93.5%), and mAP@50 (96.5%).",
      filename: "model_report.md",
      type: "Markdown Document",
      size: "2.3 KB",
      content: "# YOLOv8 Training Results\n- Precision: 94.1%\n- Recall: 93.5%\n- mAP@0.5: 96.5%\n- Convergence: Epoch 49 peak."
    },
    {
      title: "Offline Detection Test Report",
      description: "Lists detection speeds on test splits, highlighting inference latency (5.3ms on GPU) and false positive frequencies.",
      filename: "detection_report.md",
      type: "Markdown Document",
      size: "1.8 KB",
      content: "# YOLO Inference Performance\n- GPU Latency: 5.3ms\n- CPU Latency: 18.5ms\n- Target counts validated on raw images."
    },
    {
      title: "Camera Calibration Log",
      description: "Detailed parameters generated from chess-board grid captures, showing reprojection error evaluations and matrix configurations.",
      filename: "calibration_report.md",
      type: "Markdown Document",
      size: "1.2 KB",
      content: "# Camera Intrinsic Parameters\n- Reprojection error: 0.28 pixels\n- Dimensions: 9x6 board."
    }
  ];

  const handleDownload = (report: typeof reports[0]) => {
    const blob = new Blob([report.content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = report.filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`${report.filename} downloaded successfully!`);
  };

  return (
    <div className="space-y-6 pb-12">
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {reports.map((report, idx) => (
          <div key={idx} className="p-6 rounded-3xl glass border border-white/5 flex flex-col justify-between h-[230px]">
            <div className="space-y-3">
              <span className="text-[9px] font-mono bg-blue-500/10 px-2 py-0.5 rounded text-blue-400 font-bold uppercase tracking-wider">
                {report.type}
              </span>
              <h4 className="font-semibold text-white text-sm tracking-tight leading-tight">{report.title}</h4>
              <p className="text-xs text-zinc-500 leading-normal">{report.description}</p>
            </div>
            
            <div className="flex items-center justify-between border-t border-white/5 pt-4 mt-2">
              <span className="text-[10px] text-zinc-600 font-mono">{report.size}</span>
              <button
                onClick={() => handleDownload(report)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 border border-white/5 hover:bg-zinc-800 text-xs font-semibold text-white transition"
              >
                <Download size={12} />
                Download
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
