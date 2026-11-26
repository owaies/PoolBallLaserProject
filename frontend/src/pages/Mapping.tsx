import { useState, useRef, useEffect } from "react";
import { Map, ArrowRightLeft, Crosshair, HelpCircle } from "lucide-react";
import { apiService } from "../services/api";
import { toast } from "react-hot-toast";

export default function Mapping() {
  const [pixelX, setPixelX] = useState(400);
  const [pixelY, setPixelY] = useState(300);
  const [worldX, setWorldX] = useState<number | null>(null);
  const [worldY, setWorldY] = useState<number | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  // Mapped dimensions for visual canvas
  const TABLE_WIDTH = 700;
  const TABLE_HEIGHT = 400;

  const handleMap = async () => {
    try {
      const data = await apiService.mapCoordinates(pixelX, pixelY);
      setWorldX(data.world_x);
      setWorldY(data.world_y);
    } catch (err) {
      toast.error("Failed to map coordinates.");
    }
  };

  // Triggers recalculation when pixel inputs update
  useEffect(() => {
    handleMap();
  }, [pixelX, pixelY]);

  // Click on visual felt board to map
  const handleTableClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!tableRef.current) return;
    const rect = tableRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Scale local click to standard 800x600 image space for backend
    const scaleX = (clickX / rect.width) * 800;
    const scaleY = (clickY / rect.height) * 600;

    setPixelX(Math.round(scaleX));
    setPixelY(Math.round(scaleY));
  };

  return (
    <div className="space-y-6 pb-12">
      <div className="grid lg:grid-cols-3 gap-6">
        
        {/* Left Side: Parameters Inputs */}
        <div className="lg:col-span-1 p-6 rounded-3xl glass border border-white/5 space-y-6 h-fit">
          <div className="flex items-center gap-2 border-b border-white/5 pb-4">
            <Map className="text-blue-500" size={18} />
            <h3 className="font-semibold text-white text-sm">Coordinate Translator</h3>
          </div>

          <div className="space-y-4">
            {/* Input Pixel X */}
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Camera Pixel X (0 - 800)
              </label>
              <input
                type="number"
                min="0"
                max="800"
                value={pixelX}
                onChange={(e) => setPixelX(Number(e.target.value))}
                className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>

            {/* Input Pixel Y */}
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-mono tracking-wider text-zinc-500 block">
                Camera Pixel Y (0 - 600)
              </label>
              <input
                type="number"
                min="0"
                max="600"
                value={pixelY}
                onChange={(e) => setPixelY(Number(e.target.value))}
                className="w-full px-4 py-2.5 rounded-xl bg-zinc-950/50 border border-white/5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
          </div>

          {/* Translation Result output */}
          <div className="p-4 rounded-2xl bg-zinc-950/50 border border-white/5 space-y-3 font-mono">
            <span className="text-[10px] uppercase tracking-wider text-zinc-500 flex items-center gap-1.5">
              <ArrowRightLeft size={12} />
              Real Table Target
            </span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-zinc-600 block">World X:</span>
                <span className="text-white text-sm font-semibold">
                  {worldX !== null ? `${worldX.toFixed(1)} mm` : "Computing..."}
                </span>
              </div>
              <div>
                <span className="text-zinc-600 block">World Y:</span>
                <span className="text-white text-sm font-semibold">
                  {worldY !== null ? `${worldY.toFixed(1)} mm` : "Computing..."}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side: Virtual Interactive Felt Table */}
        <div className="lg:col-span-2 p-6 rounded-3xl glass border border-white/5 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <Crosshair size={16} className="text-blue-500 animate-pulse" />
              Interactive Plane Mapper
            </h3>
            <span className="text-[10px] font-mono text-zinc-500 flex items-center gap-1">
              <HelpCircle size={12} />
              Click inside green felt to map
            </span>
          </div>

          <div className="flex justify-center p-4">
            {/* Visual Pool Table felt representation */}
            <div
              ref={tableRef}
              onClick={handleTableClick}
              className="relative rounded-2xl border-4 border-amber-900 bg-emerald-800 shadow-inner cursor-crosshair overflow-hidden transition"
              style={{ width: `${TABLE_WIDTH}px`, height: `${TABLE_HEIGHT}px` }}
            >
              {/* Pocket holes */}
              <div className="absolute top-0 left-0 w-8 h-8 rounded-full bg-zinc-950 -mt-3.5 -ml-3.5 border border-zinc-900" />
              <div className="absolute top-0 left-1/2 w-8 h-8 rounded-full bg-zinc-950 -mt-3.5 -ml-4 border border-zinc-900" />
              <div className="absolute top-0 right-0 w-8 h-8 rounded-full bg-zinc-950 -mt-3.5 -mr-3.5 border border-zinc-900" />
              <div className="absolute bottom-0 left-0 w-8 h-8 rounded-full bg-zinc-950 -mb-3.5 -ml-3.5 border border-zinc-900" />
              <div className="absolute bottom-0 left-1/2 w-8 h-8 rounded-full bg-zinc-950 -mb-3.5 -ml-4 border border-zinc-900" />
              <div className="absolute bottom-0 right-0 w-8 h-8 rounded-full bg-zinc-950 -mb-3.5 -mr-3.5 border border-zinc-900" />

              {/* Grid Lines */}
              <div className="absolute inset-0 opacity-15 pointer-events-none" 
                   style={{ 
                     backgroundImage: "linear-gradient(to right, #fff 1px, transparent 1px), linear-gradient(to bottom, #fff 1px, transparent 1px)", 
                     backgroundSize: "25px 25px" 
                   }} 
              />

              {/* Target Marker Pin */}
              <div
                className="absolute w-5 h-5 rounded-full border-2 border-white bg-blue-600 shadow-lg flex items-center justify-center -mt-2.5 -ml-2.5 pointer-events-none"
                style={{
                  left: `${(pixelX / 800) * 100}%`,
                  top: `${(pixelY / 600) * 100}%`,
                }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-white animate-ping" />
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
