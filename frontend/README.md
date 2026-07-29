# Pool Ball Identification & Laser Positioning Frontend

This React web application serves as the control dashboard interface for the AI-Based Pool Ball Identification and Laser Positioning System. It interacts with the FastAPI backend REST API.

## Features

- **Apple Scroll Animation Hero:** Dynamic scrub frame canvas showing a 3D pool table scanner optical sequence on scrolling.
- **System Dashboard:** Real-time model checkpoints, GPU utilization, active device logs, and Chart.js analytics.
- **Single Image YOLOv8 Detection:** Interactive bounding-box viewer with exports for CSV, JSON coordinates, and annotated images.
- **Batch Processing:** Run folder-level YOLOv8 inferences asynchronously.
- **Millimeter Coordinate Projection:** Grid-mapping simulation tool translating pixel values into physical pool table boundaries using homography matrices.
- **Intrinsic Calibration Matrix Visualizer:** Renders loaded camera intrinsic coefficients.

---

## Folder Structure

```
frontend/
├── public/
│   └── videoframes/     # Preloaded optical sequence JPG frames
├── src/
│   ├── components/      # AppleScrollCanvas.tsx
│   ├── layouts/         # DashboardLayout.tsx
│   ├── pages/           # Home, Dashboard, Detection, Calibration, etc.
│   ├── services/        # api.ts (Axios connection)
│   ├── types/           # index.ts (TS interfaces)
│   ├── App.tsx          # Router definitions
│   └── index.css        # Tailwind v4 globals & custom glassmorphism
```

---

## Setup & Running Instructions

### Prerequisites
- Node.js (V20+ recommended)
- Running instances of the FastAPI Backend on `http://127.0.0.1:8000` (or configured host)

### Installation
1. Navigate into the frontend folder:
   ```bash
   cd frontend
   ```
2. Install packages:
   ```bash
   npm install
   ```

### Run Dev Server
Launch Vite development server:
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build Production Bundle
To compile optimized static bundles:
```bash
npm run build
```
The compiled output is saved in `frontend/dist/`.
