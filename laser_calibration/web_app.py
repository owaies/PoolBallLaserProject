"""
Flask Web Application for PoolBot Laser Calibration System.
Provides a modern, responsive browser interface for Chrome.
"""

import io
import os
import sys
import json
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file, Response

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser_calibration.table_geometry import TABLE, get_pocket_locations, is_within_playing_surface
from laser_calibration.calibration_data import CalibrationDataset, CalibrationNode, get_default_nine_samples
from laser_calibration.interpolator import BilinearGridInterpolator, ScatteredInterpolator, GeometricModelInterpolator
from laser_calibration.esp32_client import ESP32Client, ServoSoftLimits
from laser_calibration.camera_homography import CameraHomography, ImageCornerSelection, detect_red_laser_spot
from laser_calibration.spot_measurement import MeasurementSession, SpotMeasurementRecord

app = Flask(__name__)

# State
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "calibration_points.json")
if os.path.exists(json_path):
    dataset = CalibrationDataset.load_from_file(json_path)
else:
    dataset = CalibrationDataset()

interpolators = {
    "Bilinear Grid": BilinearGridInterpolator(dataset.nodes),
    "Scattered IDW": ScatteredInterpolator(dataset.nodes),
    "Geometric Kinematic": GeometricModelInterpolator(),
}
current_model_name = "Bilinear Grid"
active_interpolator = interpolators[current_model_name]

soft_limits = ServoSoftLimits(pan_min=15, pan_max=175, tilt_min=25, tilt_max=155)
esp32 = ESP32Client(base_url="http://192.168.31.2", simulation_mode=True, soft_limits=soft_limits)

camera_homography = CameraHomography()
session_measurements = MeasurementSession()
uploaded_image_cv = None
uploaded_image_b64 = None

# Embed the complete HTML/JS/CSS frontend
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PoolBot Laser Calibration System</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg-dark: #090d16;
  --bg-card: #131d2e;
  --bg-card-light: #1e293b;
  --border-color: #334155;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --accent-blue: #0284c7;
  --accent-blue-hover: #0369a1;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --accent-amber: #f59e0b;
  --felt-green: #0a4d2e;
  --felt-border: #06331e;
  --rail-wood: #543118;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg-dark);
  color: var(--text-main);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
header {
  background: var(--bg-card);
  border-bottom: 1px solid var(--border-color);
  padding: 12px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 100;
}
.brand h1 { font-size: 1.15rem; font-weight: 700; color: #fff; }
.brand p { font-size: 0.8rem; color: var(--text-muted); }
.header-actions { display: flex; align-items: center; gap: 12px; }
.badge {
  font-size: 0.75rem;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 9999px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.badge-sim { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
.badge-live { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
.badge-stop { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }

button {
  font-family: inherit;
  font-weight: 600;
  border: none;
  border-radius: 6px;
  padding: 8px 14px;
  cursor: pointer;
  transition: all 0.15s ease;
  font-size: 0.85rem;
}
.btn-blue { background: var(--accent-blue); color: #fff; }
.btn-blue:hover { background: var(--accent-blue-hover); }
.btn-red { background: var(--accent-red); color: #fff; }
.btn-red:hover { background: #dc2626; }
.btn-green { background: var(--accent-green); color: #fff; }
.btn-green:hover { background: #059669; }
.btn-secondary { background: var(--bg-card-light); color: var(--text-main); border: 1px solid var(--border-color); }
.btn-secondary:hover { background: #334155; }

main {
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 16px;
  padding: 16px 24px;
  flex: 1;
}
@media (max-width: 1024px) {
  main { grid-template-columns: 1fr; }
}

.card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 16px;
  display: flex;
  flex-direction: column;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.card-title { font-size: 0.95rem; font-weight: 600; color: #fff; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* Table Canvas */
.canvas-wrapper {
  position: relative;
  width: 100%;
  aspect-ratio: 2.004 / 1;
  background: #050b14;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}
canvas#poolTableCanvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: crosshair;
}
.table-readouts {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 12px;
}
.stat-box {
  background: var(--bg-card-light);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 10px 14px;
}
.stat-label { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: #fff; }

/* Tabs */
.tabs-nav {
  display: flex;
  border-bottom: 1px solid var(--border-color);
  gap: 4px;
  margin-bottom: 14px;
}
.tab-btn {
  background: transparent;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  border-radius: 0;
  padding: 8px 14px;
}
.tab-btn.active {
  color: var(--accent-blue);
  border-bottom-color: var(--accent-blue);
  background: rgba(2, 132, 199, 0.08);
}
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Form Controls */
.form-group { margin-bottom: 12px; }
.form-label { display: block; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; }
.input-row { display: flex; gap: 8px; align-items: center; }
input[type="number"], select, input[type="text"] {
  background: var(--bg-card-light);
  border: 1px solid var(--border-color);
  color: #fff;
  border-radius: 6px;
  padding: 8px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.9rem;
  width: 100%;
}
.jog-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
  margin-top: 8px;
}
th {
  text-align: left;
  padding: 8px;
  background: var(--bg-card-light);
  color: var(--text-muted);
  font-weight: 600;
  border-bottom: 1px solid var(--border-color);
}
td {
  padding: 8px;
  border-bottom: 1px solid rgba(51, 65, 85, 0.5);
  font-family: 'JetBrains Mono', monospace;
}
tr:hover td { background: rgba(255, 255, 255, 0.02); }

/* Camera Canvas */
.cam-wrapper {
  position: relative;
  width: 100%;
  height: 280px;
  background: #000;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
}
canvas#camCanvas {
  max-width: 100%;
  max-height: 100%;
  display: block;
}

footer {
  background: #060911;
  padding: 6px 24px;
  font-size: 0.75rem;
  color: var(--text-muted);
  display: flex;
  justify-content: space-between;
  border-top: 1px solid rgba(51, 65, 85, 0.5);
}
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>PoolBot Laser Calibration & Verification</h1>
    <p>AI-Based Pool Ball Identification & Laser Pan-Tilt System</p>
  </div>
  <div class="header-actions">
    <label style="display:flex; align-items:center; gap:6px; cursor:pointer; font-size:0.85rem; font-weight:600; color:var(--accent-amber);">
      <input type="checkbox" id="simModeCheckbox" checked onchange="toggleSimulationMode(this.checked)">
      SIMULATION MODE
    </label>
    <span id="espStatusBadge" class="badge badge-sim">ESP32: SIMULATED</span>
    <button class="btn-blue" onclick="centerServos()">CENTER (90°)</button>
    <button class="btn-red" onclick="emergencyStop()">EMERGENCY STOP</button>
  </div>
</header>

<main>
  <!-- Left Side: Scaled Virtual Pool Table -->
  <section class="card">
    <div class="card-header">
      <span class="card-title">Virtual Playing Surface (Exact 2341.9 × 1168.4 mm Scale)</span>
      <span class="mono" id="mouseCoords" style="font-size:0.8rem; color:var(--text-muted);">Cursor: X: --- mm | Y: --- mm</span>
    </div>

    <div class="canvas-wrapper">
      <canvas id="poolTableCanvas" width="900" height="449"></canvas>
    </div>

    <div class="table-readouts">
      <div class="stat-box">
        <div class="stat-label">Selected Target (xr, yr)</div>
        <div class="stat-value mono" id="dispTargetCoords" style="color:var(--accent-green);">1170.9, 584.2 mm</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Predicted Servo Angles</div>
        <div class="stat-value mono" id="dispPredictedAngles" style="color:var(--accent-blue);">P: 125.0° | T: 84.0°</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Active Model & Safety</div>
        <div class="stat-value" id="dispModelStatus" style="font-size:0.95rem; color:#fff;">Bilinear Grid (Safe)</div>
      </div>
    </div>
  </section>

  <!-- Right Side: Control Panels & Tabs -->
  <section class="card">
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('tab-servo', this)">Servo Control</button>
      <button class="tab-btn" onclick="switchTab('tab-nodes', this)">Calibration Nodes</button>
      <button class="tab-btn" onclick="switchTab('tab-camera', this)">Camera Verification</button>
      <button class="tab-btn" onclick="switchTab('tab-results', this)">Results & CSV</button>
    </div>

    <!-- TAB 1: SERVO CONTROL -->
    <div id="tab-servo" class="tab-content active">
      <div class="form-group">
        <label class="form-label">Manual Target Coordinates (mm)</label>
        <div class="input-row">
          <input type="number" id="inpX" value="1170.9" step="1" min="0" max="2341.88" placeholder="X">
          <input type="number" id="inpY" value="584.2" step="1" min="0" max="1168.40" placeholder="Y">
          <button class="btn-secondary" onclick="applyManualCoords()">Apply</button>
        </div>
      </div>

      <div style="background:var(--bg-card-light); border:1px solid var(--border-color); border-radius:8px; padding:12px; margin-bottom:12px;">
        <label style="display:flex; align-items:center; gap:8px; font-size:0.85rem; font-weight:600; cursor:pointer; margin-bottom:10px;">
          <input type="checkbox" id="chkRequireConfirm" checked>
          Require explicit confirmation before moving servos
        </label>
        <button class="btn-blue" style="width:100%; padding:10px; font-size:0.95rem;" onclick="sendPredictedToServos()">
          SEND PREDICTED ANGLES TO SERVOS
        </button>
        <div id="cmdFeedback" class="mono" style="font-size:0.8rem; color:var(--text-muted); margin-top:6px;">Servos ready.</div>
      </div>

      <!-- Live ESP32 Telemetry -->
      <div class="stat-box" style="margin-bottom:12px;">
        <div class="stat-label">Live ESP32 Telemetry</div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; margin-top:4px;" class="mono">
          <div><small style="color:var(--text-muted)">Pan:</small> <span id="telemPan" style="font-weight:700; color:var(--accent-blue)">90°</span></div>
          <div><small style="color:var(--text-muted)">Tilt:</small> <span id="telemTilt" style="font-weight:700; color:var(--accent-blue)">90°</span></div>
          <div><small style="color:var(--text-muted)">Laser:</small> <span id="telemLaser" style="font-weight:700; color:var(--accent-amber)">OFF</span></div>
          <div><small style="color:var(--text-muted)">State:</small> <span id="telemState" style="font-weight:700; color:var(--accent-green)">READY</span></div>
        </div>
      </div>

      <!-- Manual Jog -->
      <div class="form-group">
        <label class="form-label">Manual Pan Jog (±1°, ±5°)</label>
        <div class="jog-grid">
          <button class="btn-secondary" onclick="jogServo('pan', -5)">-5°</button>
          <button class="btn-secondary" onclick="jogServo('pan', -1)">-1°</button>
          <button class="btn-secondary" onclick="jogServo('pan', +1)">+1°</button>
          <button class="btn-secondary" onclick="jogServo('pan', +5)">+5°</button>
        </div>
      </div>
      <div class="form-group">
        <label class="form-label">Manual Tilt Jog (±1°, ±5°)</label>
        <div class="jog-grid">
          <button class="btn-secondary" onclick="jogServo('tilt', -5)">-5°</button>
          <button class="btn-secondary" onclick="jogServo('tilt', -1)">-1°</button>
          <button class="btn-secondary" onclick="jogServo('tilt', +1)">+1°</button>
          <button class="btn-secondary" onclick="jogServo('tilt', +5)">+5°</button>
        </div>
      </div>

      <!-- Laser Safety Controls -->
      <div style="background:rgba(245, 158, 11, 0.08); border:1px solid rgba(245, 158, 11, 0.3); border-radius:8px; padding:10px;">
        <div style="font-size:0.75rem; color:var(--accent-amber); font-weight:600; margin-bottom:6px;">
          SAFETY: Laser is OFF by default. Never point laser at eyes or reflective surfaces.
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn-secondary" style="color:var(--accent-amber);" onclick="toggleLaserExplicit()">MANUAL LASER TOGGLE</button>
          <button class="btn-secondary" onclick="forceLaserOff()">FORCE LASER OFF</button>
        </div>
      </div>
    </div>

    <!-- TAB 2: CALIBRATION NODES -->
    <div id="tab-nodes" class="tab-content">
      <div class="input-row" style="margin-bottom:10px;">
        <select id="cmbModel" onchange="onModelChange(this.value)" style="width:65%;">
          <option value="Bilinear Grid">Bilinear Grid Interpolation</option>
          <option value="Scattered IDW">Scattered 2D IDW</option>
          <option value="Geometric Kinematic">Geometric Kinematic Model</option>
        </select>
        <button class="btn-secondary" onclick="resetDefaultNodes()">Reset Defaults</button>
      </div>

      <div style="max-height:300px; overflow-y:auto;">
        <table>
          <thead>
            <tr>
              <th>Label</th><th>X (mm)</th><th>Y (mm)</th><th>Pan</th><th>Tilt</th>
            </tr>
          </thead>
          <tbody id="nodesTableBody"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: CAMERA VERIFICATION -->
    <div id="tab-camera" class="tab-content">
      <div id="camGuide" style="font-size:0.85rem; font-weight:600; color:var(--accent-blue); margin-bottom:8px;">
        1. Upload a photo of the table. Keep phone camera in a fixed position.
      </div>

      <div class="input-row" style="margin-bottom:10px;">
        <input type="file" id="photoUpload" accept="image/*" onchange="uploadPhoto(this)" style="display:none">
        <button class="btn-blue" onclick="document.getElementById('photoUpload').click()">1. Upload Photo...</button>
        <button class="btn-secondary" onclick="startCornerCalibration()">2. Mark 4 Corners</button>
        <button class="btn-secondary" onclick="startSpotPickMode()">3. Mark Laser Spot</button>
        <button class="btn-secondary" onclick="autoDetectSpot()">Auto Red Spot</button>
      </div>

      <div class="cam-wrapper">
        <canvas id="camCanvas" width="600" height="340"></canvas>
      </div>

      <!-- Measurement readout -->
      <div style="background:var(--bg-card-light); border:1px solid var(--border-color); border-radius:8px; padding:10px; margin-top:10px;">
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:6px;" class="mono">
          <div><small style="color:var(--text-muted)">Measured (xm, ym):</small> <span id="camMeasCoords">--- mm</span></div>
          <div><small style="color:var(--text-muted)">Requested (xr, yr):</small> <span id="camReqCoords">--- mm</span></div>
          <div><small style="color:var(--text-muted)">Offset (ex, ey):</small> <span id="camExEy">--- mm</span></div>
          <div><small style="color:var(--text-muted)">Euclidean Error (e):</small> <span id="camEuclidean" style="font-weight:700; color:var(--accent-amber)">--- mm</span></div>
        </div>
        <button id="btnLogMeasurement" class="btn-green" style="width:100%; margin-top:8px;" onclick="logSpotMeasurement()" disabled>
          Record Measurement to Session Log
        </button>
      </div>
    </div>

    <!-- TAB 4: RESULTS & CSV -->
    <div id="tab-results" class="tab-content">
      <div class="table-readouts" style="margin-top:0; margin-bottom:12px;">
        <div class="stat-box">
          <div class="stat-label">Total Tests</div>
          <div class="stat-value mono" id="statCount">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Mean Error</div>
          <div class="stat-value mono" id="statMean" style="color:var(--accent-blue)">0.0 mm</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">RMS Error</div>
          <div class="stat-value mono" id="statRms" style="color:var(--accent-amber)">0.0 mm</div>
        </div>
      </div>

      <div style="max-height:260px; overflow-y:auto;">
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Target (X, Y)</th><th>Cmd (P, T)</th><th>Meas (X, Y)</th><th>Error (e)</th>
            </tr>
          </thead>
          <tbody id="resultsTableBody"></tbody>
        </table>
      </div>

      <div style="display:flex; justify-content:space-between; margin-top:10px;">
        <button class="btn-green" onclick="exportCsv()">Export Results to CSV</button>
        <button class="btn-secondary" style="color:var(--accent-red)" onclick="clearResults()">Clear Logs</button>
      </div>
    </div>
  </section>
</main>

<footer>
  <span>Pool Table: 2341.88 × 1168.40 mm (Inner Felt) | Soft Limits: Pan [15°-175°], Tilt [25°-155°]</span>
  <span id="footerStatus">Ready. Simulation Mode Enabled.</span>
</footer>

<script>
// Global State
const TABLE_W_MM = 2341.88;
const TABLE_H_MM = 1168.40;
const CUSHION_X_MM = 40.64;
const CUSHION_Y_MM = 21.59;
const OUTER_W_MM = 2423.16;
const OUTER_H_MM = 1211.58;

let currentTargetX = 1170.9;
let currentTargetY = 584.2;
let predictedPan = 125.0;
let predictedTilt = 84.0;
let isSimulationMode = true;
let calibrationNodes = [];

// Camera verification state
let camImg = null;
let camCorners = [];
let camCornerStep = -1;
let camSpotMode = false;
let lastMeasurement = null;

// Canvas setup
const tableCanvas = document.getElementById('poolTableCanvas');
const tableCtx = tableCanvas.getContext('2d');
const camCanvas = document.getElementById('camCanvas');
const camCtx = camCanvas.getContext('2d');

function init() {
  fetchNodes();
  updatePrediction();
  setInterval(pollESPStatus, 1500);

  tableCanvas.addEventListener('mousemove', onTableMouseMove);
  tableCanvas.addEventListener('click', onTableClick);
  camCanvas.addEventListener('click', onCamCanvasClick);
  renderTableCanvas();
}

function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  if (tabId === 'tab-camera') renderCamCanvas();
}

// -------------------------------------------------------------
// VIRTUAL POOL TABLE RENDERING
// -------------------------------------------------------------
function renderTableCanvas() {
  const cw = tableCanvas.width;
  const ch = tableCanvas.height;
  const ctx = tableCtx;
  ctx.clearRect(0, 0, cw, ch);

  const pad = 24;
  const scale = Math.min((cw - 2 * pad) / OUTER_W_MM, (ch - 2 * pad) / OUTER_H_MM);
  const outerW = OUTER_W_MM * scale;
  const outerH = OUTER_H_MM * scale;
  const ox = (cw - outerW) / 2;
  const oy = (ch - outerH) / 2;

  const ix = ox + CUSHION_X_MM * scale;
  const iy = oy + CUSHION_Y_MM * scale;
  const iw = TABLE_W_MM * scale;
  const ih = TABLE_H_MM * scale;

  window.tableTransform = { ox, oy, ix, iy, scale, iw, ih };

  // Outer wood frame
  ctx.fillStyle = '#45220c';
  ctx.fillRect(ox, oy, outerW, outerH);
  ctx.strokeStyle = '#281305';
  ctx.lineWidth = 3;
  ctx.strokeRect(ox, oy, outerW, outerH);

  // Cushion border
  ctx.fillStyle = '#062919';
  ctx.fillRect(ix - 5, iy - 5, iw + 10, ih + 10);

  // Inner felt
  ctx.fillStyle = '#0c5c37';
  ctx.fillRect(ix, iy, iw, ih);

  // Subtle grid
  ctx.strokeStyle = '#09472b';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(ix + iw / 2, iy); ctx.lineTo(ix + iw / 2, iy + ih);
  ctx.moveTo(ix, iy + ih / 2); ctx.lineTo(ix + iw, iy + ih / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Pockets
  const pocketR = 12;
  ctx.fillStyle = '#090d16';
  const pockets = [
    [ix, iy], [ix + iw / 2, iy], [ix + iw, iy],
    [ix, iy + ih], [ix + iw / 2, iy + ih], [ix + iw, iy + ih]
  ];
  pockets.forEach(([px, py]) => {
    ctx.beginPath();
    ctx.arc(px, py, pocketR, 0, Math.PI * 2);
    ctx.fill();
  });

  // Calibration Nodes
  calibrationNodes.forEach(node => {
    const nx = ix + node.x_mm * scale;
    const ny = iy + node.y_mm * scale;

    ctx.beginPath();
    ctx.arc(nx, ny, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.fillText(`${node.label} (${node.pan_deg.toFixed(0)}°, ${node.tilt_deg.toFixed(0)}°)`, nx + 8, ny - 6);
  });

  // Target Crosshair
  const tx = ix + currentTargetX * scale;
  const ty = iy + currentTargetY * scale;

  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(tx, ty, 9, 0, Math.PI * 2);
  ctx.moveTo(tx - 14, ty); ctx.lineTo(tx + 14, ty);
  ctx.moveTo(tx, ty - 14); ctx.lineTo(tx, ty + 14);
  ctx.stroke();

  ctx.fillStyle = '#ef4444';
  ctx.beginPath();
  ctx.arc(tx, ty, 3, 0, Math.PI * 2);
  ctx.fill();
}

function onTableMouseMove(e) {
  if (!window.tableTransform) return;
  const rect = tableCanvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (tableCanvas.width / rect.width);
  const py = (e.clientY - rect.top) * (tableCanvas.height / rect.height);
  const { ix, iy, scale } = window.tableTransform;

  const x_mm = (px - ix) / scale;
  const y_mm = (py - iy) / scale;

  if (x_mm >= 0 && x_mm <= TABLE_W_MM && y_mm >= 0 && y_mm <= TABLE_H_MM) {
    document.getElementById('mouseCoords').textContent = `Cursor: X: ${x_mm.toFixed(1)} mm | Y: ${y_mm.toFixed(1)} mm`;
  } else {
    document.getElementById('mouseCoords').textContent = `Cursor: [Outside Felt Surface]`;
  }
}

function onTableClick(e) {
  if (!window.tableTransform) return;
  const rect = tableCanvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (tableCanvas.width / rect.width);
  const py = (e.clientY - rect.top) * (tableCanvas.height / rect.height);
  const { ix, iy, scale } = window.tableTransform;

  let x_mm = (px - ix) / scale;
  let y_mm = (py - iy) / scale;

  x_mm = Math.max(0, Math.min(TABLE_W_MM, x_mm));
  y_mm = Math.max(0, Math.min(TABLE_H_MM, y_mm));

  currentTargetX = Math.round(x_mm * 10) / 10;
  currentTargetY = Math.round(y_mm * 10) / 10;

  document.getElementById('inpX').value = currentTargetX;
  document.getElementById('inpY').value = currentTargetY;

  updatePrediction();
  renderTableCanvas();
}

function applyManualCoords() {
  const x = parseFloat(document.getElementById('inpX').value);
  const y = parseFloat(document.getElementById('inpY').value);
  if (isNaN(x) || isNaN(y) || x < 0 || x > TABLE_W_MM || y < 0 || y > TABLE_H_MM) {
    alert(`Please enter valid coordinates within:\nX: [0, ${TABLE_W_MM}]\nY: [0, ${TABLE_H_MM}]`);
    return;
  }
  currentTargetX = x;
  currentTargetY = y;
  updatePrediction();
  renderTableCanvas();
}

// -------------------------------------------------------------
// PREDICTION & API CALLS
// -------------------------------------------------------------
async function updatePrediction() {
  document.getElementById('dispTargetCoords').textContent = `${currentTargetX.toFixed(1)}, ${currentTargetY.toFixed(1)} mm`;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x_mm: currentTargetX, y_mm: currentTargetY })
    });
    const data = await res.json();
    predictedPan = data.pan;
    predictedTilt = data.tilt;

    document.getElementById('dispPredictedAngles').textContent = `P: ${predictedPan.toFixed(1)}° | T: ${predictedTilt.toFixed(1)}°`;
    document.getElementById('dispModelStatus').textContent = `${data.model} (${data.safe ? 'Safe' : 'Clamped'})`;
  } catch (err) {
    console.error("Predict error:", err);
  }
}

async function fetchNodes() {
  try {
    const res = await fetch('/api/nodes');
    const data = await res.json();
    calibrationNodes = data.nodes || [];
    renderTableCanvas();
    renderNodesTable();
  } catch (err) {
    console.error("Fetch nodes error:", err);
  }
}

function renderNodesTable() {
  const tbody = document.getElementById('nodesTableBody');
  tbody.innerHTML = '';
  calibrationNodes.forEach(n => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${n.label}</strong></td>
      <td>${n.x_mm.toFixed(1)}</td>
      <td>${n.y_mm.toFixed(1)}</td>
      <td>${n.pan_deg.toFixed(1)}°</td>
      <td>${n.tilt_deg.toFixed(1)}°</td>
    `;
    tbody.appendChild(tr);
  });
}

async function onModelChange(val) {
  await fetch('/api/model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: val })
  });
  updatePrediction();
}

async function resetDefaultNodes() {
  if (!confirm("Reset calibration nodes to the 9 default empirical samples?")) return;
  await fetch('/api/nodes/reset', { method: 'POST' });
  fetchNodes();
  updatePrediction();
}

// -------------------------------------------------------------
// ESP32 COMMANDS & TELEMETRY
// -------------------------------------------------------------
async function toggleSimulationMode(enabled) {
  isSimulationMode = enabled;
  await fetch('/api/simulation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: isSimulationMode })
  });
  pollESPStatus();
}

async function sendPredictedToServos() {
  const reqConfirm = document.getElementById('chkRequireConfirm').checked;
  if (reqConfirm && !isSimulationMode) {
    const ok = confirm(`CONFIRM SERVO MOVEMENT:\n\nTarget: (${currentTargetX}, ${currentTargetY}) mm\nPan: ${predictedPan.toFixed(1)}°\nTilt: ${predictedTilt.toFixed(1)}°\n\nEnsure laser path is safe from eyes and reflective obstacles.`);
    if (!ok) return;
  }

  const res = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pan: predictedPan, tilt: predictedTilt, confirmed: true })
  });
  const data = await res.json();
  const feedback = document.getElementById('cmdFeedback');
  feedback.textContent = data.message;
  feedback.style.color = data.success ? 'var(--accent-green)' : 'var(--accent-red)';
}

async function jogServo(axis, delta) {
  const res = await fetch('/api/jog', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ axis, delta })
  });
  const data = await res.json();
  document.getElementById('cmdFeedback').textContent = data.message;
}

async function centerServos() {
  const res = await fetch('/api/center', { method: 'POST' });
  const data = await res.json();
  alert(data.message);
  pollESPStatus();
}

async function emergencyStop() {
  const res = await fetch('/api/stop', { method: 'POST' });
  const data = await res.json();
  alert(`EMERGENCY STOP TRIGGERED:\n${data.message}\nLaser turned OFF immediately. Use CENTER to resume.`);
  pollESPStatus();
}

async function toggleLaserExplicit() {
  const currentLaser = document.getElementById('telemLaser').textContent === 'ON';
  if (currentLaser) {
    await fetch('/api/laser', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: 0 })
    });
  } else {
    const ok = confirm("CRITICAL LASER SAFETY WARNING:\n\n1. Laser is OFF by default.\n2. Eye protection required.\n3. Safe non-reflective target surface required.\n\nTurn laser ON now?");
    if (ok) {
      await fetch('/api/laser', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: 1, acknowledged: true })
      });
    }
  }
  pollESPStatus();
}

async function forceLaserOff() {
  await fetch('/api/laser', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: 0 })
  });
  pollESPStatus();
}

async function pollESPStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const badge = document.getElementById('espStatusBadge');
    if (data.is_simulation) {
      badge.className = 'badge badge-sim';
      badge.textContent = 'ESP32: SIMULATED';
    } else if (data.connected) {
      badge.className = 'badge badge-live';
      badge.textContent = `ESP32: CONNECTED (${data.latency_ms}ms)`;
    } else {
      badge.className = 'badge badge-stop';
      badge.textContent = 'ESP32: DISCONNECTED';
    }

    document.getElementById('telemPan').textContent = `${data.pan}°`;
    document.getElementById('telemTilt').textContent = `${data.tilt}°`;
    document.getElementById('telemLaser').textContent = data.laser ? 'ON' : 'OFF';
    document.getElementById('telemLaser').style.color = data.laser ? 'var(--accent-red)' : 'var(--accent-amber)';
    document.getElementById('telemState').textContent = data.stopped ? 'STOPPED' : 'READY';
    document.getElementById('telemState').style.color = data.stopped ? 'var(--accent-red)' : 'var(--accent-green)';
  } catch (err) {
    // server down
  }
}

// -------------------------------------------------------------
// CAMERA VERIFICATION & HOMOGRAPHY
// -------------------------------------------------------------
function uploadPhoto(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('image', file);

  fetch('/api/image/upload', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        camImg = new Image();
        camImg.onload = () => {
          renderCamCanvas();
          document.getElementById('camGuide').textContent = "Photo loaded. Click '2. Mark 4 Corners' to calibrate table boundaries.";
        };
        camImg.src = data.image_data;
      }
    });
}

function startCornerCalibration() {
  if (!camImg) { alert("Please upload a photo of the table first."); return; }
  camCorners = [];
  camCornerStep = 0;
  camSpotMode = false;
  updateCornerGuide();
  renderCamCanvas();
}

function updateCornerGuide() {
  const steps = [
    "Click Corner 1: TOP-LEFT inner felt corner",
    "Click Corner 2: TOP-RIGHT inner felt corner",
    "Click Corner 3: BOTTOM-RIGHT inner felt corner",
    "Click Corner 4: BOTTOM-LEFT inner felt corner"
  ];
  if (camCornerStep < 4) {
    document.getElementById('camGuide').textContent = `CALIBRATION: ${steps[camCornerStep]}`;
  } else {
    document.getElementById('camGuide').textContent = "Homography computed! Now click '3. Mark Laser Spot'.";
  }
}

function startSpotPickMode() {
  camSpotMode = true;
  document.getElementById('camGuide').textContent = "SPOT MODE: Click the physical red laser spot on the photo.";
}

function onCamCanvasClick(e) {
  if (!camImg || !window.camTransform) return;
  const rect = camCanvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (camCanvas.width / rect.width);
  const cy = (e.clientY - rect.top) * (camCanvas.height / rect.height);
  const { ox, oy, scale } = window.camTransform;

  const u = (cx - ox) / scale;
  const v = (cy - oy) / scale;

  if (camCornerStep >= 0 && camCornerStep < 4) {
    camCorners.push([u, v]);
    camCornerStep++;
    if (camCornerStep === 4) {
      // Compute homography
      fetch('/api/homography/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corners: camCorners })
      }).then(r => r.json()).then(data => {
        if (data.success) {
          alert("Perspective Homography computed successfully!");
        }
      });
    }
    updateCornerGuide();
    renderCamCanvas();
  } else if (camSpotMode) {
    evaluateSpot(u, v);
  }
}

async function autoDetectSpot() {
  const res = await fetch('/api/homography/autodetect');
  const data = await res.json();
  if (data.detected) {
    evaluateSpot(data.u, data.v);
  } else {
    alert("Could not automatically locate red laser spot. Please click manually.");
  }
}

async function evaluateSpot(u, v) {
  const res = await fetch('/api/homography/measure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ u, v, target_x: currentTargetX, target_y: currentTargetY, pan: predictedPan, tilt: predictedTilt })
  });
  const data = await res.json();
  if (data.success) {
    lastMeasurement = data.record;
    document.getElementById('camMeasCoords').textContent = `${data.record.measured_x_mm.toFixed(1)}, ${data.record.measured_y_mm.toFixed(1)} mm`;
    document.getElementById('camReqCoords').textContent = `${data.record.target_x_mm.toFixed(1)}, ${data.record.target_y_mm.toFixed(1)} mm`;
    document.getElementById('camExEy').textContent = `ex: ${data.record.error_x_mm > 0 ? '+' : ''}${data.record.error_x_mm.toFixed(1)} | ey: ${data.record.error_y_mm > 0 ? '+' : ''}${data.record.error_y_mm.toFixed(1)} mm`;
    document.getElementById('camEuclidean').textContent = `${data.record.euclidean_error_mm.toFixed(1)} mm`;
    document.getElementById('btnLogMeasurement').disabled = false;
    renderCamCanvas(u, v, data.record.euclidean_error_mm);
  }
}

function logSpotMeasurement() {
  if (!lastMeasurement) return;
  fetch('/api/results/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(lastMeasurement)
  }).then(r => r.json()).then(data => {
    alert(`Logged spot measurement! Error: ${lastMeasurement.euclidean_error_mm.toFixed(1)} mm`);
    document.getElementById('btnLogMeasurement').disabled = true;
    fetchResults();
  });
}

function renderCamCanvas(spotU, spotV, spotErr) {
  const cw = camCanvas.width;
  const ch = camCanvas.height;
  const ctx = camCtx;
  ctx.clearRect(0, 0, cw, ch);

  if (!camImg) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText("Upload table photo to begin camera verification", cw / 2, ch / 2);
    return;
  }

  const scale = Math.min(cw / camImg.width, ch / camImg.height);
  const nw = camImg.width * scale;
  const nh = camImg.height * scale;
  const ox = (cw - nw) / 2;
  const oy = (ch - nh) / 2;
  window.camTransform = { ox, oy, scale };

  ctx.drawImage(camImg, ox, oy, nw, nh);

  // Draw corners
  if (camCorners.length > 0) {
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    camCorners.forEach(([u, v], i) => {
      const su = ox + u * scale;
      const sv = oy + v * scale;
      if (i === 0) ctx.moveTo(su, sv); else ctx.lineTo(su, sv);
    });
    if (camCorners.length === 4) ctx.closePath();
    ctx.stroke();

    camCorners.forEach(([u, v], i) => {
      const su = ox + u * scale;
      const sv = oy + v * scale;
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(su, sv, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px Inter';
      ctx.fillText(`C${i+1}`, su + 6, sv - 6);
    });
  }

  // Draw spot
  if (spotU !== undefined && spotV !== undefined) {
    const su = ox + spotU * scale;
    const sv = oy + spotV * scale;
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(su, sv, 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 11px JetBrains Mono';
    ctx.fillText(`e=${spotErr.toFixed(1)}mm`, su + 12, sv + 4);
  }
}

// -------------------------------------------------------------
// RESULTS TAB
// -------------------------------------------------------------
async function fetchResults() {
  const res = await fetch('/api/results');
  const data = await res.json();
  const tbody = document.getElementById('resultsTableBody');
  tbody.innerHTML = '';

  (data.records || []).forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.timestamp.split(' ')[1] || r.timestamp}</td>
      <td>(${r.target_x_mm.toFixed(1)}, ${r.target_y_mm.toFixed(1)})</td>
      <td>(${r.commanded_pan_deg.toFixed(1)}°, ${r.commanded_tilt_deg.toFixed(1)}°)</td>
      <td>(${r.measured_x_mm.toFixed(1)}, ${r.measured_y_mm.toFixed(1)})</td>
      <td style="color:var(--accent-amber); font-weight:700;">${r.euclidean_error_mm.toFixed(1)} mm</td>
    `;
    tbody.appendChild(tr);
  });

  const s = data.stats || {};
  document.getElementById('statCount').textContent = s.count || 0;
  document.getElementById('statMean').textContent = `${(s.mean_error_mm || 0).toFixed(1)} mm`;
  document.getElementById('statRms').textContent = `${(s.rms_error_mm || 0).toFixed(1)} mm`;
}

function exportCsv() {
  window.location.href = '/api/results/export';
}

async function clearResults() {
  if (!confirm("Clear all recorded session results?")) return;
  await fetch('/api/results/clear', { method: 'POST' });
  fetchResults();
}

window.onload = init;
</script>
</body>
</html>
"""

# =============================================================================
# API ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    x = float(data.get("x_mm", TABLE.INNER_LENGTH_MM / 2.0))
    y = float(data.get("y_mm", TABLE.INNER_BREADTH_MM / 2.0))
    pan, tilt, info = active_interpolator.predict(x, y)
    safe = soft_limits.is_pan_safe(pan) and soft_limits.is_tilt_safe(tilt)
    return jsonify({
        "pan": round(pan, 1),
        "tilt": round(tilt, 1),
        "model": current_model_name,
        "safe": safe,
        "info": info
    })

@app.route("/api/nodes", methods=["GET"])
def api_get_nodes():
    return jsonify({
        "nodes": [n.to_dict() for n in dataset.nodes]
    })

@app.route("/api/nodes/reset", methods=["POST"])
def api_reset_nodes():
    dataset.reset_to_defaults()
    for interp in interpolators.values():
        interp.fit(dataset.nodes)
    return jsonify({"success": True})

@app.route("/api/model", methods=["POST"])
def api_set_model():
    global current_model_name, active_interpolator
    data = request.json or {}
    model_name = data.get("model", "Bilinear Grid")
    if model_name in interpolators:
        current_model_name = model_name
        active_interpolator = interpolators[model_name]
        active_interpolator.fit(dataset.nodes)
    return jsonify({"success": True, "model": current_model_name})

@app.route("/api/simulation", methods=["POST"])
def api_simulation():
    data = request.json or {}
    enabled = bool(data.get("enabled", True))
    esp32.set_simulation_mode(enabled)
    return jsonify({"success": True, "simulation": enabled})

@app.route("/api/status", methods=["GET"])
def api_status():
    status = esp32.get_status()
    return jsonify({
        "connected": status.connected,
        "pan": status.pan,
        "tilt": status.tilt,
        "laser": status.laser,
        "stopped": status.stopped,
        "latency_ms": status.latency_ms,
        "is_simulation": status.is_simulation
    })

@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.json or {}
    pan = float(data.get("pan", 90))
    tilt = float(data.get("tilt", 90))
    confirmed = bool(data.get("confirmed", False))
    success, msg = esp32.move_servos(pan, tilt, confirmed=confirmed)
    return jsonify({"success": success, "message": msg})

@app.route("/api/jog", methods=["POST"])
def api_jog():
    data = request.json or {}
    axis = str(data.get("axis", "pan"))
    delta = int(data.get("delta", 0))
    status = esp32.get_status()
    if axis == "pan":
        success, msg = esp32.move_servos(status.pan + delta, status.tilt, confirmed=True)
    else:
        success, msg = esp32.move_servos(status.pan, status.tilt + delta, confirmed=True)
    return jsonify({"success": success, "message": msg})

@app.route("/api/center", methods=["POST"])
def api_center():
    success, msg = esp32.center_servos()
    return jsonify({"success": success, "message": msg})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    success, msg = esp32.stop_servos()
    return jsonify({"success": success, "message": msg})

@app.route("/api/laser", methods=["POST"])
def api_laser():
    data = request.json or {}
    state = int(data.get("state", 0))
    ack = bool(data.get("acknowledged", False))
    success, msg = esp32.set_laser(enable=(state == 1), safety_acknowledgment=ack)
    return jsonify({"success": success, "message": msg})

@app.route("/api/image/upload", methods=["POST"])
def api_upload_image():
    global uploaded_image_cv, uploaded_image_b64
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    uploaded_image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Base64 data url for browser display
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else "jpeg"
    uploaded_image_b64 = f"data:image/{ext};base64,{b64}"

    return jsonify({"success": True, "image_data": uploaded_image_b64})

@app.route("/api/homography/calibrate", methods=["POST"])
def api_calibrate_homography():
    data = request.json or {}
    corners = data.get("corners", [])
    if len(corners) != 4:
        return jsonify({"success": False, "error": "Exactly 4 corners required"}), 400

    corner_sel = ImageCornerSelection(
        top_left=tuple(corners[0]),
        top_right=tuple(corners[1]),
        bottom_right=tuple(corners[2]),
        bottom_left=tuple(corners[3]),
    )
    success = camera_homography.compute_homography(corner_sel)
    return jsonify({"success": success})

@app.route("/api/homography/measure", methods=["POST"])
def api_measure_spot():
    data = request.json or {}
    u = float(data.get("u", 0))
    v = float(data.get("v", 0))
    tx = float(data.get("target_x", 0))
    ty = float(data.get("target_y", 0))
    pan = float(data.get("pan", 90))
    tilt = float(data.get("tilt", 90))

    mapped = camera_homography.image_to_table(u, v)
    if not mapped:
        return jsonify({"success": False, "error": "Homography not calibrated or degenerate"}), 400

    xm, ym = mapped
    rec = SpotMeasurementRecord.create(
        target_x_mm=tx,
        target_y_mm=ty,
        commanded_pan_deg=pan,
        commanded_tilt_deg=tilt,
        measured_x_mm=xm,
        measured_y_mm=ym,
    )
    return jsonify({"success": True, "record": rec.to_dict()})

@app.route("/api/homography/autodetect", methods=["GET"])
def api_autodetect_spot():
    if uploaded_image_cv is None:
        return jsonify({"detected": False, "error": "No image loaded"})
    spot = detect_red_laser_spot(uploaded_image_cv)
    if spot:
        return jsonify({"detected": True, "u": spot[0], "v": spot[1], "confidence": spot[2]})
    return jsonify({"detected": False})

@app.route("/api/results", methods=["GET"])
def api_get_results():
    stats = session_measurements.calculate_statistics()
    return jsonify({
        "records": [r.to_dict() for r in session_measurements.records],
        "stats": {
            "count": stats.count,
            "mean_error_mm": stats.mean_error_mm,
            "median_error_mm": stats.median_error_mm,
            "max_error_mm": stats.max_error_mm,
            "rms_error_mm": stats.rms_error_mm,
            "mean_ex_mm": stats.mean_ex_mm,
            "mean_ey_mm": stats.mean_ey_mm,
        }
    })

@app.route("/api/results/add", methods=["POST"])
def api_add_result():
    data = request.json or {}
    rec = SpotMeasurementRecord(
        id=data.get("id", ""),
        timestamp=data.get("timestamp", ""),
        target_x_mm=float(data.get("target_x_mm", 0)),
        target_y_mm=float(data.get("target_y_mm", 0)),
        commanded_pan_deg=float(data.get("commanded_pan_deg", 90)),
        commanded_tilt_deg=float(data.get("commanded_tilt_deg", 90)),
        measured_x_mm=float(data.get("measured_x_mm", 0)),
        measured_y_mm=float(data.get("measured_y_mm", 0)),
        error_x_mm=float(data.get("error_x_mm", 0)),
        error_y_mm=float(data.get("error_y_mm", 0)),
        euclidean_error_mm=float(data.get("euclidean_error_mm", 0)),
        notes=data.get("notes", ""),
        image_file=data.get("image_file", ""),
    )
    session_measurements.add_record(rec)
    return jsonify({"success": True})

@app.route("/api/results/clear", methods=["POST"])
def api_clear_results():
    session_measurements.clear()
    return jsonify({"success": True})

@app.route("/api/results/export", methods=["GET"])
def api_export_csv():
    output_path = os.path.join(base_dir, "browser_calibration_results.csv")
    session_measurements.export_to_csv(output_path)
    return send_file(output_path, as_attachment=True, download_name="laser_calibration_results.csv")

def start_server(port=5000):
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    start_server()
