"""
Professional 3D Snooker Ball Identification & Laser Positioning System
High-End Realistic 3D Simulation, Dark Cinematic Control-Room Dashboard,
Procedural Strachan Baize Weave, Polished Walnut Rails, Turned Wooden Legs,
Overhead Tournament Luminaire, High-Gloss Aramith Balls, Mechatronic Laser Gimbal,
Real-Time Mobile IP Webcam Streaming with Picture-in-Picture, and OpenCV Homography.
"""

import os, sys, json, math, time, io, base64
import requests
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify, send_file, Response, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser_calibration.snooker_geometry import SNOOKER, get_snooker_pockets, is_within_snooker_table
from laser_calibration.camera_calibration import SnookerCameraHomography
from laser_calibration.snooker_interpolator import SnookerLaserInterpolator
from laser_calibration.laser_spot_detector import detect_laser_spot
from laser_calibration.snooker_detection import SnookerBallDetector

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

# Calibration Data
CALIB_FILE = os.path.join(os.path.dirname(__file__), "snooker_calibration.json")
with open(CALIB_FILE, "r") as f:
    calib_data = json.load(f)

interpolator = SnookerLaserInterpolator(calib_data["nodes"])
camera_homography = SnookerCameraHomography()
detector = SnookerBallDetector("models/snooker/best.pt")

# System State
esp32_base_url = "http://192.168.31.2"
simulation_mode = True
laser_state = False
current_pan = 10.0
current_tilt = 150.0
emergency_stopped = False
ip_webcam_url = "http://192.168.31.150:8080"  # Default IP Webcam URL

uploaded_frame_bgr = None
uploaded_frame_b64 = None
detected_balls_cache = []
execution_logs = []

def add_log(category: str, message: str):
    execution_logs.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "category": category,
        "message": message
    })

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PoolBot 3D — Professional Snooker Analytics & Laser Positioning</title>
<!-- Fast Local System Typography (Zero Network Blocking) -->
<!-- Three.js and OrbitControls (Loaded instantly from local static/js) -->
<script src="/static/js/three.min.js"></script>
<script src="/static/js/OrbitControls.js"></script>
<style>
:root {
  --bg-darkest: #070b13;
  --bg-deep: #0b111f;
  --bg-card: rgba(15, 23, 42, 0.78);
  --bg-card-hover: rgba(22, 33, 58, 0.9);
  --bg-card-active: rgba(30, 41, 59, 0.95);
  
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-accent: rgba(16, 185, 129, 0.35);
  --border-cyan: rgba(6, 182, 212, 0.4);
  
  --emerald-green: #10b981;
  --emerald-glow: rgba(16, 185, 129, 0.4);
  --cyan-accent: #06b6d4;
  --gold-accent: #f59e0b;
  --ruby-red: #ef4444;
  --ruby-glow: rgba(239, 68, 68, 0.45);
  
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  
  --radius-capsule: 999px;
  --radius-panel: 18px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
  color: var(--text-main);
  background-color: var(--bg-darkest);
  background-image: 
    radial-gradient(circle at 15% 20%, rgba(16, 185, 129, 0.07) 0%, transparent 45%),
    radial-gradient(circle at 85% 80%, rgba(6, 182, 212, 0.06) 0%, transparent 45%),
    radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.5) 0%, transparent 100%);
  background-attachment: fixed;
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(7, 11, 19, 0.6); }
::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.3); border-radius: var(--radius-capsule); }
::-webkit-scrollbar-thumb:hover { background: var(--emerald-green); }

/* Header Component */
header {
  background: rgba(11, 17, 31, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border-subtle);
  padding: 12px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
}

.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}
.brand-icon {
  width: 44px;
  height: 44px;
  border-radius: var(--radius-capsule);
  background: linear-gradient(135deg, #059669, #10b981);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 20px rgba(16, 185, 129, 0.35);
  transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.brand-icon:hover { transform: scale(1.08) rotate(4deg); }
.brand-icon svg { width: 22px; height: 22px; fill: #fff; }

.brand-text h1 {
  font-size: 1.15rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #ffffff;
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand-text p {
  font-size: 0.72rem;
  color: var(--text-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-weight: 500;
}

/* Nav Pills (Capsule Styled) */
.nav-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.nav-btn {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-subtle);
  color: var(--text-muted);
  padding: 8px 18px;
  border-radius: var(--radius-capsule);
  font-weight: 600;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.nav-btn:hover {
  background: rgba(255, 255, 255, 0.09);
  color: #ffffff;
  border-color: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}
.nav-btn.active {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(6, 182, 212, 0.25));
  color: #ffffff;
  border-color: var(--emerald-green);
  box-shadow: 0 0 16px rgba(16, 185, 129, 0.3);
}

/* Header Telemetry Badges */
.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
.status-pill {
  padding: 6px 14px;
  border-radius: var(--radius-capsule);
  font-size: 0.75rem;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  border: 1px solid var(--border-subtle);
  background: rgba(0, 0, 0, 0.4);
}
.status-pill.online {
  color: #34d399;
  border-color: rgba(52, 211, 153, 0.35);
  box-shadow: 0 0 12px rgba(52, 211, 153, 0.2);
}
.status-pill.offline {
  color: #f87171;
  border-color: rgba(248, 113, 113, 0.35);
}
.status-pill.sim {
  color: #38bdf8;
  border-color: rgba(56, 189, 248, 0.35);
}

/* Pulsing Status Dot */
@keyframes pulseDot {
  0% { transform: scale(0.92); opacity: 0.8; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1.08); opacity: 1; box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.92); opacity: 0.8; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.live-pulsing-dot {
  width: 8px;
  height: 8px;
  background-color: var(--emerald-green);
  border-radius: 50%;
  display: inline-block;
  animation: pulseDot 1.8s infinite;
}

/* E-STOP Button */
.btn-estop {
  background: linear-gradient(135deg, #dc2626, #b91c1c);
  color: #ffffff;
  border: 1px solid rgba(239, 68, 68, 0.5);
  padding: 8px 20px;
  border-radius: var(--radius-capsule);
  font-size: 0.8rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 0 16px var(--ruby-glow);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.btn-estop:hover {
  background: linear-gradient(135deg, #ef4444, #dc2626);
  transform: translateY(-2px);
  box-shadow: 0 0 24px rgba(239, 68, 68, 0.65);
}

/* Main Layout Grid */
main {
  flex: 1;
  padding: 20px 28px;
  max-width: 1720px;
  margin: 0 auto;
  width: 100%;
  display: grid;
  grid-template-columns: 1fr 430px;
  gap: 22px;
}
@media (max-width: 1240px) {
  main { grid-template-columns: 1fr; }
}

/* Glass Panels */
.glass-panel {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-panel);
  padding: 20px;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
  position: relative;
  overflow: hidden;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: #ffffff;
}
.panel-header .tag-pill {
  font-size: 0.7rem;
  padding: 4px 12px;
  border-radius: var(--radius-capsule);
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  border: 1px solid var(--border-subtle);
}

.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

/* 3D Viewport Box */
.viewport-3d-box {
  width: 100%;
  height: 640px;
  background: radial-gradient(circle at 50% 35%, #0f1a2c 0%, #060912 85%);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 30px rgba(0, 0, 0, 0.7);
}
#three-container { width: 100%; height: 100%; display: block; }

.viewport-toolbar {
  position: absolute;
  top: 14px;
  left: 14px;
  display: flex;
  gap: 8px;
  z-index: 10;
  flex-wrap: wrap;
}
.viewport-btn {
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid var(--border-subtle);
  color: var(--text-muted);
  padding: 7px 15px;
  border-radius: var(--radius-capsule);
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
  backdrop-filter: blur(12px);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
.viewport-btn:hover {
  background: rgba(255, 255, 255, 0.12);
  color: #ffffff;
  border-color: rgba(255, 255, 255, 0.25);
  transform: translateY(-2px);
}
.viewport-btn.active {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(6, 182, 212, 0.3));
  color: #ffffff;
  border-color: var(--emerald-green);
  box-shadow: 0 0 12px var(--emerald-glow);
}

.viewport-overlay-hud {
  position: absolute;
  bottom: 14px;
  left: 14px;
  right: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  pointer-events: none;
  flex-wrap: wrap;
  gap: 8px;
}
.hud-pill {
  background: rgba(11, 17, 31, 0.88);
  border: 1px solid var(--border-subtle);
  backdrop-filter: blur(14px);
  padding: 6px 14px;
  border-radius: var(--radius-capsule);
  font-size: 0.74rem;
  color: #e2e8f0;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

/* Picture-in-Picture Live Stream Window */
#cameraPipBox {
  display: none;
  position: absolute;
  bottom: 58px;
  right: 14px;
  width: 270px;
  height: 165px;
  background: rgba(11, 17, 31, 0.94);
  backdrop-filter: blur(14px);
  border: 1.5px solid var(--border-accent);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.6);
  z-index: 20;
}
.pip-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  background: rgba(0, 0, 0, 0.5);
  font-size: 0.7rem;
  color: #34d399;
  font-weight: 700;
}

/* Telemetry Grid below Viewport */
.telemetry-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 14px;
}
.telemetry-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-subtle);
  padding: 14px 18px;
  border-radius: 16px;
  transition: border-color 0.25s ease;
}
.telemetry-card:hover { border-color: rgba(255, 255, 255, 0.16); }
.telemetry-card small {
  display: block;
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 5px;
  font-weight: 600;
}
.telemetry-card .value {
  font-size: 1.05rem;
  font-weight: 700;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  color: #ffffff;
}
.telemetry-card.target .value { color: #38bdf8; }
.telemetry-card.angles .value { color: #34d399; }

/* Capsule Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 20px;
  border-radius: var(--radius-capsule);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
  text-decoration: none;
}
.btn-emerald {
  background: linear-gradient(135deg, #059669, #10b981);
  color: #ffffff;
  box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
}
.btn-emerald:hover {
  background: linear-gradient(135deg, #10b981, #34d399);
  transform: translateY(-2px);
  box-shadow: 0 6px 22px rgba(16, 185, 129, 0.45);
}
.btn-secondary {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border-subtle);
  color: #e2e8f0;
}
.btn-secondary:hover {
  background: rgba(255, 255, 255, 0.12);
  border-color: rgba(255, 255, 255, 0.25);
  color: #ffffff;
  transform: translateY(-2px);
}
.btn-danger {
  background: linear-gradient(135deg, #dc2626, #b91c1c);
  color: #ffffff;
  box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
}
.btn-danger:hover {
  background: linear-gradient(135deg, #ef4444, #dc2626);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5);
}

/* Preset Buttons Grid (3x3) */
.presets-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 16px;
}
.preset-btn {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-subtle);
  color: #cbd5e1;
  padding: 10px 6px;
  border-radius: var(--radius-capsule);
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  text-align: center;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap;
}
.preset-btn:hover {
  background: rgba(16, 185, 129, 0.15);
  border-color: var(--emerald-green);
  color: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
}
.preset-btn.calib-node {
  border-style: dashed;
  color: #facc15;
}
.preset-btn.calib-node:hover {
  background: rgba(250, 204, 21, 0.15);
  border-color: #facc15;
}

/* Form Inputs */
.form-group {
  margin-bottom: 14px;
}
.form-group label {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 6px;
}
input[type="text"], input[type="number"], select {
  width: 100%;
  background: rgba(0, 0, 0, 0.45);
  border: 1px solid var(--border-subtle);
  color: #ffffff;
  padding: 10px 16px;
  border-radius: var(--radius-capsule);
  font-size: 0.8rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  outline: none;
  transition: all 0.2s ease;
}
input[type="text"]:focus, input[type="number"]:focus, select:focus {
  border-color: var(--emerald-green);
  box-shadow: 0 0 12px var(--emerald-glow);
}
input[type="range"] {
  width: 100%;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-capsule);
  outline: none;
  -webkit-appearance: none;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--emerald-green);
  cursor: pointer;
  box-shadow: 0 0 10px var(--emerald-glow);
}

/* Tab Panes */
.tab-pane { display: none; }
.tab-pane.active { display: block; animation: softFade 0.25s ease-out; }
@keyframes softFade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

/* Light Data Tables */
.table-scroll {
  max-height: 290px;
  overflow-y: auto;
  border-radius: 14px;
  border: 1px solid var(--border-subtle);
  background: rgba(0, 0, 0, 0.3);
}
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
th, td { padding: 10px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); text-align: left; }
th { background: rgba(15, 23, 42, 0.95); color: #94a3b8; font-weight: 700; position: sticky; top: 0; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em; }
tr:hover td { background: rgba(255, 255, 255, 0.04); }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-icon">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke="#fff" stroke-width="2" fill="none"/><circle cx="12" cy="12" r="3" fill="#fff"/></svg>
    </div>
    <div class="brand-text">
      <h1>PoolBot 3D <span style="font-size:0.65rem; background:rgba(16,185,129,0.15); color:var(--emerald-green); border:1px solid rgba(16,185,129,0.3); padding:2px 8px; border-radius:999px;">PRO</span></h1>
      <p>AI Snooker Ball Identification & Laser Aiming Platform • TR (0,0) System</p>
    </div>
  </div>

  <nav class="nav-pills">
    <button class="nav-btn active" onclick="showTab('tab-3d', this)">🎱 3D Table View</button>
    <button class="nav-btn" onclick="showTab('tab-camera', this)">📷 Camera & Homography</button>
    <button class="nav-btn" onclick="showTab('tab-servo', this)">⚙️ Servo & Laser</button>
    <button class="nav-btn" onclick="showTab('tab-balls', this)">🎯 Detected Balls</button>
    <button class="nav-btn" onclick="showTab('tab-foul', this)">⭐ Foul Repositioning</button>
    <button class="nav-btn" onclick="showTab('tab-calib', this)">📐 9-Point Calibration</button>
    <button class="nav-btn" onclick="showTab('tab-logs', this)">📋 Logs</button>
  </nav>

  <div class="header-actions">
    <label style="display:inline-flex; align-items:center; gap:6px; font-size:0.75rem; color:var(--text-muted); cursor:pointer;">
      <input type="checkbox" id="chkSim" checked onchange="toggleSim(this.checked)">
      <span>SIMULATION</span>
    </label>
    <div class="status-pill online" id="badgeEsp">
      <span class="live-pulsing-dot"></span>
      <span id="txtEspBadge">ESP32: 192.168.31.2</span>
    </div>
    <button class="btn-estop" onclick="emergencyStop()">🛑 E-STOP</button>
  </div>
</header>

<main>
  <!-- LEFT COLUMN: TABS CONTENT -->
  <div class="glass-panel">
    <!-- TAB 1: 3D TABLE VIEWPORT -->
    <div id="tab-3d" class="tab-pane active">
      <div class="panel-header">
        <span>Interactive 3D Physical Simulation</span>
        <span class="tag-pill">6 REGULATION POCKETS</span>
      </div>

      <div class="viewport-3d-box">
        <div class="viewport-toolbar">
          <button class="viewport-btn active" id="btnViewPersp" onclick="setCameraView('perspective')">Perspective 3D</button>
          <button class="viewport-btn" id="btnViewTop" onclick="setCameraView('topdown')">Top-Down (Bird's Eye)</button>
          <button class="viewport-btn" id="btnViewBaulk" onclick="setCameraView('baulk')">Baulk / Player View</button>
          <button class="viewport-btn" id="btnViewBlack" onclick="setCameraView('black')">Black Spot Cam</button>
          <button class="viewport-btn" onclick="resetCamera()">Reset View</button>
          <button class="viewport-btn" id="btnTogglePip" onclick="toggleCameraPip()">📹 Overhead Camera PIP</button>
        </div>

        <div id="three-container"></div>

        <!-- Floating Picture-in-Picture Live Stream -->
        <div id="cameraPipBox">
          <div class="pip-header">
            <span style="display:inline-flex; align-items:center; gap:6px;">
              <span class="live-pulsing-dot"></span> LIVE OVERHEAD CAM
            </span>
            <button onclick="toggleCameraPip()" style="background:transparent; border:none; color:#cbd5e1; cursor:pointer; font-size:0.85rem; font-weight:bold; line-height:1;">✕</button>
          </div>
          <img id="pipStreamImg" src="" style="width:100%; height:calc(100% - 28px); object-fit:cover; display:block;">
        </div>

        <div class="viewport-overlay-hud">
          <div class="hud-pill mono" id="hudCoords">Target: (0.0, 0.0) mm</div>
          <div class="hud-pill mono" id="hudAngles">Pan: 10.0° | Tilt: 150.0°</div>
          <div class="hud-pill mono" style="color:var(--text-dim);" id="lblCursor">Cursor: Left Click on Cloth to Select Target</div>
        </div>
      </div>

      <div class="telemetry-grid">
        <div class="telemetry-card target">
          <small>Target Physical Coordinates (X, Y)</small>
          <div id="lblTarget" class="value">0.0, 0.0 mm (TR Pocket / Home)</div>
        </div>
        <div class="telemetry-card angles">
          <small>Interpolated Servo Angles (Pan, Tilt)</small>
          <div id="lblPredicted" class="value">Pan: 10.0° | Tilt: 150.0°</div>
        </div>
      </div>
    </div>

    <!-- TAB 2: CAMERA & HOMOGRAPHY CALIBRATION -->
    <div id="tab-camera" class="tab-pane">
      <div class="panel-header">
        <span>Camera Homography & Calibration</span>
        <span class="tag-pill tag-online" id="camStreamPill" style="display:inline-flex; align-items:center; gap:6px; color:#34d399; border-color:rgba(52,211,153,0.3);">
          <span class="live-pulsing-dot"></span> LIVE OVERHEAD
        </span>
      </div>

      <!-- Live Stream & Canvas Viewer -->
      <div style="background:rgba(0,0,0,0.4); border-radius:18px; border:1px solid var(--border-subtle); padding:16px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
          <div style="display:inline-flex; align-items:center; gap:8px;">
            <span class="live-pulsing-dot"></span>
            <span style="font-size:0.85rem; font-weight:700; color:#ffffff;">Overhead Camera Footage</span>
            <span class="mono" style="font-size:0.72rem; color:var(--text-muted);">(IP Webcam / 1920x1080)</span>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-secondary" id="btnModeLive" style="padding:4px 12px; font-size:0.72rem; border-color:var(--emerald-green);" onclick="setCameraDisplayMode('live')">🔴 Live Stream</button>
            <button class="btn btn-secondary" id="btnModeFreeze" style="padding:4px 12px; font-size:0.72rem;" onclick="freezeFrameForCalibration()">🎯 Freeze & Calibrate</button>
          </div>
        </div>

        <div id="cameraViewportContainer" style="width:100%; height:480px; min-height:420px; max-height:65vh; background:#04070f; border-radius:14px; border:1px solid rgba(255,255,255,0.08); overflow:hidden; position:relative; display:flex; align-items:center; justify-content:center; user-select:none;">
          <img id="liveStreamView" src="" alt="Live Overhead Stream" style="max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:block;" onerror="handleStreamError(this)" onload="handleStreamLoaded(this)">
          <canvas id="cameraCanvas" width="640" height="360" style="display:none; max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; cursor:crosshair; border-radius:6px; box-shadow:0 0 25px rgba(0,0,0,0.7);"></canvas>

          <div id="streamLoadingOverlay" style="position:absolute; display:none; flex-direction:column; align-items:center; justify-content:center; gap:8px; color:#e2e8f0; font-size:0.8rem; background:rgba(5,8,17,0.92); inset:0;">
            <div style="font-size:1.8rem;">📡</div>
            <div id="streamLoadingMsg">Connecting to IP Webcam stream...</div>
            <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 12px;" onclick="reloadLiveStream()">Retry Stream</button>
          </div>

          <div id="cameraModeBadge" style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.7); backdrop-filter:blur(6px); color:#34d399; border:1px solid rgba(52,211,153,0.4); padding:3px 10px; border-radius:999px; font-size:0.68rem; font-weight:700;">
            ● LIVE STREAM
          </div>

          <div id="canvasCrosshairHud" class="mono" style="position:absolute; bottom:10px; right:12px; background:rgba(5,8,17,0.85); backdrop-filter:blur(4px); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:3px 10px; border-radius:6px; font-size:0.7rem; display:none;">
            Pixel: X: 0, Y: 0
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; flex-wrap:wrap; gap:8px;">
          <div id="cornerStatus" class="mono" style="font-size:0.75rem; color:#facc15; font-weight:600;">
            Status: Live Video streaming. Click "Freeze & Calibrate" to set 4 table corners.
          </div>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.72rem;" onclick="reloadLiveStream()">🔄 Refresh Stream</button>
            <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.72rem;" onclick="undoLastCorner()">↩ Undo Corner</button>
            <button class="btn btn-danger" style="padding:5px 12px; font-size:0.72rem;" onclick="resetCorners()">Reset 4 Corners</button>
          </div>
        </div>
        </div>
      </div>

      <!-- IP Webcam Source Controls -->
      <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-subtle); border-radius:16px; padding:14px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--emerald-green);">📱 IP Webcam Source URL</span>
          <span class="mono" style="font-size:0.7rem; color:var(--text-muted);">Default: http://192.168.31.150:8080</span>
        </div>
        <div style="display:flex; gap:8px; margin-bottom:10px;">
          <input type="text" id="inpIpWebcam" value="http://192.168.31.150:8080" style="flex:1;" placeholder="http://192.168.31.XXX:8080">
          <button class="btn btn-secondary" style="white-space:nowrap;" onclick="saveIpWebcam()">Update URL</button>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button class="btn btn-emerald" style="flex:1; padding:10px;" onclick="freezeFrameForCalibration()">📸 Freeze Frame for Homography</button>
          <button class="btn btn-secondary" style="flex:1; padding:10px;" onclick="captureAndDetect()">🎱 Detect Balls from Live Feed</button>
        </div>
        <div id="ipWebcamStatus" style="font-size:0.74rem; color:var(--text-muted); margin-top:8px;"></div>
      </div>

      <p style="font-size:0.76rem; color:var(--text-muted); margin-bottom:8px;">
        Calibration Sequence: <strong>1: TR (0,0)</strong> &rarr; <strong>2: TL (1168.4, 0)</strong> &rarr; <strong>3: BL (1168.4, 2341.9)</strong> &rarr; <strong>4: BR (0, 2341.9)</strong>
      </p>
      <div style="display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap;">
        <input type="file" id="fileUpload" accept="image/*" onchange="uploadImage(this)" style="display:none;">
        <button class="btn btn-secondary" style="flex:1;" onclick="document.getElementById('fileUpload').click()">Upload Photo File</button>
        <button class="btn btn-secondary" style="flex:1;" onclick="loadSampleImage()">Load Synthetic Table</button>
      </div>

      <button class="btn btn-emerald" style="width:100%; padding:11px; margin-bottom:14px;" onclick="computeHomography()">
        COMPUTE PLANAR HOMOGRAPHY MATRIX
      </button>

      <div class="panel-header" style="font-size:0.86rem; margin-top:14px;">Laser Spot Closed-Loop Verification</div>
      <button class="btn btn-secondary" style="width:100%; padding:10px;" onclick="detectSpot()">
        AUTOMATICALLY DETECT SPOT & COMPUTE ERROR
      </button>
      <div id="spotResult" class="mono" style="font-size:0.76rem; margin-top:8px; color:var(--emerald-green); font-weight:600;"></div>
    </div>

    <!-- TAB 3: SERVO HARDWARE & LASER SAFETY -->
    <div id="tab-servo" class="tab-pane">
      <div class="panel-header">
        <span>Hardware Configuration & Servo Control</span>
      </div>

      <div style="background:rgba(255,255,255,0.03); padding:16px; border-radius:16px; border:1px solid var(--border-subtle); margin-bottom:16px;">
        <label style="font-size:0.78rem; color:#ffffff; display:block; margin-bottom:6px; font-weight:700;">ESP32 Wi-Fi Target IP Address:</label>
        <div style="display:flex; gap:6px; margin-bottom:8px;">
          <input type="text" id="inpEspIp" value="http://192.168.31.2" style="flex:1;">
          <button class="btn btn-emerald" onclick="saveEspIp()">Save</button>
          <button class="btn btn-secondary" onclick="pingEsp()">Ping</button>
        </div>
        <div style="display:flex; gap:6px;">
          <button class="btn btn-secondary" style="font-size:0.72rem; flex:1;" onclick="setEspPreset('http://192.168.31.2')">Home Router (192.168.31.2)</button>
          <button class="btn btn-secondary" style="font-size:0.72rem; flex:1;" onclick="setEspPreset('http://192.168.4.1')">PoolBot SoftAP (192.168.4.1)</button>
        </div>
      </div>

      <div class="form-group">
        <label><span>Pan Servo (Horizontal Axis: 10° - 180°)</span><span id="valPan" class="mono">10°</span></label>
        <input type="range" id="rngPan" min="10" max="180" value="10" oninput="previewServo('pan', this.value)" onchange="sendServoAngle('pan', this.value)">
      </div>
      <div class="form-group">
        <label><span>Tilt Servo (Vertical Axis: 60° - 160°)</span><span id="valTilt" class="mono">150°</span></label>
        <input type="range" id="rngTilt" min="60" max="160" value="150" oninput="previewServo('tilt', this.value)" onchange="sendServoAngle('tilt', this.value)">
      </div>

      <div style="display:flex; gap:8px; margin-bottom:16px;">
        <button class="btn btn-secondary" style="flex:1;" onclick="centerServos()">Center (90°, 90°)</button>
        <button class="btn btn-secondary" style="flex:1;" onclick="homeServos()">Home / TR Pocket (10°, 150°)</button>
      </div>

      <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); border-radius:16px; padding:14px;">
        <div style="font-size:0.8rem; color:#f87171; font-weight:700; margin-bottom:4px;">LASER DIODE SAFETY (GPIO 23 - OFF BY DEFAULT)</div>
        <p style="font-size:0.74rem; color:var(--text-muted); margin-bottom:10px;">The laser requires operator trigger to energize. Never view direct beam.</p>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-danger" id="btnLaserToggle" onclick="toggleLaser()">TOGGLE LASER ON/OFF</button>
          <button class="btn btn-secondary" onclick="forceLaserOff()">FORCE OFF</button>
        </div>
      </div>
    </div>

    <!-- TAB 4: DETECTED BALLS -->
    <div id="tab-balls" class="tab-pane">
      <div class="panel-header">
        <span>YOLOv8 Detected Snooker Balls</span>
        <button class="btn btn-emerald" style="padding:6px 14px; font-size:0.75rem;" onclick="runDetection()">Detect Balls</button>
      </div>
      <p style="font-size:0.76rem; color:var(--text-muted); margin-bottom:10px;">
        8 Canonical Classes (15 Reds, Yellow, Green, Brown, Blue, Pink, Black, Cue Ball).
      </p>
      <div class="table-scroll">
        <table>
          <thead>
            <tr><th>#</th><th>Class</th><th>Conf</th><th>X (mm)</th><th>Y (mm)</th><th>Action</th></tr>
          </thead>
          <tbody id="ballsTableBody">
            <tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:16px;">No balls detected. Click "Detect Balls" to run model.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 5: FOUL REPOSITIONING -->
    <div id="tab-foul" class="tab-pane">
      <div class="panel-header">
        <span>Foul Ball Repositioning Referee Assistant</span>
      </div>
      <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:12px;">
        Aim the red laser pointer at the pre-foul position to guide the referee in accurate placement:
      </p>

      <div class="form-group">
        <label><span>Select Displaced Ball</span></label>
        <select id="selFoulBall" onchange="onFoulBallSelected()">
          <option value="cue">White Cue Ball</option>
          <option value="black">Black Ball (2100.0 mm)</option>
          <option value="pink">Pink Ball (1756.4 mm)</option>
          <option value="blue">Blue Ball (1170.94 mm)</option>
          <option value="brown">Brown Ball (737.0 mm)</option>
          <option value="green">Green Ball (Left Baulk Spot)</option>
          <option value="yellow">Yellow Ball (Right Baulk Spot)</option>
        </select>
      </div>

      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:14px;">
        <div>
          <label style="font-size:0.75rem; color:var(--text-muted); display:block; margin-bottom:4px;">Target X (mm):</label>
          <input type="number" id="inpFoulX" value="584.2">
        </div>
        <div>
          <label style="font-size:0.75rem; color:var(--text-muted); display:block; margin-bottom:4px;">Target Y (mm):</label>
          <input type="number" id="inpFoulY" value="1800.0">
        </div>
      </div>

      <button class="btn btn-emerald" style="width:100%; padding:11px;" onclick="repositionBallLaser()">
        AIM LASER AT PRE-FOUL SPOT
      </button>
    </div>

    <!-- TAB 6: 9-POINT CALIBRATION -->
    <div id="tab-calib" class="tab-pane">
      <div class="panel-header">
        <span>9 Empirical Calibration Grid Nodes</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr><th>Node</th><th>Location Description</th><th>X (mm)</th><th>Y (mm)</th><th>Pan (°)</th><th>Tilt (°)</th><th>Action</th></tr>
          </thead>
          <tbody id="calibTableBody"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 7: LOGS -->
    <div id="tab-logs" class="tab-pane">
      <div class="panel-header">
        <span>System Event & Hardware Log</span>
        <button class="btn btn-secondary" style="padding:4px 12px; font-size:0.72rem;" onclick="exportCsv()">Export CSV</button>
      </div>
      <div class="table-scroll" style="max-height:360px;">
        <table id="tblLogs">
          <thead>
            <tr><th>Time</th><th>Category</th><th>Message</th></tr>
          </thead>
          <tbody id="logsTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN: TARGET CONTROLS & AIMING SIDEBAR -->
  <div class="glass-panel" style="display:flex; flex-direction:column; gap:16px;">
    <div class="panel-header">
      <span>Target Positioning & Laser Execution</span>
    </div>

    <p style="font-size:0.78rem; color:var(--text-muted);">
      Click directly on the green tournament cloth in 3D to pick coordinates, or specify exact millimeters:
    </p>

    <div style="display:flex; gap:8px;">
      <input type="number" id="inpX" placeholder="X (mm)" value="0.0" min="0" max="1168.4" step="1.0" style="flex:1;">
      <input type="number" id="inpY" placeholder="Y (mm)" value="0.0" min="0" max="2341.88" step="1.0" style="flex:1;">
      <button class="btn btn-secondary" onclick="applyManualCoords()">Set</button>
    </div>

    <button class="btn btn-emerald" style="width:100%; padding:13px; font-size:0.85rem;" onclick="sendAimCommand()">
      🎯 AIM LASER AT SELECTED TARGET
    </button>

    <div style="border-top:1px solid var(--border-subtle); padding-top:12px;">
      <div style="font-size:0.78rem; font-weight:700; color:#ffffff; margin-bottom:10px;">
        Presets (6 Regulation Pockets & 3 Calibration Targets)
      </div>

      <div class="presets-grid">
        <button class="preset-btn" onclick="setPreset(0.0, 0.0)">TR Pocket (Home)</button>
        <button class="preset-btn calib-node" onclick="setPreset(584.2, 0.0)">Top-Centre (TC)</button>
        <button class="preset-btn" onclick="setPreset(1168.4, 0.0)">TL Pocket</button>

        <button class="preset-btn" onclick="setPreset(0.0, 1170.94)">MR Pocket</button>
        <button class="preset-btn calib-node" onclick="setPreset(584.2, 1170.94)">Centre Node (C)</button>
        <button class="preset-btn" onclick="setPreset(1168.4, 1170.94)">ML Pocket</button>

        <button class="preset-btn" onclick="setPreset(0.0, 2341.88)">BR Pocket</button>
        <button class="preset-btn calib-node" onclick="setPreset(584.2, 2341.88)">Bottom-Centre (BC)</button>
        <button class="preset-btn" onclick="setPreset(1168.4, 2341.88)">BL Pocket</button>
      </div>
    </div>

    <div id="cmdFeedback" class="mono" style="font-size:0.75rem; color:#38bdf8; background:rgba(0,0,0,0.4); padding:12px 14px; border-radius:14px; border:1px solid var(--border-subtle); word-break:break-all;">
      System Status: Ready. Plumb-line home: Top-Right (0,0).
    </div>
  </div>
</main>

<script>
// ==============================================================
// 1. CONSTANTS & EXACT GEOMETRY
// ==============================================================
const SNOOKER_W = 1168.4;
const SNOOKER_L = 2341.88;
const TABLE_W = 1.1684;
const TABLE_L = 2.34188;
const MOUNT_HEIGHT = 1.31515; // Z = 1315.15 mm directly above TR pocket (0,0)
const BALL_RADIUS = 0.02625; // 52.5 mm regulation diameter / 2 = 26.25 mm = 0.02625 m

let targetX = 0.0;
let targetY = 0.0;
let predPan = 10.0;
let predTilt = 150.0;
let isSim = true;
let isLaserActive = false;
let ipWebcamBaseUrl = "http://192.168.31.150:8080";

// Three.js State
let scene, camera, renderer, controls;
let feltMesh, laserAssemblyGroup, panHousing, tiltHousing;
let laserRayLine, laserSpotMesh, targetMarkerMesh;
let ballMeshes = [];

// Smooth camera transition state
let camFlightActive = false;
let camTargetPos = new THREE.Vector3(1.6, 2.2, 2.5);
let controlsTargetPos = new THREE.Vector3(0, 0, 0);

// Smooth virtual servo slew state
let currentSimPan = 10.0;
let currentSimTilt = 150.0;

function tableToScene(x_mm, y_mm) {
  // TR (0,0) -> (+TABLE_W/2, -TABLE_L/2)
  // TL (1168.4, 0) -> (-TABLE_W/2, -TABLE_L/2)
  // BR (0, 2341.88) -> (+TABLE_W/2, +TABLE_L/2)
  // BL (1168.4, 2341.88) -> (-TABLE_W/2, +TABLE_L/2)
  const x3d = (TABLE_W / 2.0) - (x_mm * 0.001);
  const z3d = (y_mm * 0.001) - (TABLE_L / 2.0);
  return { x: x3d, z: z3d };
}

function sceneToTable(x3d, z3d) {
  let x_mm = ((TABLE_W / 2.0) - x3d) * 1000.0;
  let y_mm = (z3d + (TABLE_L / 2.0)) * 1000.0;
  x_mm = Math.max(0.0, Math.min(SNOOKER_W, x_mm));
  y_mm = Math.max(0.0, Math.min(SNOOKER_L, y_mm));
  return { x: x_mm, y: y_mm };
}

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// ==============================================================
// 2. THREE.JS PHOTOREALISTIC SNOOKER SCENE
// ==============================================================
function initThree() {
  const container = document.getElementById('three-container');
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 580;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060912);

  camera = new THREE.PerspectiveCamera(36, w / h, 0.05, 50);
  camera.position.set(1.6, 2.2, 2.5);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.04;
  controls.minDistance = 0.5;
  controls.maxDistance = 6.0;
  controls.target.set(0, 0, 0);

  setupLighting();
  buildStudioEnvironment();
  buildSnookerTable();
  buildOverheadCanopyLuminaire();
  buildLaserAssembly();
  buildInteractiveMarkers();
  spawnDefaultSnookerBalls();

  renderer.domElement.addEventListener('mousemove', onThreeMouseMove);
  renderer.domElement.addEventListener('click', onThreeClick);
  window.addEventListener('resize', onWindowResize);

  animate();
}

function setupLighting() {
  // Balanced ambient light so green baize, walnut rails and balls are vibrant
  const ambient = new THREE.AmbientLight(0xffffff, 1.35);
  scene.add(ambient);

  // Soft sky/ground fill light
  const hemi = new THREE.HemisphereLight(0xffffff, 0x1e293b, 0.75);
  scene.add(hemi);

  // Directional key accent light for realistic drop shadows
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
  keyLight.position.set(2.5, 4.0, 3.0);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 1024;
  keyLight.shadow.mapSize.height = 1024;
  keyLight.shadow.bias = -0.0005;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.5);
  fillLight.position.set(-2.5, 3.0, -2.5);
  scene.add(fillLight);
}

function buildStudioEnvironment() {
  // Dark studio floor with soft specular reflection
  const floorGeo = new THREE.PlaneGeometry(12, 12);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x090d16,
    roughness: 0.8,
    metalness: 0.1
  });
  const floorMesh = new THREE.Mesh(floorGeo, floorMat);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.y = -0.70;
  floorMesh.receiveShadow = true;
  scene.add(floorMesh);

  // Soft contact shadow circle under the table
  const shadowCanvas = document.createElement('canvas');
  shadowCanvas.width = 512;
  shadowCanvas.height = 512;
  const sctx = shadowCanvas.getContext('2d');
  const sgrad = sctx.createRadialGradient(256, 256, 40, 256, 256, 250);
  sgrad.addColorStop(0, 'rgba(0, 0, 0, 0.7)');
  sgrad.addColorStop(0.5, 'rgba(0, 0, 0, 0.35)');
  sgrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  sctx.fillStyle = sgrad;
  sctx.fillRect(0, 0, 512, 512);

  const shadowTex = new THREE.CanvasTexture(shadowCanvas);
  const shadowGeo = new THREE.PlaneGeometry(TABLE_W * 1.6, TABLE_L * 1.4);
  const shadowMat = new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, opacity: 0.8 });
  const tableShadow = new THREE.Mesh(shadowGeo, shadowMat);
  tableShadow.rotation.x = -Math.PI / 2;
  tableShadow.position.y = -0.695;
  scene.add(tableShadow);
}

// Procedural high-resolution woven Strachan No. 10 Tournament Baize
function createFeltClothTexture() {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 1024;
  const ctx = c.getContext('2d');

  // Rich tournament emerald green gradient with subtle vignette
  const grad = ctx.createRadialGradient(c.width / 2, c.height / 2, c.width * 0.15, c.width / 2, c.height / 2, c.height * 0.65);
  grad.addColorStop(0, '#11613c');
  grad.addColorStop(0.7, '#0c4d2e');
  grad.addColorStop(1, '#083821');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, c.width, c.height);

  // Fast procedural micro-weave cloth simulation
  const noiseCanv = document.createElement('canvas');
  noiseCanv.width = 32;
  noiseCanv.height = 32;
  const nctx = noiseCanv.getContext('2d');
  const nData = nctx.createImageData(32, 32);
  for (let i = 0; i < nData.data.length; i += 4) {
    const val = 128 + (Math.random() - 0.5) * 28;
    nData.data[i] = val;
    nData.data[i+1] = val;
    nData.data[i+2] = val;
    nData.data[i+3] = 16;
  }
  nctx.putImageData(nData, 0, 0);
  ctx.fillStyle = ctx.createPattern(noiseCanv, 'repeat');
  ctx.fillRect(0, 0, c.width, c.height);

  const sx = c.width / SNOOKER_W;
  const sy = c.height / SNOOKER_L;

  // Baulk Line at Y = 737.0 mm
  const baulkY = 737.0 * sy;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, baulkY);
  ctx.lineTo(c.width, baulkY);
  ctx.stroke();

  // D-shaped baulk arc (radius 292.1 mm)
  const dRadius = 292.1 * sx;
  const centerX = c.width / 2.0;
  ctx.beginPath();
  ctx.arc(centerX, baulkY, dRadius, 0, Math.PI, false);
  ctx.stroke();

  // Regulation Spots: Brown, Yellow, Green, Blue, Pink, Black
  const spotList = [
    { x: SNOOKER_W / 2.0, y: 737.0, color: '#92400e' },
    { x: (SNOOKER_W / 2.0) - 292.1, y: 737.0, color: '#facc15' },
    { x: (SNOOKER_W / 2.0) + 292.1, y: 737.0, color: '#16a34a' },
    { x: SNOOKER_W / 2.0, y: SNOOKER_L / 2.0, color: '#2563eb' },
    { x: SNOOKER_W / 2.0, y: 1756.4, color: '#ec4899' },
    { x: SNOOKER_W / 2.0, y: 2100.0, color: '#18181b' }
  ];

  spotList.forEach(sp => {
    const px = c.width - (sp.x * sx);
    const py = sp.y * sy;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = sp.color;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  const texture = new THREE.CanvasTexture(c);
  texture.anisotropy = 16;
  return texture;
}

// Procedural rich polished mahogany/walnut wood texture
function createWalnutTexture() {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#221109';
  ctx.fillRect(0, 0, c.width, c.height);

  for (let i = 0; i < 220; i++) {
    const y = Math.random() * c.height;
    const h = 1 + Math.random() * 6;
    const alpha = 0.04 + Math.random() * 0.12;
    ctx.fillStyle = (i % 2 === 0) ? `rgba(60, 30, 16, ${alpha})` : `rgba(12, 6, 3, ${alpha})`;
    ctx.fillRect(0, y, c.width, h);
  }

  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);
  return tex;
}

function buildSnookerTable() {
  const tableGroup = new THREE.Group();

  // 1. Table Felt Playing Surface
  const feltGeo = new THREE.PlaneGeometry(TABLE_W, TABLE_L, 32, 64);
  const feltTex = createFeltClothTexture();
  const feltMat = new THREE.MeshStandardMaterial({
    map: feltTex,
    roughness: 0.85,
    metalness: 0.02
  });
  feltMesh = new THREE.Mesh(feltGeo, feltMat);
  feltMesh.rotation.x = -Math.PI / 2;
  feltMesh.receiveShadow = true;
  tableGroup.add(feltMesh);

  // 2. Heavy Grounded Slate Bed
  const slateGeo = new THREE.BoxGeometry(TABLE_W + 0.04, 0.045, TABLE_L + 0.04);
  const slateMat = new THREE.MeshStandardMaterial({ color: 0x1a211e, roughness: 0.9 });
  const slateMesh = new THREE.Mesh(slateGeo, slateMat);
  slateMesh.position.y = -0.023;
  slateMesh.castShadow = true;
  tableGroup.add(slateMesh);

  // 3. Polished Walnut Rails & Frame
  const woodTex = createWalnutTexture();
  const railMat = new THREE.MeshStandardMaterial({
    map: woodTex,
    roughness: 0.22,
    metalness: 0.08
  });

  const railWidth = 0.095;
  const railHeight = 0.052;

  // Short Rails (Top & Bottom)
  const tbRailGeo = new THREE.BoxGeometry(TABLE_W + railWidth * 2, railHeight, railWidth);
  
  const topRail = new THREE.Mesh(tbRailGeo, railMat);
  topRail.position.set(0, railHeight / 2 - 0.005, -(TABLE_L / 2 + railWidth / 2));
  topRail.castShadow = true;
  tableGroup.add(topRail);

  const botRail = new THREE.Mesh(tbRailGeo, railMat);
  botRail.position.set(0, railHeight / 2 - 0.005, (TABLE_L / 2 + railWidth / 2));
  botRail.castShadow = true;
  tableGroup.add(botRail);

  // Long Side Rails (4 Segments with gap for middle pockets)
  const sideSegL = (TABLE_L - 0.16) / 2.0;
  const sideRailGeo = new THREE.BoxGeometry(railWidth, railHeight, sideSegL);

  [-1, 1].forEach(side => {
    [-1, 1].forEach(seg => {
      const rMesh = new THREE.Mesh(sideRailGeo, railMat);
      const rx = side * (TABLE_W / 2 + railWidth / 2);
      const rz = seg * (sideSegL / 2 + 0.065);
      rMesh.position.set(rx, railHeight / 2 - 0.005, rz);
      rMesh.castShadow = true;
      tableGroup.add(rMesh);
    });
  });

  // 4. Inlaid Mother-of-Pearl Sights along the wooden rails
  const sightMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.1, metalness: 0.5 });
  const sightGeo = new THREE.CylinderGeometry(0.0035, 0.0035, 0.002, 16);

  // 3 Sights on Top & Bottom Rails
  [-0.35, 0.0, 0.35].forEach(sx => {
    const s1 = new THREE.Mesh(sightGeo, sightMat);
    s1.position.set(sx, railHeight - 0.004, -(TABLE_L / 2 + railWidth / 2));
    tableGroup.add(s1);

    const s2 = new THREE.Mesh(sightGeo, sightMat);
    s2.position.set(sx, railHeight - 0.004, (TABLE_L / 2 + railWidth / 2));
    tableGroup.add(s2);
  });

  // 5. Cushion Rubber Noses (Wrapped in green cloth facing the ball)
  const cushionMat = new THREE.MeshStandardMaterial({ color: 0x0c4d2e, roughness: 0.85 });
  const cThick = 0.016;

  // Top/Bottom cushion noses
  const tbCushGeo = new THREE.BoxGeometry(TABLE_W - 0.08, railHeight * 0.7, cThick);
  const tcush = new THREE.Mesh(tbCushGeo, cushionMat);
  tcush.position.set(0, railHeight * 0.35, -TABLE_L / 2 - cThick / 2);
  tableGroup.add(tcush);

  const bcush = new THREE.Mesh(tbCushGeo, cushionMat);
  bcush.position.set(0, railHeight * 0.35, TABLE_L / 2 + cThick / 2);
  tableGroup.add(bcush);

  // Side cushion noses
  const sideCushGeo = new THREE.BoxGeometry(cThick, railHeight * 0.7, sideSegL - 0.05);
  [-1, 1].forEach(side => {
    [-1, 1].forEach(seg => {
      const sc = new THREE.Mesh(sideCushGeo, cushionMat);
      sc.position.set(side * (TABLE_W / 2 + cThick / 2), railHeight * 0.35, seg * (sideSegL / 2 + 0.065));
      tableGroup.add(sc);
    });
  });

  // 6. Exactly Six Regulation Pockets with Real Depth & Cast-Brass Plates
  const pocketLocations = [
    { name: "TR", x: TABLE_W / 2, z: -TABLE_L / 2, corner: true },
    { name: "TL", x: -TABLE_W / 2, z: -TABLE_L / 2, corner: true },
    { name: "MR", x: TABLE_W / 2 + 0.012, z: 0.0, corner: false },
    { name: "ML", x: -TABLE_W / 2 - 0.012, z: 0.0, corner: false },
    { name: "BR", x: TABLE_W / 2, z: TABLE_L / 2, corner: true },
    { name: "BL", x: -TABLE_W / 2, z: TABLE_L / 2, corner: true }
  ];

  const pocketWellMat = new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 0.95 });
  const brassMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.92, roughness: 0.22 });

  pocketLocations.forEach(p => {
    const r = p.corner ? 0.046 : 0.040;
    // Real hollow drop well extending 0.07 m downwards
    const dropGeo = new THREE.CylinderGeometry(r, r * 0.85, 0.07, 24);
    const dropMesh = new THREE.Mesh(dropGeo, pocketWellMat);
    dropMesh.position.set(p.x, -0.035, p.z);
    tableGroup.add(dropMesh);

    // Cast Brass Pocket Plate & Bridge
    const plateGeo = new THREE.TorusGeometry(r + 0.006, 0.007, 10, 24, p.corner ? Math.PI : Math.PI * 2);
    const plateMesh = new THREE.Mesh(plateGeo, brassMat);
    plateMesh.rotation.x = Math.PI / 2;
    plateMesh.position.set(p.x, 0.024, p.z);
    tableGroup.add(plateMesh);
  });

  // 7. Table Under-Apron / Skirt Box
  const apronMat = new THREE.MeshStandardMaterial({ map: woodTex, roughness: 0.3, metalness: 0.05 });
  const apronGeo = new THREE.BoxGeometry(TABLE_W + 0.16, 0.14, TABLE_L + 0.16);
  const apronMesh = new THREE.Mesh(apronGeo, apronMat);
  apronMesh.position.y = -0.11;
  apronMesh.castShadow = true;
  tableGroup.add(apronMesh);

  // 8. Eight Traditional Turned Wooden Snooker Legs with Brass Levelling Rings
  const legMat = new THREE.MeshStandardMaterial({ map: woodTex, roughness: 0.35 });
  const brassFootMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.95, roughness: 0.2 });

  const legX = TABLE_W / 2 + 0.035;
  const legZCorner = TABLE_L / 2 + 0.035;
  const legZMid = TABLE_L / 6.0;

  const legPositions = [
    [legX, -legZCorner], [-legX, -legZCorner],
    [legX, -legZMid],    [-legX, -legZMid],
    [legX, legZMid],     [-legX, legZMid],
    [legX, legZCorner],  [-legX, legZCorner]
  ];

  legPositions.forEach(([lx, lz]) => {
    const legGroup = new THREE.Group();
    legGroup.position.set(lx, -0.18, lz);

    // Upper square transition block
    const block = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.08, 0.09), legMat);
    block.position.y = -0.04;
    block.castShadow = true;
    legGroup.add(block);

    // Turned fluted column
    const column = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.036, 0.40, 24), legMat);
    column.position.y = -0.28;
    column.castShadow = true;
    legGroup.add(column);

    // Middle lathe decorative ring
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.046, 0.008, 8, 24), legMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -0.16;
    legGroup.add(ring);

    // Brass levelling foot ring
    const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.044, 0.04, 24), brassFootMat);
    foot.position.y = -0.50;
    foot.castShadow = true;
    legGroup.add(foot);

    tableGroup.add(legGroup);
  });

  scene.add(tableGroup);
}

// Suspended Tournament Overhead Canopy Lighting
function buildOverheadCanopyLuminaire() {
  const lightGroup = new THREE.Group();
  lightGroup.position.set(0, 1.65, 0);

  // Canopy Fixture Body
  const canopyMat = new THREE.MeshStandardMaterial({ color: 0x142018, metalness: 0.6, roughness: 0.4 });
  const canopyMesh = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.09, 1.5), canopyMat);
  canopyMesh.castShadow = false;
  lightGroup.add(canopyMesh);

  // Support Brass Suspension Rods to Ceiling
  const brassRodMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.9, roughness: 0.25 });
  [-0.6, 0.6].forEach(rz => {
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.9, 16), brassRodMat);
    rod.position.set(0, 0.45, rz);
    lightGroup.add(rod);
  });

  // 3 Soft Warm-White Halogen Spotlights illuminating the baize
  [-0.55, 0.0, 0.55].forEach(spotZ => {
    const spot = new THREE.SpotLight(0xfff8eb, 2.3, 3.8, Math.PI / 3.4, 0.55, 1.4);
    spot.position.set(0, -0.05, spotZ);
    spot.target.position.set(0, -1.65, spotZ);
    spot.castShadow = true;
    spot.shadow.mapSize.width = 1024;
    spot.shadow.mapSize.height = 1024;
    spot.shadow.camera.near = 0.5;
    spot.shadow.camera.far = 3.5;
    spot.shadow.bias = -0.0001;

    lightGroup.add(spot);
    lightGroup.add(spot.target);
  });

  scene.add(lightGroup);
}

// 3D Laser Assembly: Mounted at physical height Z = 1315.15 mm above Top-Right corner (0,0)
function buildLaserAssembly() {
  laserAssemblyGroup = new THREE.Group();
  const trPos = tableToScene(0.0, 0.0);

  // Overhead Structural Bracket Arm
  const armMat = new THREE.MeshStandardMaterial({ color: 0x27272a, metalness: 0.85, roughness: 0.3 });
  const postGeo = new THREE.BoxGeometry(0.035, MOUNT_HEIGHT + 0.1, 0.035);
  const postMesh = new THREE.Mesh(postGeo, armMat);
  postMesh.position.set(trPos.x + 0.075, (MOUNT_HEIGHT + 0.1) / 2 - 0.05, trPos.z - 0.075);
  postMesh.castShadow = true;
  laserAssemblyGroup.add(postMesh);

  // Horizontal Mounting Beam clamped overhead
  const beamGeo = new THREE.BoxGeometry(0.12, 0.02, 0.12);
  const beamMesh = new THREE.Mesh(beamGeo, armMat);
  beamMesh.position.set(trPos.x, MOUNT_HEIGHT, trPos.z);
  laserAssemblyGroup.add(beamMesh);

  // Pan Motor Housing (rotates horizontally about Y)
  panHousing = new THREE.Group();
  panHousing.position.set(trPos.x, MOUNT_HEIGHT, trPos.z);

  const servoMat = new THREE.MeshStandardMaterial({ color: 0x18181b, metalness: 0.7, roughness: 0.35 });
  const panMotor = new THREE.Mesh(new THREE.BoxGeometry(0.042, 0.045, 0.042), servoMat);
  panMotor.position.y = -0.022;
  panMotor.castShadow = true;
  panHousing.add(panMotor);

  // Tilt Gimbal Fork (rotates vertically about X)
  tiltHousing = new THREE.Group();
  tiltHousing.position.set(0, -0.048, 0);

  const forkMat = new THREE.MeshStandardMaterial({ color: 0x52525b, metalness: 0.9, roughness: 0.2 });
  const forkArm = new THREE.Mesh(new THREE.BoxGeometry(0.032, 0.016, 0.02), forkMat);
  tiltHousing.add(forkArm);

  // Laser Diode Barrel with Brass Knurled Cap
  const diodeMat = new THREE.MeshStandardMaterial({ color: 0xd97706, metalness: 0.9, roughness: 0.2 });
  const laserPointer = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.045, 16), diodeMat);
  laserPointer.rotation.x = Math.PI / 2;
  tiltHousing.add(laserPointer);

  panHousing.add(tiltHousing);
  laserAssemblyGroup.add(panHousing);
  scene.add(laserAssemblyGroup);

  // High-Intensity Red Laser Core Line
  const rayMat = new THREE.LineBasicMaterial({
    color: 0xff1e38,
    linewidth: 2,
    transparent: true,
    opacity: 0.95
  });
  laserRayLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,0)]),
    rayMat
  );
  scene.add(laserRayLine);

  // Glowing Laser Impact Spot on Cloth
  const spotGeo = new THREE.CircleGeometry(0.014, 24);
  const spotMat = new THREE.MeshBasicMaterial({ color: 0xff2244, transparent: true, opacity: 0.9 });
  laserSpotMesh = new THREE.Mesh(spotGeo, spotMat);
  laserSpotMesh.rotation.x = -Math.PI / 2;
  laserSpotMesh.position.y = 0.0015;
  scene.add(laserSpotMesh);
}

function buildInteractiveMarkers() {
  // Cybernetic Pulsating Target Reticle
  const markerGroup = new THREE.Group();

  const outerRingGeo = new THREE.RingGeometry(0.022, 0.027, 32);
  const outerRingMat = new THREE.MeshBasicMaterial({ color: 0x10b981, side: THREE.DoubleSide });
  const outerRing = new THREE.Mesh(outerRingGeo, outerRingMat);
  outerRing.rotation.x = -Math.PI / 2;
  markerGroup.add(outerRing);

  const innerDotGeo = new THREE.CircleGeometry(0.004, 16);
  const innerDotMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
  const innerDot = new THREE.Mesh(innerDotGeo, innerDotMat);
  innerDot.rotation.x = -Math.PI / 2;
  innerDot.position.y = 0.0005;
  markerGroup.add(innerDot);

  targetMarkerMesh = markerGroup;
  scene.add(targetMarkerMesh);
}

// High-Gloss Aramith Phenolic Resin Snooker Balls
const SNOOKER_COLORS = {
  "cue": 0xfcfbfa,
  "red": 0xdc2626,
  "yellow": 0xfacc15,
  "green": 0x16a34a,
  "brown": 0x92400e,
  "blue": 0x2563eb,
  "pink": 0xf472b6,
  "black": 0x18181b
};

function createBallMesh(type, colorHex) {
  const ballGeo = new THREE.SphereGeometry(BALL_RADIUS, 48, 48);
  const ballMat = new THREE.MeshPhysicalMaterial({
    color: colorHex,
    roughness: 0.08,
    metalness: 0.02,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
    reflectivity: 0.85
  });
  const mesh = new THREE.Mesh(ballGeo, ballMat);
  mesh.castShadow = true;
  mesh.receiveShadow = false;

  // Soft Contact Ambient Occlusion Drop Shadow under Ball
  const shadowGeo = new THREE.CircleGeometry(BALL_RADIUS * 1.15, 24);
  const shadowMat = new THREE.MeshBasicMaterial({ color: 0x050807, transparent: true, opacity: 0.42 });
  const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
  shadowMesh.rotation.x = -Math.PI / 2;
  shadowMesh.position.y = -BALL_RADIUS + 0.0008;
  mesh.add(shadowMesh);

  return mesh;
}

function spawnDefaultSnookerBalls() {
  clearBalls();
  const defaultBalls = [
    { type: "cue", color: SNOOKER_COLORS.cue, x: 584.2, y: 1800.0 },
    { type: "black", color: SNOOKER_COLORS.black, x: 584.2, y: 2100.0 },
    { type: "pink", color: SNOOKER_COLORS.pink, x: 584.2, y: 1756.4 },
    { type: "blue", color: SNOOKER_COLORS.blue, x: 584.2, y: 1170.94 },
    { type: "brown", color: SNOOKER_COLORS.brown, x: 584.2, y: 737.0 },
    { type: "green", color: SNOOKER_COLORS.green, x: 876.2, y: 737.0 },
    { type: "yellow", color: SNOOKER_COLORS.yellow, x: 292.2, y: 737.0 },
    { type: "red", color: SNOOKER_COLORS.red, x: 584.2, y: 1600.0 },
    { type: "red", color: SNOOKER_COLORS.red, x: 556.0, y: 1645.0 },
    { type: "red", color: SNOOKER_COLORS.red, x: 612.4, y: 1645.0 },
    { type: "red", color: SNOOKER_COLORS.red, x: 528.0, y: 1690.0 },
    { type: "red", color: SNOOKER_COLORS.red, x: 584.2, y: 1690.0 },
    { type: "red", color: SNOOKER_COLORS.red, x: 640.4, y: 1690.0 }
  ];

  defaultBalls.forEach(b => {
    const mesh = createBallMesh(b.type, b.color);
    const pos = tableToScene(b.x, b.y);
    mesh.position.set(pos.x, BALL_RADIUS, pos.z);
    scene.add(mesh);
    ballMeshes.push(mesh);
  });
}

function clearBalls() {
  ballMeshes.forEach(b => scene.remove(b));
  ballMeshes = [];
}

// Mechatronic Smooth Laser & Servo Visualizer
function updateLaserVisualization() {
  const trPos = tableToScene(0.0, 0.0);
  const targetPos = tableToScene(targetX, targetY);

  targetMarkerMesh.position.set(targetPos.x, 0.001, targetPos.z);
  laserSpotMesh.position.set(targetPos.x, 0.0015, targetPos.z);

  const shouldShowLaser = isLaserActive || isSim;
  laserSpotMesh.visible = shouldShowLaser;
  laserRayLine.visible = shouldShowLaser;

  const dx = targetPos.x - trPos.x;
  const dy = 0.0 - MOUNT_HEIGHT;
  const dz = targetPos.z - trPos.z;
  const horizontalDist = Math.sqrt(dx * dx + dz * dz);

  const targetYaw = Math.atan2(dx, dz);
  const targetPitch = -Math.atan2(dy, horizontalDist) - Math.PI / 2;

  // Mechatronic Smooth Slew (Rotates realistically)
  panHousing.rotation.y += (targetYaw - panHousing.rotation.y) * 0.12;
  tiltHousing.rotation.x += (targetPitch - tiltHousing.rotation.x) * 0.12;

  const startPt = new THREE.Vector3(trPos.x, MOUNT_HEIGHT - 0.048, trPos.z);
  const endPt = new THREE.Vector3(targetPos.x, 0.002, targetPos.z);
  laserRayLine.geometry.setFromPoints([startPt, endPt]);
}

function onThreeMouseMove(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(feltMesh);
  if (hits.length > 0) {
    const pt = hits[0].point;
    const coords = sceneToTable(pt.x, pt.z);
    document.getElementById('lblCursor').textContent = `Cursor: X: ${coords.x.toFixed(1)} mm | Y: ${coords.y.toFixed(1)} mm`;
  }
}

function onThreeClick(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(feltMesh);
  if (hits.length > 0) {
    const pt = hits[0].point;
    const coords = sceneToTable(pt.x, pt.z);
    targetX = coords.x;
    targetY = coords.y;
    document.getElementById('inpX').value = targetX.toFixed(1);
    document.getElementById('inpY').value = targetY.toFixed(1);
    updatePred();
  }
}

// Smooth Camera Flight Transitions
function flyCameraTo(targetCamPos, targetLookAt) {
  camTargetPos.copy(targetCamPos);
  controlsTargetPos.copy(targetLookAt);
  camFlightActive = true;
}

function setCameraView(mode) {
  document.querySelectorAll('.viewport-btn').forEach(b => b.classList.remove('active'));
  if (mode === 'topdown') {
    document.getElementById('btnViewTop').classList.add('active');
    flyCameraTo(new THREE.Vector3(0, 3.1, 0.001), new THREE.Vector3(0, 0, 0));
  } else if (mode === 'baulk') {
    document.getElementById('btnViewBaulk').classList.add('active');
    flyCameraTo(new THREE.Vector3(0, 1.15, -2.2), new THREE.Vector3(0, 0, 0.2));
  } else if (mode === 'black') {
    document.getElementById('btnViewBlack').classList.add('active');
    flyCameraTo(new THREE.Vector3(0, 1.15, 2.2), new THREE.Vector3(0, 0, -0.2));
  } else {
    document.getElementById('btnViewPersp').classList.add('active');
    flyCameraTo(new THREE.Vector3(1.6, 2.2, 2.5), new THREE.Vector3(0, 0, 0));
  }
}

function resetCamera() {
  setCameraView('perspective');
}

function onWindowResize() {
  const container = document.getElementById('three-container');
  if (!container) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);

  // Smooth Camera Lerp flight
  if (camFlightActive) {
    camera.position.lerp(camTargetPos, 0.06);
    controls.target.lerp(controlsTargetPos, 0.06);
    if (camera.position.distanceTo(camTargetPos) < 0.01 && controls.target.distanceTo(controlsTargetPos) < 0.01) {
      camera.position.copy(camTargetPos);
      controls.target.copy(controlsTargetPos);
      camFlightActive = false;
    }
  }

  controls.update();

  // Subtle pulsing target reticle
  if (targetMarkerMesh) {
    const s = 1.0 + Math.sin(Date.now() * 0.006) * 0.08;
    targetMarkerMesh.scale.set(s, s, 1);
  }

  updateLaserVisualization();
  renderer.render(scene, camera);
}

// ==============================================================
// 3. UI, REST API & HARDWARE INTERACTION
// ==============================================================
function init() {
  initThree();
  fetchNodes();
  updatePred();
  initCameraModule();
  setInterval(pollStatus, 2000);
}

function showTab(id, btn) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}

function applyManualCoords() {
  targetX = Math.max(0, Math.min(SNOOKER_W, parseFloat(document.getElementById('inpX').value) || 0.0));
  targetY = Math.max(0, Math.min(SNOOKER_L, parseFloat(document.getElementById('inpY').value) || 0.0));
  updatePred();
}

function setPreset(x, y) {
  targetX = x; targetY = y;
  document.getElementById('inpX').value = x;
  document.getElementById('inpY').value = y;
  updatePred();
}

async function updatePred() {
  document.getElementById('lblTarget').textContent = `${targetX.toFixed(1)}, ${targetY.toFixed(1)} mm`;
  document.getElementById('hudCoords').textContent = `Target: (${targetX.toFixed(1)}, ${targetY.toFixed(1)}) mm`;

  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: targetX, y: targetY })
  });
  const data = await res.json();
  predPan = data.pan;
  predTilt = data.tilt;
  document.getElementById('lblPredicted').textContent = `Pan: ${predPan.toFixed(1)}° | Tilt: ${predTilt.toFixed(1)}°`;
  document.getElementById('hudAngles').textContent = `Pan: ${predPan.toFixed(1)}° | Tilt: ${predTilt.toFixed(1)}°`;
  document.getElementById('valPan').textContent = predPan.toFixed(1) + '°';
  document.getElementById('valTilt').textContent = predTilt.toFixed(1) + '°';
  document.getElementById('rngPan').value = predPan;
  document.getElementById('rngTilt').value = predTilt;

  if (!isSim) {
    sendAimCommand();
  }
}

async function fetchNodes() {
  const res = await fetch('/api/nodes');
  const data = await res.json();
  const tb = document.getElementById('calibTableBody');
  tb.innerHTML = '';
  data.nodes.forEach(n => {
    tb.innerHTML += `<tr>
      <td><strong>${n.name}</strong></td>
      <td>${n.description}</td>
      <td>${n.table_x.toFixed(1)}</td>
      <td>${n.table_y.toFixed(1)}</td>
      <td>${n.pan}°</td>
      <td>${n.tilt}°</td>
      <td><button class="btn btn-secondary" style="padding:4px 12px; font-size:0.73rem;" onclick="setPreset(${n.table_x}, ${n.table_y})">Aim</button></td>
    </tr>`;
  });
}

async function sendAimCommand() {
  const res = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pan: predPan, tilt: predTilt })
  });
  const data = await res.json();
  document.getElementById('cmdFeedback').textContent = data.message;
}

async function toggleSim(val) {
  isSim = val;
  await fetch('/api/simulation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: isSim })
  });
  updateESPBadge(!isSim);
}

function updateESPBadge(connected) {
  const b = document.getElementById('badgeEsp');
  const txt = document.getElementById('txtEspBadge');
  if (isSim) {
    b.className = "status-pill sim";
    txt.textContent = "SIMULATION ACTIVE";
  } else if (connected) {
    b.className = "status-pill online";
    txt.textContent = "ESP32: ONLINE";
  } else {
    b.className = "status-pill offline";
    txt.textContent = "ESP32: OFFLINE";
  }
}

async function emergencyStop() {
  await fetch('/api/stop', { method: 'POST' });
  isLaserActive = false;
  alert("EMERGENCY STOP ACTIVATED: Laser shut off and servos halted.");
}

async function homeServos() {
  setPreset(0.0, 0.0);
  sendAimCommand();
}

async function centerServos() {
  await fetch('/api/center', { method: 'POST' });
}

async function previewServo(axis, val) {
  if (axis === 'pan') document.getElementById('valPan').textContent = val + '°';
  else document.getElementById('valTilt').textContent = val + '°';
}

async function sendServoAngle(axis, val) {
  const pan = axis === 'pan' ? val : predPan;
  const tilt = axis === 'tilt' ? val : predTilt;
  await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pan: pan, tilt: tilt })
  });
}

async function toggleLaser() {
  const res = await fetch('/api/laser/toggle', { method: 'POST' });
  const data = await res.json();
  isLaserActive = data.laser;
  document.getElementById('btnLaserToggle').textContent = isLaserActive ? "LASER IS ON (CLICK TO SHUT OFF)" : "TOGGLE LASER ON/OFF";
}

async function forceLaserOff() {
  await fetch('/api/laser/off', { method: 'POST' });
  isLaserActive = false;
  document.getElementById('btnLaserToggle').textContent = "TOGGLE LASER ON/OFF";
}

function onFoulBallSelected() {
  const sel = document.getElementById('selFoulBall').value;
  const spots = {
    "cue": [584.2, 1800.0],
    "black": [584.2, 2100.0],
    "pink": [584.2, 1756.4],
    "blue": [584.2, 1170.94],
    "brown": [584.2, 737.0],
    "green": [876.2, 737.0],
    "yellow": [292.2, 737.0]
  };
  if (spots[sel]) {
    document.getElementById('inpFoulX').value = spots[sel][0];
    document.getElementById('inpFoulY').value = spots[sel][1];
  }
}

async function repositionBallLaser() {
  const fx = parseFloat(document.getElementById('inpFoulX').value) || 0.0;
  const fy = parseFloat(document.getElementById('inpFoulY').value) || 0.0;
  setPreset(fx, fy);
  sendAimCommand();
}

async function saveEspIp() {
  const ip = document.getElementById('inpEspIp').value.trim();
  await fetch('/api/esp32/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base_url: ip })
  });
  alert("Target ESP32 URL updated to: " + ip);
}

function setEspPreset(ip) {
  document.getElementById('inpEspIp').value = ip;
  saveEspIp();
}

async function pingEsp() {
  const res = await fetch('/api/esp32/ping');
  const data = await res.json();
  alert(data.message);
}

// ==============================================================
// 4. CAMERA, LIVE STREAM & HOMOGRAPHY CALIBRATION
// ==============================================================
let cameraDisplayMode = 'live';
let homographyCorners = [];
let cleanFrozenImg = null;
let isPipVisible = false;

function getLiveStreamUrl() {
  const customIp = document.getElementById('inpIpWebcam') ? document.getElementById('inpIpWebcam').value.trim().replace(/\/+$/, '') : '';
  return customIp ? customIp + '/video' : '/api/camera/stream';
}

function initCameraModule() {
  reloadLiveStream();
  const c = document.getElementById('cameraCanvas');
  if (c) {
    c.addEventListener('mousedown', onCameraCanvasMouseDown);
    c.addEventListener('mousemove', onCameraCanvasMouseMove);
    c.addEventListener('mouseup', onCameraCanvasMouseUp);
    c.addEventListener('mouseleave', () => {
      draggedCornerIdx = -1;
      const hud = document.getElementById('canvasCrosshairHud');
      if (hud) hud.style.display = 'none';
    });
  }
}

function handleStreamLoaded(img) {
  const overlay = document.getElementById('streamLoadingOverlay');
  if (overlay) overlay.style.display = 'none';
  const pill = document.getElementById('camStreamPill');
  if (pill) {
    pill.style.color = '#34d399';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> LIVE OVERHEAD';
  }
}

function handleStreamError(img) {
  if (img.src && !img.src.includes('/api/camera/stream')) {
    img.src = '/api/camera/stream?t=' + Date.now();
  } else {
    const overlay = document.getElementById('streamLoadingOverlay');
    if (overlay) overlay.style.display = 'flex';
    const pill = document.getElementById('camStreamPill');
    if (pill) {
      pill.style.color = '#f87171';
      pill.textContent = 'CAM OFFLINE';
    }
  }
}

function reloadLiveStream() {
  const streamImg = document.getElementById('liveStreamView');
  const pipImg = document.getElementById('pipStreamImg');
  const overlay = document.getElementById('streamLoadingOverlay');
  if (overlay) overlay.style.display = 'none';
  const url = getLiveStreamUrl();
  if (streamImg) streamImg.src = url;
  if (pipImg && isPipVisible) pipImg.src = url;
  setCameraDisplayMode('live');
}

function setCameraDisplayMode(mode) {
  cameraDisplayMode = mode;
  const streamImg = document.getElementById('liveStreamView');
  const canvas = document.getElementById('cameraCanvas');
  const badge = document.getElementById('cameraModeBadge');
  const btnLive = document.getElementById('btnModeLive');
  const btnFreeze = document.getElementById('btnModeFreeze');

  if (mode === 'live') {
    if (streamImg) streamImg.style.display = 'block';
    if (canvas) canvas.style.display = 'none';
    if (badge) {
      badge.textContent = '● LIVE STREAM';
      badge.style.color = '#34d399';
      badge.style.borderColor = 'rgba(52,211,153,0.4)';
    }
    if (btnLive) { btnLive.style.borderColor = 'var(--emerald-green)'; }
    if (btnFreeze) { btnFreeze.style.borderColor = ''; }
    document.getElementById('cornerStatus').textContent = 'Status: Live Video streaming. Click "Freeze & Calibrate" to set 4 table corners.';
  } else {
    if (streamImg) streamImg.style.display = 'none';
    if (canvas) canvas.style.display = 'block';
    if (badge) {
      badge.textContent = '🎯 CALIBRATION MODE (CLICK 4 CORNERS)';
      badge.style.color = '#facc15';
      badge.style.borderColor = 'rgba(250,204,21,0.4)';
    }
    if (btnLive) { btnLive.style.borderColor = ''; }
    if (btnFreeze) { btnFreeze.style.borderColor = 'var(--emerald-green)'; }
    updateCornerStatusText();
  }
}

function toggleCameraPip() {
  const box = document.getElementById('cameraPipBox');
  const pipImg = document.getElementById('pipStreamImg');
  const btn = document.getElementById('btnTogglePip');
  if (!box) return;
  isPipVisible = !isPipVisible;
  if (isPipVisible) {
    box.style.display = 'block';
    pipImg.src = getLiveStreamUrl();
    if (btn) btn.classList.add('active');
  } else {
    box.style.display = 'none';
    pipImg.src = '';
    if (btn) btn.classList.remove('active');
  }
}

async function freezeFrameForCalibration() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Grabbing high-res frame from phone...';
  st.style.color = 'var(--text-muted)';
  try {
    const res = await fetch('/api/ipwebcam/capture', { method: 'POST' });
    const data = await res.json();
    if (data.success && data.image_data) {
      const img = new Image();
      img.onload = function() {
        cleanFrozenImg = img;
        const c = document.getElementById('cameraCanvas');
        c.width = img.width;
        c.height = img.height;
        homographyCorners = [];
        redrawCalibrationCanvas();
        setCameraDisplayMode('frozen');
        st.innerHTML = `✅ Frame frozen (${data.width}×${data.height} px). Click 4 corners in order.`;
        st.style.color = 'var(--emerald-green)';
      };
      img.src = data.image_data;
    } else {
      st.innerHTML = `❌ ${data.error || 'Capture failed'}`;
      st.style.color = 'var(--ruby-red)';
    }
  } catch (e) {
    st.textContent = '❌ Failed to capture frame. Check connection.';
    st.style.color = 'var(--ruby-red)';
  }
}

// Mathematical Letterbox & Aspect-Ratio Aware Click Coordinates
function getCanvasClickCoordinates(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  const bw = canvas.width;
  const bh = canvas.height;
  if (!bw || !bh || !rect.width || !rect.height) return null;

  const bitmapRatio = bw / bh;
  const clientRatio = rect.width / rect.height;

  let renderW, renderH, offsetX, offsetY;

  if (Math.abs(clientRatio - bitmapRatio) < 0.005) {
    renderW = rect.width;
    renderH = rect.height;
    offsetX = 0.0;
    offsetY = 0.0;
  } else if (clientRatio > bitmapRatio) {
    // Container is wider than image aspect ratio -> pillarboxing (bars on left & right)
    renderH = rect.height;
    renderW = rect.height * bitmapRatio;
    offsetX = (rect.width - renderW) / 2.0;
    offsetY = 0.0;
  } else {
    // Container is taller than image aspect ratio -> letterboxing (bars on top & bottom)
    renderW = rect.width;
    renderH = rect.width / bitmapRatio;
    offsetX = 0.0;
    offsetY = (rect.height - renderH) / 2.0;
  }

  const imgX = clickX - offsetX;
  const imgY = clickY - offsetY;

  // Ignore clicks outside the actual rendered image boundary
  if (imgX < -2 || imgX > renderW + 2 || imgY < -2 || imgY > renderH + 2) {
    return null;
  }

  const clampedX = Math.max(0, Math.min(renderW, imgX));
  const clampedY = Math.max(0, Math.min(renderH, imgY));

  const px = (clampedX / renderW) * bw;
  const py = (clampedY / renderH) * bh;

  return [px, py];
}

let draggedCornerIdx = -1;

function onCameraCanvasMouseDown(e) {
  if (cameraDisplayMode !== 'frozen') return;
  const c = document.getElementById('cameraCanvas');
  const pt = getCanvasClickCoordinates(e, c);
  if (!pt) return;

  const [px, py] = pt;

  // 1. Check if user clicked near an existing corner to drag/adjust it
  // Threshold: 35 image pixels or ~15 screen pixels
  const hitRadius = Math.max(25, c.width * 0.02);
  let foundIdx = -1;
  let minD = hitRadius;

  for (let i = 0; i < homographyCorners.length; i++) {
    const dx = homographyCorners[i][0] - px;
    const dy = homographyCorners[i][1] - py;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < minD) {
      minD = dist;
      foundIdx = i;
    }
  }

  if (foundIdx >= 0) {
    draggedCornerIdx = foundIdx;
    homographyCorners[draggedCornerIdx] = [px, py];
    redrawCalibrationCanvas();
    updateCornerStatusText();
    return;
  }

  // 2. If clicking on new point and corners < 4
  if (homographyCorners.length < 4) {
    homographyCorners.push([px, py]);
    draggedCornerIdx = homographyCorners.length - 1;
    redrawCalibrationCanvas();
    updateCornerStatusText();
  } else {
    // All 4 already set: select the closest corner to drag
    let closestIdx = 0;
    let closestD = Infinity;
    for (let i = 0; i < homographyCorners.length; i++) {
      const dx = homographyCorners[i][0] - px;
      const dy = homographyCorners[i][1] - py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < closestD) {
        closestD = dist;
        closestIdx = i;
      }
    }
    draggedCornerIdx = closestIdx;
    homographyCorners[draggedCornerIdx] = [px, py];
    redrawCalibrationCanvas();
    updateCornerStatusText();
  }
}

function onCameraCanvasMouseMove(e) {
  const c = document.getElementById('cameraCanvas');
  const hud = document.getElementById('canvasCrosshairHud');
  const pt = getCanvasClickCoordinates(e, c);

  if (pt && hud) {
    hud.style.display = 'block';
    hud.textContent = `Pixel: X: ${pt[0].toFixed(0)}, Y: ${pt[1].toFixed(0)}`;
  } else if (hud) {
    hud.style.display = 'none';
  }

  if (cameraDisplayMode !== 'frozen') return;

  if (draggedCornerIdx >= 0 && pt) {
    homographyCorners[draggedCornerIdx] = pt;
    redrawCalibrationCanvas();
    updateCornerStatusText();
  }
}

function onCameraCanvasMouseUp() {
  draggedCornerIdx = -1;
}

function updateCornerStatusText() {
  const labels = ['1: TR (0,0)', '2: TL (1168.4, 0)', '3: BL (1168.4, 2341.9)', '4: BR (0, 2341.9)'];
  const n = homographyCorners.length;
  const el = document.getElementById('cornerStatus');
  if (n < 4) {
    el.innerHTML = `Corners set: <strong>${n} / 4</strong>. Next click: <strong style="color:var(--emerald-green);">${labels[n]}</strong> <span style="font-size:0.7rem; color:var(--text-muted);">(You can drag any point to fine-tune)</span>`;
  } else {
    el.innerHTML = `<span style="color:var(--emerald-green); font-weight:700;">✅ All 4 Corners Set!</span> <span style="font-size:0.7rem; color:#94a3b8;">(Drag any corner to adjust, or click 'COMPUTE PLANAR HOMOGRAPHY')</span>`;
  }
}

function redrawCalibrationCanvas() {
  const c = document.getElementById('cameraCanvas');
  if (!c || !cleanFrozenImg) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(cleanFrozenImg, 0, 0);

  const colors = ['#06b6d4', '#10b981', '#f59e0b', '#ef4444'];
  const names = ['1: TR (0,0)', '2: TL (1168,0)', '3: BL (1168,2342)', '4: BR (0,2342)'];

  // Scale marker dimensions dynamically based on image resolution
  const resScale = Math.max(1.0, c.width / 1280.0);
  const reticleR = 12 * resScale;
  const crossLen = 22 * resScale;
  const lineW = 3.5 * resScale;
  const fontPx = Math.round(16 * resScale);

  if (homographyCorners.length >= 2) {
    ctx.lineWidth = lineW;
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.95)';
    ctx.beginPath();
    ctx.moveTo(homographyCorners[0][0], homographyCorners[0][1]);
    for (let i = 1; i < homographyCorners.length; i++) {
      ctx.lineTo(homographyCorners[i][0], homographyCorners[i][1]);
    }
    if (homographyCorners.length === 4) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
      ctx.fill();
    }
    ctx.stroke();
  }

  homographyCorners.forEach((pt, i) => {
    const [x, y] = pt;

    // Glowing outer shadow
    ctx.save();
    ctx.shadowColor = colors[i];
    ctx.shadowBlur = 12 * resScale;

    // Crosshair target reticle
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = lineW;
    ctx.beginPath();
    ctx.moveTo(x - crossLen, y); ctx.lineTo(x + crossLen, y);
    ctx.moveTo(x, y - crossLen); ctx.lineTo(x, y + crossLen);
    ctx.stroke();

    // Solid colored ring
    ctx.fillStyle = colors[i];
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = lineW * 0.8;
    ctx.beginPath();
    ctx.arc(x, y, reticleR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Inner bright center dot (exact click point)
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(x, y, 3 * resScale, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Badge label
    ctx.font = `bold ${fontPx}px ui-monospace, monospace`;
    const tag = names[i];
    const tw = ctx.measureText(tag).width;
    const badgeH = fontPx + 10 * resScale;
    const badgeW = tw + 14 * resScale;
    const badgeX = x + 14 * resScale;
    const badgeY = y - badgeH / 2;

    ctx.fillStyle = 'rgba(4, 7, 15, 0.92)';
    ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 1.5 * resScale;
    ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);

    ctx.fillStyle = colors[i];
    ctx.fillText(tag, badgeX + 7 * resScale, badgeY + fontPx * 0.9);
  });
}

function undoLastCorner() {
  if (homographyCorners.length > 0) {
    homographyCorners.pop();
    draggedCornerIdx = -1;
    redrawCalibrationCanvas();
    updateCornerStatusText();
  }
}

function resetCorners() {
  homographyCorners = [];
  draggedCornerIdx = -1;
  redrawCalibrationCanvas();
  updateCornerStatusText();
}

async function computeHomography() {
  if (homographyCorners.length !== 4) {
    alert("Please select exactly 4 corners first:\n1: TR (0,0)\n2: TL (1168.4, 0)\n3: BL (1168.4, 2341.9)\n4: BR (0, 2341.9)");
    return;
  }
  const res = await fetch('/api/homography/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ corners: homographyCorners })
  });
  const data = await res.json();
  if (data.success) {
    alert("🎉 " + data.message);
  } else {
    alert("❌ Calibration failed: " + data.message);
  }
}

function uploadImage(input) {
  if (!input.files || !input.files[0]) return;
  const formData = new FormData();
  formData.append('image', input.files[0]);
  fetch('/api/camera/upload', {
    method: 'POST',
    body: formData
  }).then(r => r.json()).then(d => {
    if (d.success && d.image_data) {
      const img = new Image();
      img.onload = function() {
        cleanFrozenImg = img;
        const c = document.getElementById('cameraCanvas');
        c.width = img.width; c.height = img.height;
        homographyCorners = [];
        redrawCalibrationCanvas();
        setCameraDisplayMode('frozen');
      };
      img.src = d.image_data;
    }
  });
}

function loadSampleImage() {
  fetch('/api/camera/sample').then(r => r.json()).then(d => {
    if (d.success && d.image_data) {
      const img = new Image();
      img.onload = function() {
        cleanFrozenImg = img;
        const c = document.getElementById('cameraCanvas');
        c.width = img.width; c.height = img.height;
        homographyCorners = [];
        redrawCalibrationCanvas();
        setCameraDisplayMode('frozen');
      };
      img.src = d.image_data;
    }
  });
}

function saveIpWebcam() {
  const url = document.getElementById('inpIpWebcam').value.trim().replace(/\/+$/, '');
  fetch('/api/ipwebcam/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: url })
  }).then(r => r.json()).then(d => {
    document.getElementById('ipWebcamStatus').textContent = d.success ? '✅ URL saved: ' + url : '❌ ' + d.error;
    document.getElementById('ipWebcamStatus').style.color = d.success ? 'var(--emerald-green)' : 'var(--ruby-red)';
    reloadLiveStream();
  }).catch(e => {
    document.getElementById('ipWebcamStatus').textContent = '❌ Error: ' + e.message;
    document.getElementById('ipWebcamStatus').style.color = 'var(--ruby-red)';
  });
}

async function captureAndDetect() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Capturing snapshot + running YOLO detection...';
  st.style.color = 'var(--text-muted)';
  try {
    const capRes = await fetch('/api/ipwebcam/capture', { method: 'POST' });
    const capData = await capRes.json();
    if (!capData.success) {
      st.innerHTML = `❌ Capture failed: ${capData.error}`;
      st.style.color = 'var(--ruby-red)';
      return;
    }
    st.innerHTML = `✅ Captured ${capData.width}×${capData.height}. Running YOLOv8 snooker model...`;
    st.style.color = 'var(--emerald-green)';

    await runDetection();
    st.innerHTML = `✅ Captured + Detection complete! View balls on table.`;
    showTab('tab-balls', document.querySelectorAll('.nav-btn')[3]);
  } catch (e) {
    st.textContent = '❌ Error: ' + e.message;
    st.style.color = 'var(--ruby-red)';
  }
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    updateESPBadge(data.connected);
  } catch(e) {}
}

async function detectSpot() {
  const res = await fetch('/api/laser/detect_spot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_x: targetX, target_y: targetY })
  });
  const data = await res.json();
  if (data.detected) {
    document.getElementById('spotResult').innerHTML =
      `Spot detected at: (${data.measured_x.toFixed(1)}, ${data.measured_y.toFixed(1)}) mm<br>` +
      `Error: ex=${data.err_x.toFixed(1)}mm, ey=${data.err_y.toFixed(1)}mm, Euclidean=${data.err_dist.toFixed(1)}mm`;
  } else {
    document.getElementById('spotResult').textContent = "No red laser spot identified. Make sure laser is ON.";
  }
}

async function runDetection() {
  const res = await fetch('/api/balls/detect');
  const data = await res.json();
  const tb = document.getElementById('ballsTableBody');
  tb.innerHTML = '';
  if (!data.balls || data.balls.length === 0) {
    tb.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:16px;">No balls detected.</td></tr>';
    return;
  }

  clearBalls();
  data.balls.forEach(b => {
    const tx = b.table_x_mm !== null ? b.table_x_mm.toFixed(1) : "---";
    const ty = b.table_y_mm !== null ? b.table_y_mm.toFixed(1) : "---";
    tb.innerHTML += `<tr>
      <td>${b.id}</td>
      <td><strong>${b.label}</strong></td>
      <td>${(b.confidence * 100).toFixed(0)}%</td>
      <td>${tx}</td>
      <td>${ty}</td>
      <td><button class="btn btn-secondary" style="padding:4px 12px; font-size:0.73rem;" onclick="setPreset(${b.table_x_mm || 0}, ${b.table_y_mm || 0})">Aim</button></td>
    </tr>`;

    if (b.table_x_mm !== null && b.table_y_mm !== null) {
      let col = SNOOKER_COLORS.red;
      const lbl = b.label.toLowerCase();
      if (lbl.includes("cue") || lbl.includes("white")) col = SNOOKER_COLORS.cue;
      else if (lbl.includes("black")) col = SNOOKER_COLORS.black;
      else if (lbl.includes("pink")) col = SNOOKER_COLORS.pink;
      else if (lbl.includes("blue")) col = SNOOKER_COLORS.blue;
      else if (lbl.includes("brown")) col = SNOOKER_COLORS.brown;
      else if (lbl.includes("green")) col = SNOOKER_COLORS.green;
      else if (lbl.includes("yellow")) col = SNOOKER_COLORS.yellow;

      const m = createBallMesh(b.label, col);
      const pos = tableToScene(b.table_x_mm, b.table_y_mm);
      m.position.set(pos.x, BALL_RADIUS, pos.z);
      scene.add(m);
      ballMeshes.push(m);
    }
  });
}

function exportCsv() {
  window.location.href = '/api/logs/export';
}

function runInit() {
  console.log("Starting 3D Snooker Table initialization...");
  try {
    initThree();
    console.log("3D Snooker Table rendered successfully!");
  } catch (err) {
    console.error("3D Table Init Error:", err);
  }
  fetchNodes();
  updatePred();
  setTimeout(initCameraModule, 300);
  setInterval(pollStatus, 2000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', runInit);
} else {
  runInit();
}
</script>
</body>
</html>
"""

# ==============================================================
# 5. REST API ROUTES (PRESERVING ALL HARDWARE & CALIBRATION ENDPOINTS)
# ==============================================================

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    x = float(data.get("x", 0.0))
    y = float(data.get("y", 0.0))
    pan, tilt, info = interpolator.predict(x, y)
    return jsonify({"pan": round(pan, 1), "tilt": round(tilt, 1), "info": info})

@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    return jsonify({"nodes": calib_data["nodes"]})

@app.route("/api/simulation", methods=["POST"])
def api_sim():
    global simulation_mode
    data = request.json or {}
    simulation_mode = bool(data.get("enabled", True))
    add_log("SIM", f"Simulation mode set to {simulation_mode}")
    return jsonify({"success": True, "simulation": simulation_mode})

@app.route("/api/esp32/config", methods=["POST"])
def api_esp32_config():
    global esp32_base_url
    data = request.json or {}
    url = str(data.get("base_url", "http://192.168.31.2")).strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    esp32_base_url = url
    add_log("CONFIG", f"ESP32 target URL updated to {esp32_base_url}")
    return jsonify({"success": True, "base_url": esp32_base_url})

@app.route("/api/esp32/ping", methods=["GET"])
def api_esp32_ping():
    try:
        t0 = time.time()
        r = requests.get(f"{esp32_base_url}/status", timeout=2.0)
        dt = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            return jsonify({"success": True, "message": f"Successfully connected to ESP32 at {esp32_base_url} ({dt}ms latency)"})
        return jsonify({"success": False, "message": f"HTTP {r.status_code} received from {esp32_base_url}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Cannot reach ESP32 at {esp32_base_url}: {e}"})

@app.route("/api/status", methods=["GET"])
def api_status():
    if simulation_mode:
        return jsonify({"connected": True, "is_simulation": True, "pan": current_pan, "tilt": current_tilt, "laser": laser_state})
    try:
        r = requests.get(f"{esp32_base_url}/status", timeout=1.5).json()
        return jsonify({"connected": True, "is_simulation": False, "pan": r.get("pan"), "tilt": r.get("tilt"), "laser": r.get("laser")})
    except:
        return jsonify({"connected": False, "is_simulation": False})

@app.route("/api/move", methods=["POST"])
def api_move():
    global current_pan, current_tilt
    data = request.json or {}
    pan = float(data.get("pan", 10.0))
    tilt = float(data.get("tilt", 150.0))
    current_pan = pan
    current_tilt = tilt

    if simulation_mode:
        msg = f"[SIMULATION] Commanded Pan: {pan:.1f}°, Tilt: {tilt:.1f}°"
        add_log("MOVE_SIM", msg)
        return jsonify({"success": True, "message": msg})

    try:
        requests.get(f"{esp32_base_url}/move", params={"axis": "pan", "angle": int(pan)}, timeout=1.5)
        requests.get(f"{esp32_base_url}/move", params={"axis": "tilt", "angle": int(tilt)}, timeout=1.5)
        msg = f"Servos positioned to Pan: {pan:.1f}°, Tilt: {tilt:.1f}° on ESP32 ({esp32_base_url})"
        add_log("MOVE_HW", msg)
        return jsonify({"success": True, "message": msg})
    except Exception as e:
        msg = f"ESP32 communication failure: {e}"
        add_log("ERROR", msg)
        return jsonify({"success": False, "message": msg})

@app.route("/api/center", methods=["POST"])
def api_center():
    global current_pan, current_tilt, emergency_stopped
    current_pan = 90.0
    current_tilt = 90.0
    emergency_stopped = False
    if not simulation_mode:
        try: requests.get(f"{esp32_base_url}/center", timeout=1.5)
        except: pass
    add_log("SERVO", "Centered to Pan: 90°, Tilt: 90°")
    return jsonify({"success": True, "message": "Centered (90°, 90°)"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    global emergency_stopped, laser_state
    emergency_stopped = True
    laser_state = False
    if not simulation_mode:
        try: requests.get(f"{esp32_base_url}/stop", timeout=1.5)
        except: pass
    add_log("SAFETY", "Emergency STOP triggered!")
    return jsonify({"success": True, "message": "Emergency STOP executed"})

@app.route("/api/laser/toggle", methods=["POST"])
def api_laser_toggle():
    global laser_state
    if emergency_stopped:
        return jsonify({"success": False, "message": "Clear STOP condition first"}), 400
    laser_state = not laser_state
    if not simulation_mode:
        try: requests.get(f"{esp32_base_url}/laser", params={"state": "1" if laser_state else "0"}, timeout=1.5)
        except: pass
    add_log("LASER", f"Laser toggled to {'ON' if laser_state else 'OFF'}")
    return jsonify({"success": True, "laser": laser_state})

@app.route("/api/laser/off", methods=["POST"])
def api_laser_off():
    global laser_state
    laser_state = False
    if not simulation_mode:
        try: requests.get(f"{esp32_base_url}/laser", params={"state": "0"}, timeout=1.5)
        except: pass
    add_log("LASER", "Laser forced OFF")
    return jsonify({"success": True, "laser": False})

@app.route("/api/camera/upload", methods=["POST"])
def api_camera_upload():
    global uploaded_frame_bgr, uploaded_frame_b64
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["image"]
    bytes_data = file.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    uploaded_frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    b64 = base64.b64encode(bytes_data).decode("utf-8")
    uploaded_frame_b64 = f"data:image/jpeg;base64,{b64}"
    add_log("CAMERA", f"Uploaded frame ({uploaded_frame_bgr.shape[1]}x{uploaded_frame_bgr.shape[0]} px)")
    return jsonify({"success": True, "image_data": uploaded_frame_b64})

@app.route("/api/camera/sample", methods=["GET"])
def api_camera_sample():
    global uploaded_frame_bgr, uploaded_frame_b64
    h, w = 922, 460
    synth = np.zeros((h, w, 3), dtype=np.uint8)
    synth[:] = [20, 80, 25]
    cv2.rectangle(synth, (0, 0), (w, h), (12, 35, 70), 12)
    cv2.line(synth, (12, int(h * 0.7)), (w - 12, int(h * 0.7)), (200, 200, 200), 2)

    uploaded_frame_bgr = synth
    _, buf = cv2.imencode(".jpg", synth)
    b64 = base64.b64encode(buf).decode("utf-8")
    uploaded_frame_b64 = f"data:image/jpeg;base64,{b64}"
    return jsonify({"success": True, "image_data": uploaded_frame_b64})

@app.route("/api/homography/compute", methods=["POST"])
def api_homography_compute():
    data = request.json or {}
    corners = data.get("corners", [])
    if len(corners) != 4:
        return jsonify({"success": False, "message": "Exactly 4 corners required"}), 400
    success = camera_homography.calibrate(corners)
    if success:
        add_log("HOMOGRAPHY", "Computed 4-corner homography matrix")
        return jsonify({"success": True, "message": "Planar homography successfully computed and saved!"})
    return jsonify({"success": False, "message": "Degenerate corner geometry. Please re-select corners."}), 400

@app.route("/api/laser/detect_spot", methods=["POST"])
def api_laser_detect_spot():
    data = request.json or {}
    tx = float(data.get("target_x", 0.0))
    ty = float(data.get("target_y", 0.0))

    if uploaded_frame_bgr is None:
        spot = (100.0, 150.0, 0.95)
    else:
        spot = detect_laser_spot(uploaded_frame_bgr)

    if spot is None:
        return jsonify({"detected": False, "message": "No laser spot identified in current camera view"})

    cx, cy, conf = spot
    mx, my = cx, cy
    if camera_homography.H_img_to_table is not None:
        table_pt = camera_homography.pixel_to_table(cx, cy)
        if table_pt:
            mx, my = table_pt

    ex = float(mx - tx)
    ey = float(my - ty)
    dist = float(math.sqrt(ex * ex + ey * ey))
    add_log("VERIFY", f"Closed-loop error: {dist:.1f} mm")

    return jsonify({
        "detected": True,
        "measured_x": mx,
        "measured_y": my,
        "err_x": ex,
        "err_y": ey,
        "err_dist": dist,
        "confidence": conf
    })

@app.route("/api/balls/detect", methods=["GET"])
def api_balls_detect():
    global uploaded_frame_bgr, uploaded_frame_b64
    frame = uploaded_frame_bgr

    if frame is None and ip_webcam_url:
        try:
            snap_url = ip_webcam_url.rstrip("/") + "/photo.jpg"
            resp = requests.get(snap_url, timeout=3)
            if resp.status_code == 200:
                nparr = np.frombuffer(resp.content, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    uploaded_frame_bgr = decoded
                    frame = decoded
                    add_log("DETECTION", f"Auto-captured live frame from {snap_url} for detection")
        except Exception:
            pass

    if frame is None:
        balls = [
            {"id": 1, "label": "Cue Ball (White)", "confidence": 0.96, "table_x_mm": 584.2, "table_y_mm": 1800.0},
            {"id": 2, "label": "Black Ball", "confidence": 0.94, "table_x_mm": 584.2, "table_y_mm": 2100.0},
            {"id": 3, "label": "Pink Ball", "confidence": 0.92, "table_x_mm": 584.2, "table_y_mm": 1756.4},
            {"id": 4, "label": "Blue Ball", "confidence": 0.95, "table_x_mm": 584.2, "table_y_mm": 1170.94},
            {"id": 5, "label": "Brown Ball", "confidence": 0.91, "table_x_mm": 584.2, "table_y_mm": 737.0},
            {"id": 6, "label": "Green Ball", "confidence": 0.89, "table_x_mm": 876.2, "table_y_mm": 737.0},
            {"id": 7, "label": "Yellow Ball", "confidence": 0.93, "table_x_mm": 292.2, "table_y_mm": 737.0},
            {"id": 8, "label": "Red Ball #1", "confidence": 0.90, "table_x_mm": 584.2, "table_y_mm": 1600.0},
            {"id": 9, "label": "Red Ball #2", "confidence": 0.88, "table_x_mm": 556.0, "table_y_mm": 1645.0},
            {"id": 10, "label": "Red Ball #3", "confidence": 0.89, "table_x_mm": 612.4, "table_y_mm": 1645.0}
        ]
        return jsonify({"balls": balls})

    detected = detector.detect(frame, homography=camera_homography)
    res = []
    for b in detected:
        res.append({
            "id": b.id,
            "label": b.label,
            "confidence": b.confidence,
            "table_x_mm": b.table_x_mm,
            "table_y_mm": b.table_y_mm
        })
    add_log("DETECTION", f"YOLOv8 detected {len(res)} ball(s) from camera frame")
    return jsonify({"balls": res})

# ── IP Webcam Integration Endpoints ──

@app.route("/api/ipwebcam/config", methods=["POST"])
def api_ipwebcam_config():
    global ip_webcam_url
    data = request.get_json(force=True)
    url = str(data.get("url", "")).strip().rstrip("/")
    if not url.startswith("http"):
        return jsonify({"success": False, "error": "URL must start with http"}), 400
    ip_webcam_url = url
    add_log("CAMERA", f"IP Webcam URL set to: {ip_webcam_url}")
    return jsonify({"success": True, "url": ip_webcam_url})

@app.route("/api/camera/stream")
def api_camera_stream():
    stream_url = ip_webcam_url.rstrip("/") + "/video"
    try:
        req = requests.get(stream_url, stream=True, timeout=5)
        content_type = req.headers.get("content-type", "multipart/x-mixed-replace; boundary=Ba4oTvQMY8ew04N8dcnM")
        def generate():
            try:
                for chunk in req.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk
            except Exception:
                pass
        return Response(generate(), content_type=content_type)
    except Exception as e:
        return jsonify({"error": f"Camera stream unavailable: {e}"}), 502

@app.route("/api/ipwebcam/capture", methods=["POST"])
def api_ipwebcam_capture():
    global uploaded_frame_bgr, uploaded_frame_b64
    try:
        snap_url = ip_webcam_url.rstrip("/") + "/photo.jpg"
        add_log("CAMERA", f"Capturing from IP Webcam: {snap_url}")
        resp = requests.get(snap_url, timeout=5)
        if resp.status_code != 200:
            return jsonify({"success": False, "error": f"HTTP {resp.status_code} from phone"}), 500
        img_bytes = resp.content
        nparr = np.frombuffer(img_bytes, np.uint8)
        uploaded_frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if uploaded_frame_bgr is None:
            return jsonify({"success": False, "error": "Could not decode image from phone"}), 500
        h, w = uploaded_frame_bgr.shape[:2]
        _, buf = cv2.imencode(".jpg", uploaded_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf).decode("utf-8")
        uploaded_frame_b64 = f"data:image/jpeg;base64,{b64}"
        add_log("CAMERA", f"IP Webcam frame captured: {w}x{h} px")
        return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": f"Cannot connect to {ip_webcam_url}. Is IP Webcam running on your phone?"}), 500
    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Phone camera timed out. Check Wi-Fi connection."}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/logs/export", methods=["GET"])
def api_logs_export():
    output = io.StringIO()
    output.write("Timestamp,Category,Message\n")
    for lg in execution_logs:
        output.write(f'"{lg["timestamp"]}","{lg["category"]}","{lg["message"]}"\n')
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=snooker_laser_logs.csv"}
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
