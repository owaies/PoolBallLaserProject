"""
Professional 3D Snooker Ball Identification & Laser Positioning System
Modern Nature Background, Capsule Buttons, Clean Minimal Typography (Inter),
Smooth Fluid Hover Animations, Glassmorphic Panels, and Interactive Three.js 3D Simulation.
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
ip_webcam_url = "http://192.168.31.150:8080"  # Default IP Webcam URL (update in UI)

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
<title>PoolBot 3D — Snooker Ball Tracking & Laser Aiming</title>
<!-- Clean, Simple, Modern Sans Typography -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<!-- Three.js and OrbitControls (Served locally from static/js) -->
<script src="/static/js/three.min.js"></script>
<script src="/static/js/OrbitControls.js"></script>
<style>
:root {
  --primary-forest: #1e5238;
  --primary-green: #2d6a4f;
  --primary-light: #52b788;
  --accent-mint: #74c69d;
  --terracotta: #c85a32;
  --amber-gold: #d97706;
  
  --glass-bg: rgba(255, 255, 255, 0.76);
  --glass-bg-hover: rgba(255, 255, 255, 0.92);
  --glass-border: rgba(255, 255, 255, 0.85);
  --glass-shadow: 0 12px 35px rgba(20, 45, 30, 0.12);
  
  --text-dark: #1a2a22;
  --text-body: #334e3f;
  --text-muted: #5e7a6b;
  --text-light: #ffffff;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
  color: var(--text-dark);
  
  /* High-Res Scenic Nature Background with Soft Tint Overlay */
  background: linear-gradient(135deg, rgba(240, 248, 243, 0.88), rgba(225, 238, 230, 0.85)),
              url('/static/images/nature_bg.jpg') center center / cover fixed no-repeat;
}

/* Custom Smooth Green Scrollbars */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.4); }
::-webkit-scrollbar-thumb { background: rgba(45, 106, 79, 0.4); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-green); }

/* Header Component */
header {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--glass-border);
  padding: 12px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.brand-icon {
  width: 42px;
  height: 42px;
  border-radius: 999px;
  background: linear-gradient(135deg, #2d6a4f, #52b788);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 14px rgba(45, 106, 79, 0.35);
  transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.brand-icon:hover { transform: scale(1.08) rotate(5deg); }
.brand-icon svg { width: 22px; height: 22px; fill: #fff; }

.brand-text h1 {
  font-size: 1.15rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--primary-forest);
}
.brand-text p {
  font-size: 0.73rem;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
}

/* Nav Pills (Fully Capsule-Shaped) */
.nav-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.nav-btn {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(45, 106, 79, 0.18);
  color: var(--text-body);
  padding: 8px 18px;
  border-radius: 999px; /* Capsule */
  font-weight: 600;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.nav-btn:hover {
  background: #ffffff;
  color: var(--primary-forest);
  border-color: var(--primary-green);
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(45, 106, 79, 0.18);
}
.nav-btn.active {
  background: linear-gradient(135deg, #1e5238, #2d6a4f);
  color: #ffffff;
  border-color: #1e5238;
  box-shadow: 0 4px 16px rgba(30, 82, 56, 0.35);
}

/* Header Controls */
.header-actions { display: flex; align-items: center; gap: 12px; }
.sim-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(45, 106, 79, 0.2);
  padding: 6px 14px;
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.sim-pill:hover { transform: translateY(-1px); }
.sim-pill input[type=checkbox] {
  accent-color: var(--primary-green);
  width: 15px;
  height: 15px;
  cursor: pointer;
}
.sim-pill span {
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--primary-forest);
  letter-spacing: 0.03em;
}

.status-pill {
  font-size: 0.74rem;
  padding: 7px 15px;
  border-radius: 999px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.badge-sim { background: rgba(217, 119, 6, 0.12); color: #b45309; border: 1px solid rgba(217, 119, 6, 0.35); }
.badge-live { background: rgba(45, 106, 79, 0.15); color: #1e5238; border: 1px solid rgba(45, 106, 79, 0.4); box-shadow: 0 0 12px rgba(45, 106, 79, 0.25); }
.badge-off { background: rgba(200, 90, 50, 0.12); color: var(--terracotta); border: 1px solid rgba(200, 90, 50, 0.35); }

/* Main Grid Layout */
main {
  display: grid;
  grid-template-columns: 1.4fr 0.85fr;
  gap: 22px;
  padding: 22px 28px;
  flex: 1;
  position: relative;
  z-index: 10;
}
@media(max-width: 1200px) { main { grid-template-columns: 1fr; } }

/* Frosted Glass Nature Panels */
.glass-panel {
  background: var(--glass-bg);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--glass-border);
  border-radius: 20px;
  padding: 22px;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: var(--glass-shadow);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-panel:hover {
  background: var(--glass-bg-hover);
  box-shadow: 0 16px 45px rgba(20, 45, 30, 0.18);
  border-color: rgba(255, 255, 255, 1);
}

.panel-header {
  font-size: 0.96rem;
  font-weight: 700;
  color: var(--primary-forest);
  margin-bottom: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(45, 106, 79, 0.15);
  padding-bottom: 10px;
}
.panel-header .tag-pill {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: rgba(45, 106, 79, 0.1);
  color: var(--primary-forest);
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 700;
}

.mono { font-family: 'JetBrains Mono', monospace; }

/* 3D Viewport in Nature Theme */
.viewport-3d-box {
  width: 100%;
  height: 610px;
  background: radial-gradient(circle at 50% 50%, #edf5f0 0%, #d8e8dd 100%);
  border-radius: 16px;
  border: 1px solid rgba(45, 106, 79, 0.2);
  overflow: hidden;
  position: relative;
  box-shadow: inset 0 2px 25px rgba(30, 82, 56, 0.12);
}
#three-canvas { width: 100%; height: 100%; display: block; }

.viewport-toolbar {
  position: absolute;
  top: 14px;
  left: 14px;
  display: flex;
  gap: 8px;
  z-index: 10;
}
.viewport-btn {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(45, 106, 79, 0.25);
  color: var(--primary-forest);
  padding: 8px 16px;
  border-radius: 999px; /* Capsule */
  font-size: 0.77rem;
  font-weight: 600;
  cursor: pointer;
  backdrop-filter: blur(12px);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.viewport-btn:hover {
  background: var(--primary-green);
  color: #ffffff;
  border-color: var(--primary-green);
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(45, 106, 79, 0.25);
}
.viewport-btn.active {
  background: var(--primary-forest);
  color: #ffffff;
  border-color: var(--primary-forest);
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
}
.hud-pill {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(45, 106, 79, 0.25);
  backdrop-filter: blur(12px);
  padding: 7px 16px;
  border-radius: 999px; /* Capsule */
  font-size: 0.75rem;
  color: var(--primary-forest);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
  font-weight: 600;
}

/* Organic Telemetry Ribbon */
.telemetry-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 16px;
}
.telemetry-card {
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(45, 106, 79, 0.2);
  border-radius: 14px;
  padding: 12px 18px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
  transition: all 0.2s ease;
}
.telemetry-card:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06); }
.telemetry-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0;
  width: 5px; height: 100%;
}
.telemetry-card.target::before { background: var(--primary-green); }
.telemetry-card.angles::before { background: var(--terracotta); }

.telemetry-card small {
  display: block;
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  margin-bottom: 3px;
}
.telemetry-card .value {
  font-size: 0.96rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}
.telemetry-card.target .value { color: var(--primary-forest); }
.telemetry-card.angles .value { color: var(--terracotta); }

/* All Capsule Buttons with Rich Hover Glows */
.btn {
  border: none;
  border-radius: 999px; /* Capsule */
  padding: 12px 22px;
  font-weight: 600;
  cursor: pointer;
  font-size: 0.84rem;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.btn-forest {
  background: linear-gradient(135deg, #1e5238, #2d6a4f);
  color: #fff;
  box-shadow: 0 4px 16px rgba(30, 82, 56, 0.3);
}
.btn-forest:hover {
  background: linear-gradient(135deg, #2d6a4f, #40916c);
  box-shadow: 0 6px 22px rgba(45, 106, 79, 0.45);
  transform: translateY(-2px);
}
.btn-terracotta {
  background: linear-gradient(135deg, #c85a32, #a44322);
  color: #fff;
  box-shadow: 0 4px 16px rgba(200, 90, 50, 0.3);
}
.btn-terracotta:hover {
  background: linear-gradient(135deg, #dd6b42, #c85a32);
  box-shadow: 0 6px 22px rgba(200, 90, 50, 0.45);
  transform: translateY(-2px);
}
.btn-secondary {
  background: rgba(255, 255, 255, 0.9);
  color: var(--primary-forest);
  border: 1px solid rgba(45, 106, 79, 0.25);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.btn-secondary:hover {
  background: #ffffff;
  border-color: var(--primary-green);
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(45, 106, 79, 0.15);
}

/* Preset Capsule Matrix */
.preset-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 9px;
  margin-bottom: 16px;
}
.preset-capsule {
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(45, 106, 79, 0.2);
  color: var(--primary-forest);
  padding: 9px 8px;
  border-radius: 999px; /* Capsule */
  font-size: 0.76rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  text-align: center;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02);
}
.preset-capsule:hover {
  background: var(--primary-green);
  border-color: var(--primary-green);
  color: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(45, 106, 79, 0.25);
}

/* Form Controls */
input[type=number], input[type=text], select {
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(45, 106, 79, 0.25);
  color: var(--text-dark);
  padding: 10px 16px;
  border-radius: 999px; /* Capsule inputs */
  font-size: 0.85rem;
  font-family: inherit;
  transition: all 0.2s;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.04);
}
input[type=number]:focus, input[type=text]:focus, select:focus {
  outline: none;
  border-color: var(--primary-green);
  box-shadow: 0 0 12px rgba(45, 106, 79, 0.25);
}
input[type=range] {
  width: 100%;
  accent-color: var(--primary-green);
  margin: 8px 0;
  height: 6px;
  border-radius: 999px;
  background: rgba(45, 106, 79, 0.15);
  cursor: pointer;
}
.form-group { margin-bottom: 14px; }
.form-group label {
  display: flex;
  justify-content: space-between;
  font-size: 0.79rem;
  color: var(--text-muted);
  margin-bottom: 6px;
  font-weight: 600;
}

/* Tab Management */
.tab-pane { display: none; }
.tab-pane.active { display: block; animation: softFade 0.25s ease-out; }
@keyframes softFade { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulseDot {
  0% { transform: scale(0.92); opacity: 0.8; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
  70% { transform: scale(1.08); opacity: 1; box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
  100% { transform: scale(0.92); opacity: 0.8; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}
.live-pulsing-dot {
  width: 9px;
  height: 9px;
  background-color: #22c55e;
  border-radius: 50%;
  display: inline-block;
  animation: pulseDot 1.8s infinite;
}

/* Light Data Tables */
.table-scroll {
  max-height: 290px;
  overflow-y: auto;
  border-radius: 12px;
  border: 1px solid rgba(45, 106, 79, 0.2);
}
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
th, td { padding: 9px 12px; border-bottom: 1px solid rgba(45, 106, 79, 0.15); text-align: left; }
th { background: rgba(240, 248, 243, 0.95); color: var(--primary-forest); font-weight: 700; position: sticky; top: 0; }
tr:hover td { background: rgba(255, 255, 255, 0.75); }
</style>
</head>
<body>

<!-- BOTANICAL CAPSULE HEADER -->
<header>
  <div class="brand">
    <div class="brand-icon">
      <svg viewBox="0 0 24 24"><path d="M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75C7 8 17 8 17 8z"/></svg>
    </div>
    <div class="brand-text">
      <h1>AI Snooker Ball Identification & Laser Aiming</h1>
      <p>Precision Mechatronics System • Coordinate Origin: Top-Right (0,0)</p>
    </div>
  </div>

  <div class="nav-pills">
    <button class="nav-btn active" onclick="showTab('tab-table', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
      3D Table View
    </button>
    <button class="nav-btn" onclick="showTab('tab-camera', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
      Camera & Homography
    </button>
    <button class="nav-btn" onclick="showTab('tab-servo', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
      Servo & Laser
    </button>
    <button class="nav-btn" onclick="showTab('tab-balls', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="4"></circle></svg>
      Detected Balls
    </button>
    <button class="nav-btn" onclick="showTab('tab-foul', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
      Foul Repositioning
    </button>
    <button class="nav-btn" onclick="showTab('tab-calib', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
      9-Point Calibration
    </button>
    <button class="nav-btn" onclick="showTab('tab-logs', this)">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>
      Logs
    </button>
  </div>

  <div class="header-actions">
    <label class="sim-pill">
      <input type="checkbox" id="chkSim" checked onchange="toggleSim(this.checked)">
      <span>SIMULATION</span>
    </label>
    <span id="badgeESP" class="status-pill badge-sim">ESP32: SIMULATED</span>
    <button class="btn btn-terracotta" style="padding:8px 18px; font-size:0.8rem;" onclick="emergencyStop()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/></svg>
      E-STOP
    </button>
  </div>
</header>

<!-- MAIN WORKSPACE -->
<main>
  <!-- LEFT: 3D REALISTIC NATURE VIEWPORT -->
  <section class="glass-panel" style="padding:18px;">
    <div class="panel-header">
      <span style="display:flex; align-items:center; gap:8px;">
        Interactive 3D Physical Simulation
        <span class="tag-pill">6 Regulation Pockets</span>
      </span>
      <span class="mono" id="lblCursor" style="font-size:0.77rem; color:var(--primary-green); font-weight:600;">Cursor: Click table to aim</span>
    </div>

    <!-- 3D Viewport Box -->
    <div class="viewport-3d-box">
      <div class="viewport-toolbar">
        <button class="viewport-btn active" id="btnViewPersp" onclick="setCameraView('perspective')">Perspective 3D</button>
        <button class="viewport-btn" id="btnViewTop" onclick="setCameraView('topdown')">Top-Down (Bird's Eye)</button>
        <button class="viewport-btn" id="btnViewEnd" onclick="setCameraView('end')">Player View</button>
        <button class="viewport-btn" onclick="resetCamera()">Reset View</button>
        <button class="viewport-btn" id="btnTogglePip" onclick="toggleCameraPip()">📹 Overhead Camera PIP</button>
      </div>

      <div id="three-container" style="width:100%; height:100%;"></div>

      <!-- Floating Picture-in-Picture Live Video Feed -->
      <div id="cameraPipBox" style="display:none; position:absolute; bottom:55px; right:14px; width:260px; height:160px; background:rgba(18,36,25,0.92); backdrop-filter:blur(10px); border:1.5px solid rgba(82,183,136,0.5); border-radius:16px; overflow:hidden; box-shadow:0 12px 30px rgba(0,0,0,0.35); z-index:20;">
        <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 12px; background:rgba(0,0,0,0.4); font-size:0.7rem; color:#80ed99; font-weight:700;">
          <span style="display:inline-flex; align-items:center; gap:6px;">
            <span class="live-pulsing-dot"></span>
            LIVE OVERHEAD CAM
          </span>
          <button onclick="toggleCameraPip()" style="background:transparent; border:none; color:#cbd5e1; cursor:pointer; font-size:0.85rem; font-weight:bold; line-height:1;">✕</button>
        </div>
        <img id="pipStreamImg" src="" style="width:100%; height:calc(100% - 28px); object-fit:cover; display:block;" onerror="this.src='/api/camera/stream'">
      </div>

      <div class="viewport-overlay-hud">
        <div class="hud-pill mono" id="hudCoords">Target: (0.0, 0.0) mm</div>
        <div class="hud-pill mono" id="hudAngles">Pan: 10.0° | Tilt: 150.0°</div>
        <div class="hud-pill mono" style="color:var(--text-muted);">Orbit: Left Drag | Pan: Right Drag | Zoom: Scroll</div>
      </div>
    </div>

    <!-- Telemetry Cards -->
    <div class="telemetry-grid">
      <div class="telemetry-card target">
        <small>Target Table Position (Xr, Yr)</small>
        <div id="lblTarget" class="value">0.0, 0.0 mm (TR Pocket / Home)</div>
      </div>
      <div class="telemetry-card angles">
        <small>Calculated Servo Angles (Pan, Tilt)</small>
        <div id="lblPredicted" class="value">Pan: 10.0° | Tilt: 150.0°</div>
      </div>
    </div>
  </section>

  <!-- RIGHT: CONTROLS & MANAGEMENT TABS -->
  <section class="glass-panel">
    <!-- TAB 1: 3D TABLE VIEW & TARGETING -->
    <div id="tab-table" class="tab-pane active">
      <div class="panel-header">
        <span>Target Positioning & Laser Execution</span>
      </div>
      <p style="font-size:0.81rem; color:var(--text-muted); margin-bottom:12px;">
        Click directly on the green tournament cloth in 3D to pick coordinates, or specify exact millimeters:
      </p>

      <div style="display:flex; gap:8px; margin-bottom:14px;">
        <input type="number" id="inpX" placeholder="X (0 - 1168.4 mm)" value="0.0" style="flex:1;">
        <input type="number" id="inpY" placeholder="Y (0 - 2341.9 mm)" value="0.0" style="flex:1;">
        <button class="btn btn-secondary" onclick="applyManualCoords()">Set</button>
      </div>

      <button class="btn btn-forest" style="width:100%; padding:13px; font-size:0.94rem; margin-bottom:16px;" onclick="sendAimCommand()">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>
        AIM LASER AT SELECTED TARGET
      </button>

      <!-- Preset Capsule Matrix (6 Regulation Pockets & 3 Central Calibration Targets) -->
      <div class="panel-header" style="font-size:0.86rem; margin-bottom:10px;">
        <span>Presets (6 Regulation Pockets & 3 Calibration Targets)</span>
      </div>
      <div class="preset-grid">
        <button class="preset-capsule" onclick="setPreset(0.0, 0.0)">TR Pocket (Home)</button>
        <button class="preset-capsule" onclick="setPreset(584.2, 0.0)">Top-Centre (TC)</button>
        <button class="preset-capsule" onclick="setPreset(1168.4, 0.0)">TL Pocket</button>
        <button class="preset-capsule" onclick="setPreset(0.0, 1170.94)">MR Pocket</button>
        <button class="preset-capsule" onclick="setPreset(584.2, 1170.94)">Centre Node (C)</button>
        <button class="preset-capsule" onclick="setPreset(1168.4, 1170.94)">ML Pocket</button>
        <button class="preset-capsule" onclick="setPreset(0.0, 2341.88)">BR Pocket</button>
        <button class="preset-capsule" onclick="setPreset(584.2, 2341.88)">Bottom-Centre (BC)</button>
        <button class="preset-capsule" onclick="setPreset(1168.4, 2341.88)">BL Pocket</button>
      </div>

      <div id="cmdFeedback" class="mono" style="font-size:0.77rem; color:var(--primary-forest); background:rgba(255,255,255,0.8); padding:11px 16px; border-radius:999px; border:1px solid rgba(45,106,79,0.25);">
        System Status: Ready. Plumb-line home: Top-Right (0,0).
      </div>
    </div>

    <!-- TAB 2: CAMERA & HOMOGRAPHY CALIBRATION -->
    <div id="tab-camera" class="tab-pane">
      <div class="panel-header">
        <span>Camera Homography & Calibration</span>
        <span class="tag-pill tag-online" id="camStreamPill" style="display:inline-flex; align-items:center; gap:6px;">
          <span class="live-pulsing-dot"></span> LIVE CAM
        </span>
      </div>

      <!-- LIVE VIDEO / CALIBRATION SCREEN -->
      <div style="background:rgba(255,255,255,0.88); border-radius:18px; border:1px solid rgba(45,106,79,0.22); padding:16px; margin-bottom:14px; box-shadow:0 4px 16px rgba(0,0,0,0.04);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
          <div style="display:inline-flex; align-items:center; gap:8px;">
            <span class="live-pulsing-dot" id="streamDot"></span>
            <span id="streamTitle" style="font-size:0.83rem; font-weight:700; color:var(--primary-forest);">Live Overhead Video Footage</span>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-secondary" id="btnModeLive" style="padding:4px 12px; font-size:0.72rem; border-color:var(--primary-green); font-weight:700;" onclick="setCameraDisplayMode('live')">🔴 Live Stream</button>
            <button class="btn btn-secondary" id="btnModeFreeze" style="padding:4px 12px; font-size:0.72rem;" onclick="freezeFrameForCalibration()">🎯 Freeze & Calibrate</button>
          </div>
        </div>

        <!-- Video & Canvas Container -->
        <div id="cameraViewportContainer" style="width:100%; height:320px; background:#0b1410; border-radius:14px; border:1px solid rgba(45,106,79,0.3); overflow:hidden; position:relative; display:flex; align-items:center; justify-content:center;">
          <!-- Live Stream Image -->
          <img id="liveStreamView" src="/api/camera/stream" alt="Live Overhead Stream" style="width:100%; height:100%; object-fit:contain; display:block;" onerror="handleStreamError(this)" onload="handleStreamLoaded(this)">
          
          <!-- Calibration Canvas (hidden in live mode, visible when frozen) -->
          <canvas id="cameraCanvas" width="640" height="360" style="display:none; width:100%; height:100%; object-fit:contain; cursor:crosshair;"></canvas>
          
          <!-- Overlay Loading/Error Placeholder -->
          <div id="streamLoadingOverlay" style="position:absolute; display:none; flex-direction:column; align-items:center; justify-content:center; gap:8px; color:#e2e8f0; font-size:0.8rem; background:rgba(11,20,16,0.9); inset:0;">
            <div style="font-size:1.8rem;">📡</div>
            <div id="streamLoadingMsg">Connecting to IP Webcam video stream...</div>
            <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 12px;" onclick="reloadLiveStream()">Retry Stream</button>
          </div>

          <!-- Mode Badge -->
          <div id="cameraModeBadge" style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.7); backdrop-filter:blur(6px); color:#4ade80; border:1px solid rgba(74,222,128,0.4); padding:3px 10px; border-radius:999px; font-size:0.68rem; font-weight:700; pointer-events:none;">
            ● LIVE STREAM
          </div>
        </div>

        <!-- Status & Actions below video -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; flex-wrap:wrap; gap:8px;">
          <div id="cornerStatus" class="mono" style="font-size:0.77rem; color:#b45309; font-weight:600;">
            Status: Live Video streaming. Click "Freeze & Calibrate" to set 4 table corners.
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-secondary" style="padding:4px 10px; font-size:0.72rem;" onclick="reloadLiveStream()">🔄 Refresh Stream</button>
            <button class="btn btn-terracotta" style="padding:4px 10px; font-size:0.72rem;" onclick="resetCorners()">Reset 4 Corners</button>
          </div>
        </div>
      </div>

      <!-- IP WEBCAM CONTROLS & CAPTURE -->
      <div style="background:rgba(45,106,79,0.08); border:1px solid rgba(45,106,79,0.25); border-radius:16px; padding:14px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--primary-green);">📱 IP Webcam Source & Detection</span>
          <span class="tag-pill tag-online" style="font-size:0.65rem;">PORT 8080</span>
        </div>
        <div style="display:flex; gap:8px; margin-bottom:10px; align-items:center;">
          <input type="text" id="inpIpWebcam" value="http://192.168.31.150:8080" style="flex:1;" placeholder="http://192.168.31.XXX:8080">
          <button class="btn btn-secondary" style="font-size:0.74rem; white-space:nowrap;" onclick="saveIpWebcam()">Update URL</button>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:6px;">
          <button class="btn btn-forest" style="flex:1; padding:10px;" onclick="freezeFrameForCalibration()">📸 Freeze Frame for Homography</button>
          <button class="btn btn-secondary" style="flex:1; padding:10px;" onclick="captureAndDetect()">🎱 Detect Balls from Live Feed</button>
        </div>
        <div id="ipWebcamStatus" style="font-size:0.74rem; color:var(--text-muted); margin-top:6px;"></div>
      </div>

      <!-- FILE UPLOAD / SAMPLE BACKUP -->
      <p style="font-size:0.78rem; color:var(--text-muted); margin-bottom:8px;">
        Corner Order: <strong>1: TR (0,0)</strong> &rarr; <strong>2: TL (1168.4, 0)</strong> &rarr; <strong>3: BL (1168.4, 2341.9)</strong> &rarr; <strong>4: BR (0, 2341.9)</strong>
      </p>
      <div style="display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap;">
        <input type="file" id="fileUpload" accept="image/*" onchange="uploadImage(this)" style="display:none;">
        <button class="btn btn-secondary" style="flex:1;" onclick="document.getElementById('fileUpload').click()">Upload Photo File</button>
        <button class="btn btn-secondary" style="flex:1;" onclick="loadSampleImage()">Load Synthetic Table</button>
      </div>

      <button class="btn btn-forest" style="width:100%; padding:11px; margin-bottom:14px;" onclick="computeHomography()">
        COMPUTE PLANAR HOMOGRAPHY MATRIX
      </button>

      <div class="panel-header" style="font-size:0.86rem;">Laser Spot Closed-Loop Verification</div>
      <button class="btn btn-secondary" style="width:100%; padding:10px; font-weight:700;" onclick="detectSpot()">
        AUTOMATICALLY DETECT SPOT & COMPUTE ERROR
      </button>
      <div id="spotResult" class="mono" style="font-size:0.77rem; margin-top:8px; color:var(--primary-green); font-weight:600;"></div>
    </div>

    <!-- TAB 3: SERVO HARDWARE & LASER SAFETY -->
    <div id="tab-servo" class="tab-pane">
      <div class="panel-header">
        <span>Hardware Configuration & Controls</span>
      </div>

      <div style="background:rgba(255,255,255,0.75); padding:14px; border-radius:16px; border:1px solid rgba(45,106,79,0.2); margin-bottom:16px;">
        <label style="font-size:0.78rem; color:var(--primary-forest); display:block; margin-bottom:6px; font-weight:700;">ESP32 Wi-Fi Target IP:</label>
        <div style="display:flex; gap:6px; margin-bottom:8px;">
          <input type="text" id="inpEspIp" value="http://192.168.31.2" style="flex:1;">
          <button class="btn btn-forest" onclick="saveEspIp()">Save</button>
          <button class="btn btn-secondary" onclick="pingEsp()">Ping</button>
        </div>
        <div style="display:flex; gap:6px;">
          <button class="btn btn-secondary" style="font-size:0.74rem; flex:1;" onclick="setEspPreset('http://192.168.4.1')">PoolBot AP (192.168.4.1)</button>
          <button class="btn btn-secondary" style="font-size:0.74rem; flex:1;" onclick="setEspPreset('http://192.168.31.2')">Home Wi-Fi (192.168.31.2)</button>
        </div>
      </div>

      <div class="form-group">
        <label>
          <span>Pan Servo (GPIO 18)</span>
          <span id="valPan" class="mono" style="color:var(--primary-green); font-weight:700;">10°</span>
        </label>
        <input type="range" id="rngPan" min="10" max="170" value="10" oninput="previewServo('pan', this.value)" onchange="sendServoAngle('pan', this.value)">
      </div>

      <div class="form-group">
        <label>
          <span>Tilt Servo (GPIO 19)</span>
          <span id="valTilt" class="mono" style="color:var(--primary-green); font-weight:700;">150°</span>
        </label>
        <input type="range" id="rngTilt" min="60" max="160" value="150" oninput="previewServo('tilt', this.value)" onchange="sendServoAngle('tilt', this.value)">
      </div>

      <div style="display:flex; gap:8px; margin-bottom:16px;">
        <button class="btn btn-secondary" style="flex:1;" onclick="centerServos()">Center (90°, 90°)</button>
        <button class="btn btn-secondary" style="flex:1;" onclick="homeServos()">Home / TR Pocket (10°, 150°)</button>
      </div>

      <div style="background:rgba(200,90,50,0.08); border:1px solid rgba(200,90,50,0.3); border-radius:16px; padding:14px;">
        <div style="font-size:0.8rem; color:var(--terracotta); font-weight:700; margin-bottom:4px;">LASER DIODE SAFETY (GPIO 23 - OFF BY DEFAULT)</div>
        <p style="font-size:0.74rem; color:var(--text-muted); margin-bottom:10px;">The laser requires operator trigger to energize. Never view direct beam.</p>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-terracotta" id="btnLaserToggle" onclick="toggleLaser()">TOGGLE LASER ON/OFF</button>
          <button class="btn btn-secondary" onclick="forceLaserOff()">FORCE OFF</button>
        </div>
      </div>
    </div>

    <!-- TAB 4: DETECTED BALLS -->
    <div id="tab-balls" class="tab-pane">
      <div class="panel-header">
        <span>YOLOv8 Detected Snooker Balls</span>
        <button class="btn btn-forest" style="padding:6px 14px; font-size:0.77rem;" onclick="runDetection()">Detect Balls</button>
      </div>
      <p style="font-size:0.78rem; color:var(--text-muted); margin-bottom:10px;">
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
        <select id="selFoulBall" style="width:100%;" onchange="onFoulBallSelected()">
          <option value="cue">White Cue Ball</option>
          <option value="black">Black Ball (Spot: 584.2, 2100.0 mm)</option>
          <option value="pink">Pink Ball (Spot: 584.2, 1756.4 mm)</option>
          <option value="blue">Blue Ball (Center: 584.2, 1170.94 mm)</option>
          <option value="brown">Brown Ball (Baulk Center: 584.2, 737.0 mm)</option>
          <option value="green">Green Ball (Baulk Left: 876.2, 737.0 mm)</option>
          <option value="yellow">Yellow Ball (Baulk Right: 292.2, 737.0 mm)</option>
        </select>
      </div>

      <div style="display:flex; gap:8px; margin-bottom:14px;">
        <input type="number" id="inpFoulX" placeholder="Original X (mm)" value="584.2" style="flex:1;">
        <input type="number" id="inpFoulY" placeholder="Original Y (mm)" value="2100.0" style="flex:1;">
      </div>

      <button class="btn btn-forest" style="width:100%; padding:13px; font-size:0.94rem;" onclick="repositionBallLaser()">
        AIM LASER AT PRE-FOUL SPOT
      </button>
    </div>

    <!-- TAB 6: 9-POINT CALIBRATION TABLE -->
    <div id="tab-calib" class="tab-pane">
      <div class="panel-header">
        <span>9-Point Empirical Calibration Database</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>ID</th><th>Label</th><th>X (mm)</th><th>Y (mm)</th><th>Pan</th><th>Tilt</th></tr></thead>
          <tbody id="calibTableBody"></tbody>
        </table>
      </div>
    </div>

    <!-- TAB 7: LOGS & CSV -->
    <div id="tab-logs" class="tab-pane">
      <div class="panel-header">
        <span>System & Hardware Logs</span>
        <button class="btn btn-secondary" style="padding:6px 14px; font-size:0.77rem;" onclick="exportCsv()">Export CSV</button>
      </div>
      <div id="logsBox" class="mono" style="max-height:280px; overflow-y:auto; font-size:0.75rem; background:rgba(255,255,255,0.75); padding:14px; border-radius:14px; color:var(--text-dark); border:1px solid rgba(45,106,79,0.2);"></div>
    </div>
  </section>
</main>

<script>
// ==============================================================
// 1. GEOMETRY & GLOBAL COORDINATE DEFINITIONS
// ==============================================================
const SNOOKER_W = 1168.4;       // mm (X axis, width)
const SNOOKER_L = 2341.88;      // mm (Y axis, length)
const SCALE_3D = 0.001;         // 1 unit = 1000 mm = 1 meter
const TABLE_W = SNOOKER_W * SCALE_3D; // 1.1684 m
const TABLE_L = SNOOKER_L * SCALE_3D; // 2.34188 m
const MOUNT_HEIGHT = 1.31515;   // 1315.15 mm = 1.31515 m
const BALL_DIAMETER_MM = 52.5;  // Configurable measured ball diameter (mm)
const BALL_RADIUS = (BALL_DIAMETER_MM / 2.0) * SCALE_3D;

let targetX = 0.0;
let targetY = 0.0;
let predPan = 10.0;
let predTilt = 150.0;
let isSim = true;
let isLaserActive = false;

// Convert physical (x_mm, y_mm) in Top-Right origin system to Three.js centered scene coords (x, z)
function tableToScene(x_mm, y_mm) {
  const x3d = (TABLE_W / 2.0) - (x_mm * SCALE_3D);
  const z3d = (y_mm * SCALE_3D) - (TABLE_L / 2.0);
  return { x: x3d, z: z3d };
}

// Convert Three.js scene (x3d, z3d) on the felt back to physical (x_mm, y_mm)
function sceneToTable(x3d, z3d) {
  let x_mm = ((TABLE_W / 2.0) - x3d) / SCALE_3D;
  let y_mm = (z3d + (TABLE_L / 2.0)) / SCALE_3D;
  x_mm = Math.max(0.0, Math.min(SNOOKER_W, Math.round(x_mm * 10) / 10));
  y_mm = Math.max(0.0, Math.min(SNOOKER_L, Math.round(y_mm * 10) / 10));
  return { x: x_mm, y: y_mm };
}

// ==============================================================
// 2. THREE.JS REALISTIC SCENE SETUP
// ==============================================================
let scene, camera, renderer, controls;
let feltMesh, laserRayLine, laserSpotMesh, targetMarkerMesh;
let laserAssemblyGroup, panHousing, tiltHousing;
let ballMeshes = [];
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function initThree() {
  const container = document.getElementById('three-container');
  const w = container.clientWidth;
  const h = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xdce7e0);

  camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 50);
  camera.position.set(1.5, 2.2, 2.4);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.05;
  controls.minDistance = 0.6;
  controls.maxDistance = 5.5;
  controls.target.set(0, 0, 0);

  setupLighting();
  buildSnookerTable();
  buildLaserAssembly();
  buildInteractiveMarkers();
  spawnDefaultSnookerBalls();

  renderer.domElement.addEventListener('mousemove', onThreeMouseMove);
  renderer.domElement.addEventListener('click', onThreeClick);
  window.addEventListener('resize', onWindowResize);

  animate();
}

function setupLighting() {
  const ambient = new THREE.AmbientLight(0xfffef7, 0.85);
  scene.add(ambient);

  const overheadSpot = new THREE.SpotLight(0xfffcf0, 2.4, 6.5, Math.PI / 3, 0.45, 1.2);
  overheadSpot.position.set(0, 2.7, 0);
  overheadSpot.target.position.set(0, 0, 0);
  overheadSpot.castShadow = true;
  overheadSpot.shadow.mapSize.width = 2048;
  overheadSpot.shadow.mapSize.height = 2048;
  overheadSpot.shadow.camera.near = 0.5;
  overheadSpot.shadow.camera.far = 4.5;
  overheadSpot.shadow.bias = -0.0001;
  scene.add(overheadSpot);
  scene.add(overheadSpot.target);

  const sunLight1 = new THREE.DirectionalLight(0xfff7ed, 0.6);
  sunLight1.position.set(-2, 2.0, -2);
  scene.add(sunLight1);

  const skyLight2 = new THREE.DirectionalLight(0xe0f2fe, 0.45);
  skyLight2.position.set(2, 2.0, 2);
  scene.add(skyLight2);
}

// Procedural high-resolution felt cloth texture
function createFeltTexture() {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 2048;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#0f5b37';
  ctx.fillRect(0, 0, c.width, c.height);

  for (let i = 0; i < 22000; i++) {
    const rx = Math.random() * c.width;
    const ry = Math.random() * c.height;
    ctx.fillStyle = Math.random() > 0.5 ? 'rgba(255,255,255,0.015)' : 'rgba(0,0,0,0.025)';
    ctx.fillRect(rx, ry, 2, 2);
  }

  const sx = c.width / SNOOKER_W;
  const sy = c.height / SNOOKER_L;

  const baulkY = (SNOOKER_L - 737.0) * sy;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, baulkY);
  ctx.lineTo(c.width, baulkY);
  ctx.stroke();

  const dRadius = 292.0 * sx;
  const centerX = c.width / 2.0;
  ctx.beginPath();
  ctx.arc(centerX, baulkY, dRadius, 0, Math.PI, false);
  ctx.stroke();

  const spots = [
    { x: SNOOKER_W / 2.0, y: SNOOKER_L - 737.0 },
    { x: (SNOOKER_W / 2.0) + 292.0, y: SNOOKER_L - 737.0 },
    { x: (SNOOKER_W / 2.0) - 292.0, y: SNOOKER_L - 737.0 },
    { x: SNOOKER_W / 2.0, y: SNOOKER_L / 2.0 },
    { x: SNOOKER_W / 2.0, y: 1756.4 },
    { x: SNOOKER_W / 2.0, y: 2100.0 }
  ];

  ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
  spots.forEach(sp => {
    const px = c.width - (sp.x * sx);
    const py = sp.y * sy;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  const texture = new THREE.CanvasTexture(c);
  texture.anisotropy = 8;
  return texture;
}

function buildSnookerTable() {
  const tableGroup = new THREE.Group();

  const feltGeo = new THREE.PlaneGeometry(TABLE_W, TABLE_L, 32, 64);
  const feltTex = createFeltTexture();
  const feltMat = new THREE.MeshStandardMaterial({
    map: feltTex,
    roughness: 0.82,
    metalness: 0.04
  });
  feltMesh = new THREE.Mesh(feltGeo, feltMat);
  feltMesh.rotation.x = -Math.PI / 2;
  feltMesh.receiveShadow = true;
  tableGroup.add(feltMesh);

  const slateGeo = new THREE.BoxGeometry(TABLE_W + 0.02, 0.04, TABLE_L + 0.02);
  const slateMat = new THREE.MeshStandardMaterial({ color: 0x242e2b, roughness: 0.85 });
  const slateMesh = new THREE.Mesh(slateGeo, slateMat);
  slateMesh.position.y = -0.021;
  tableGroup.add(slateMesh);

  const railWidth = 0.085;
  const railHeight = 0.045;
  const railMat = new THREE.MeshStandardMaterial({
    color: 0x3d2314,
    roughness: 0.25,
    metalness: 0.1
  });

  const tbRailGeo = new THREE.BoxGeometry(TABLE_W + railWidth * 2, railHeight, railWidth);
  const topRail = new THREE.Mesh(tbRailGeo, railMat);
  topRail.position.set(0, railHeight / 2 - 0.005, -(TABLE_L / 2 + railWidth / 2));
  topRail.castShadow = true;
  tableGroup.add(topRail);

  const botRail = new THREE.Mesh(tbRailGeo, railMat);
  botRail.position.set(0, railHeight / 2 - 0.005, (TABLE_L / 2 + railWidth / 2));
  botRail.castShadow = true;
  tableGroup.add(botRail);

  const sideSegL = (TABLE_L - 0.14) / 2.0;
  const sideRailGeo = new THREE.BoxGeometry(railWidth, railHeight, sideSegL);

  [-1, 1].forEach(side => {
    [-1, 1].forEach(seg => {
      const rMesh = new THREE.Mesh(sideRailGeo, railMat);
      const rx = side * (TABLE_W / 2 + railWidth / 2);
      const rz = seg * (sideSegL / 2 + 0.06);
      rMesh.position.set(rx, railHeight / 2 - 0.005, rz);
      rMesh.castShadow = true;
      tableGroup.add(rMesh);
    });
  });

  // Exactly 6 Regulation Pockets
  const pocketLocations = [
    { name: "TR", x: TABLE_W / 2, z: -TABLE_L / 2, corner: true },
    { name: "TL", x: -TABLE_W / 2, z: -TABLE_L / 2, corner: true },
    { name: "MR", x: TABLE_W / 2 + 0.01, z: 0.0, corner: false },
    { name: "ML", x: -TABLE_W / 2 - 0.01, z: 0.0, corner: false },
    { name: "BR", x: TABLE_W / 2, z: TABLE_L / 2, corner: true },
    { name: "BL", x: -TABLE_W / 2, z: TABLE_L / 2, corner: true }
  ];

  const pocketInteriorMat = new THREE.MeshStandardMaterial({ color: 0x09120e, roughness: 0.95 });
  const brassMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.85, roughness: 0.25 });

  pocketLocations.forEach(p => {
    const r = p.corner ? 0.042 : 0.038;
    const dropGeo = new THREE.CylinderGeometry(r, r * 0.85, 0.06, 24);
    const dropMesh = new THREE.Mesh(dropGeo, pocketInteriorMat);
    dropMesh.position.set(p.x, -0.025, p.z);
    tableGroup.add(dropMesh);

    const plateGeo = new THREE.TorusGeometry(r + 0.005, 0.006, 8, 24, p.corner ? Math.PI : Math.PI * 2);
    const plateMesh = new THREE.Mesh(plateGeo, brassMat);
    plateMesh.rotation.x = Math.PI / 2;
    plateMesh.position.set(p.x, 0.02, p.z);
    tableGroup.add(plateMesh);
  });

  scene.add(tableGroup);
}

// 3D Laser Assembly: Mounted at physical height Z = 1315.15 mm above Top-Right corner (0,0)
function buildLaserAssembly() {
  laserAssemblyGroup = new THREE.Group();
  const trPos = tableToScene(0.0, 0.0);

  const standMat = new THREE.MeshStandardMaterial({ color: 0x4a2e1b, roughness: 0.5 });
  const postGeo = new THREE.BoxGeometry(0.04, MOUNT_HEIGHT, 0.04);
  const postMesh = new THREE.Mesh(postGeo, standMat);
  postMesh.position.set(trPos.x + 0.08, MOUNT_HEIGHT / 2, trPos.z - 0.08);
  postMesh.castShadow = true;
  laserAssemblyGroup.add(postMesh);

  const bracketGeo = new THREE.BoxGeometry(0.12, 0.02, 0.12);
  const bracketMesh = new THREE.Mesh(bracketGeo, standMat);
  bracketMesh.position.set(trPos.x, MOUNT_HEIGHT, trPos.z);
  laserAssemblyGroup.add(bracketMesh);

  panHousing = new THREE.Group();
  panHousing.position.set(trPos.x, MOUNT_HEIGHT, trPos.z);

  const servoMetalMat = new THREE.MeshStandardMaterial({ color: 0x2d3748, metalness: 0.7, roughness: 0.3 });
  const panBody = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.04), servoMetalMat);
  panBody.position.y = -0.02;
  panHousing.add(panBody);

  tiltHousing = new THREE.Group();
  tiltHousing.position.set(0, -0.045, 0);

  const laserDiodeMat = new THREE.MeshStandardMaterial({ color: 0xd97706, metalness: 0.8, roughness: 0.2 });
  const laserPointer = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.05, 16), laserDiodeMat);
  laserPointer.rotation.x = Math.PI / 2;
  tiltHousing.add(laserPointer);

  panHousing.add(tiltHousing);
  laserAssemblyGroup.add(panHousing);
  scene.add(laserAssemblyGroup);

  const rayMat = new THREE.LineBasicMaterial({
    color: 0xef4444,
    linewidth: 2,
    transparent: true,
    opacity: 0.95
  });
  const rayGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, 0)
  ]);
  laserRayLine = new THREE.Line(rayGeo, rayMat);
  scene.add(laserRayLine);

  const spotGeo = new THREE.CircleGeometry(0.012, 24);
  const spotMat = new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.95 });
  laserSpotMesh = new THREE.Mesh(spotGeo, spotMat);
  laserSpotMesh.rotation.x = -Math.PI / 2;
  laserSpotMesh.position.y = 0.002;
  scene.add(laserSpotMesh);
}

function buildInteractiveMarkers() {
  const ringGeo = new THREE.RingGeometry(0.02, 0.025, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x2d6a4f, side: THREE.DoubleSide });
  targetMarkerMesh = new THREE.Mesh(ringGeo, ringMat);
  targetMarkerMesh.rotation.x = -Math.PI / 2;
  targetMarkerMesh.position.y = 0.001;
  scene.add(targetMarkerMesh);
}

// 3D Realistic Snooker Balls
const SNOOKER_COLORS = {
  "cue": 0xfcfbfa,
  "red": 0xdc2626,
  "yellow": 0xeab308,
  "green": 0x15803d,
  "brown": 0x78350f,
  "blue": 0x1d4ed8,
  "pink": 0xf472b6,
  "black": 0x18181b
};

function createBallMesh(type, colorHex) {
  const ballGeo = new THREE.SphereGeometry(BALL_RADIUS, 32, 32);
  const ballMat = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: 0.12,
    metalness: 0.05
  });
  const mesh = new THREE.Mesh(ballGeo, ballMat);
  mesh.castShadow = true;
  mesh.receiveShadow = false;

  const shadowGeo = new THREE.CircleGeometry(BALL_RADIUS * 1.1, 16);
  const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25 });
  const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
  shadowMesh.rotation.x = -Math.PI / 2;
  shadowMesh.position.y = -BALL_RADIUS + 0.001;
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

function updateLaserVisualization() {
  const trPos = tableToScene(0.0, 0.0);
  const targetPos = tableToScene(targetX, targetY);

  targetMarkerMesh.position.x = targetPos.x;
  targetMarkerMesh.position.z = targetPos.z;

  laserSpotMesh.position.x = targetPos.x;
  laserSpotMesh.position.z = targetPos.z;
  laserSpotMesh.visible = isLaserActive || isSim;

  const dx = targetPos.x - trPos.x;
  const dy = 0.0 - MOUNT_HEIGHT;
  const dz = targetPos.z - trPos.z;
  const horizontalDist = Math.sqrt(dx * dx + dz * dz);

  const targetYaw = Math.atan2(dx, dz);
  const targetPitch = Math.atan2(dy, horizontalDist);

  panHousing.rotation.y = targetYaw;
  tiltHousing.rotation.x = -targetPitch - Math.PI / 2;

  const startPt = new THREE.Vector3(trPos.x, MOUNT_HEIGHT - 0.045, trPos.z);
  const endPt = new THREE.Vector3(targetPos.x, 0.002, targetPos.z);
  laserRayLine.geometry.setFromPoints([startPt, endPt]);
  laserRayLine.visible = isLaserActive || isSim;
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
    document.getElementById('inpX').value = targetX;
    document.getElementById('inpY').value = targetY;
    updatePred();
  }
}

function setCameraView(mode) {
  document.querySelectorAll('.viewport-btn').forEach(b => b.classList.remove('active'));
  if (mode === 'topdown') {
    document.getElementById('btnViewTop').classList.add('active');
    camera.position.set(0, 3.1, 0.001);
    controls.target.set(0, 0, 0);
  } else if (mode === 'end') {
    document.getElementById('btnViewEnd').classList.add('active');
    camera.position.set(0, 1.2, 2.2);
    controls.target.set(0, 0, 0.3);
  } else {
    document.getElementById('btnViewPersp').classList.add('active');
    camera.position.set(1.5, 2.2, 2.4);
    controls.target.set(0, 0, 0);
  }
  controls.update();
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
  controls.update();

  if (targetMarkerMesh) {
    const s = 1.0 + Math.sin(Date.now() * 0.005) * 0.12;
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

function toggleSim(val) {
  isSim = val;
  fetch('/api/simulation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: isSim })
  });
  updateESPBadge();
}

function updateESPBadge(connected = false) {
  const b = document.getElementById('badgeESP');
  if (isSim) {
    b.className = 'status-pill badge-sim'; b.textContent = 'ESP32: SIMULATED';
  } else if (connected) {
    b.className = 'status-pill badge-live'; b.textContent = 'ESP32: CONNECTED';
  } else {
    b.className = 'status-pill badge-off'; b.textContent = 'ESP32: OFFLINE';
  }
}

async function fetchNodes() {
  const res = await fetch('/api/nodes');
  const data = await res.json();
  const tb = document.getElementById('calibTableBody');
  tb.innerHTML = '';
  data.nodes.forEach(n => {
    tb.innerHTML += `<tr><td><strong>${n.id}</strong></td><td>${n.label}</td><td>${n.x_mm}</td><td>${n.y_mm}</td><td>${n.pan_deg}°</td><td>${n.tilt_deg}°</td></tr>`;
  });
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

// ── Camera, Live Stream & Homography Calibration ──
let cameraDisplayMode = 'live'; // 'live' or 'frozen'
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
    c.addEventListener('click', onCameraCanvasClick);
  }
}

function handleStreamLoaded(img) {
  const overlay = document.getElementById('streamLoadingOverlay');
  if (overlay) overlay.style.display = 'none';
  const pill = document.getElementById('camStreamPill');
  if (pill) {
    pill.className = 'tag-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> LIVE CAM';
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
      pill.className = 'tag-pill tag-sim';
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
      badge.style.color = '#4ade80';
      badge.style.borderColor = 'rgba(74,222,128,0.4)';
    }
    if (btnLive) { btnLive.style.borderColor = 'var(--primary-green)'; btnLive.style.fontWeight = '700'; }
    if (btnFreeze) { btnFreeze.style.borderColor = ''; btnFreeze.style.fontWeight = 'normal'; }
    document.getElementById('cornerStatus').textContent = 'Status: Live Video streaming. Click "Freeze & Calibrate" to set 4 table corners.';
  } else {
    if (streamImg) streamImg.style.display = 'none';
    if (canvas) canvas.style.display = 'block';
    if (badge) {
      badge.textContent = '🎯 CALIBRATION MODE (CLICK 4 CORNERS)';
      badge.style.color = '#fbbf24';
      badge.style.borderColor = 'rgba(251,191,36,0.4)';
    }
    if (btnLive) { btnLive.style.borderColor = ''; btnLive.style.fontWeight = 'normal'; }
    if (btnFreeze) { btnFreeze.style.borderColor = 'var(--primary-green)'; btnFreeze.style.fontWeight = '700'; }
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
        st.style.color = 'var(--primary-green)';
      };
      img.src = data.image_data;
    } else {
      st.innerHTML = `❌ ${data.error || 'Capture failed'}`;
      st.style.color = 'var(--terracotta)';
    }
  } catch (e) {
    st.textContent = '❌ Failed to capture frame. Check connection.';
    st.style.color = 'var(--terracotta)';
  }
}

function onCameraCanvasClick(e) {
  if (cameraDisplayMode !== 'frozen') return;
  if (homographyCorners.length >= 4) {
    alert("All 4 corners already set! Click 'Reset 4 Corners' to start over, or 'COMPUTE PLANAR HOMOGRAPHY MATRIX' to calibrate.");
    return;
  }
  const c = document.getElementById('cameraCanvas');
  const rect = c.getBoundingClientRect();
  const scaleX = c.width / rect.width;
  const scaleY = c.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;

  homographyCorners.push([px, py]);
  redrawCalibrationCanvas();
  updateCornerStatusText();
}

function updateCornerStatusText() {
  const labels = ['1: TR (0,0)', '2: TL (1168.4, 0)', '3: BL (1168.4, 2341.9)', '4: BR (0, 2341.9)'];
  const n = homographyCorners.length;
  const el = document.getElementById('cornerStatus');
  if (n < 4) {
    el.innerHTML = `Corners clicked: <strong>${n} / 4</strong>. Next corner: <strong style="color:var(--primary-green);">${labels[n]}</strong>`;
  } else {
    el.innerHTML = `<span style="color:var(--primary-green); font-weight:700;">✅ All 4 Corners Selected! Ready to compute homography matrix.</span>`;
  }
}

function redrawCalibrationCanvas() {
  const c = document.getElementById('cameraCanvas');
  if (!c || !cleanFrozenImg) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(cleanFrozenImg, 0, 0);

  const colors = ['#00e5ff', '#00e676', '#ffab00', '#ff1744'];
  const names = ['1: TR (0,0)', '2: TL (1168,0)', '3: BL (1168,2342)', '4: BR (0,2342)'];

  if (homographyCorners.length >= 2) {
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.9)';
    ctx.beginPath();
    ctx.moveTo(homographyCorners[0][0], homographyCorners[0][1]);
    for (let i = 1; i < homographyCorners.length; i++) {
      ctx.lineTo(homographyCorners[i][0], homographyCorners[i][1]);
    }
    if (homographyCorners.length === 4) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(45, 106, 79, 0.22)';
      ctx.fill();
    }
    ctx.stroke();
  }

  homographyCorners.forEach((pt, i) => {
    const [x, y] = pt;
    ctx.fillStyle = colors[i];
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(x - 14, y); ctx.lineTo(x + 14, y);
    ctx.moveTo(x, y - 14); ctx.lineTo(x, y + 14);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 16px Inter, sans-serif';
    const tag = names[i];
    const tw = ctx.measureText(tag).width;
    ctx.fillStyle = 'rgba(10, 20, 15, 0.85)';
    ctx.fillRect(x + 12, y - 14, tw + 10, 22);
    ctx.fillStyle = colors[i];
    ctx.fillText(tag, x + 17, y + 2);
  });
}

function resetCorners() {
  homographyCorners = [];
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
    document.getElementById('ipWebcamStatus').style.color = d.success ? 'var(--primary-green)' : 'var(--terracotta)';
    reloadLiveStream();
  }).catch(e => {
    document.getElementById('ipWebcamStatus').textContent = '❌ Error: ' + e.message;
    document.getElementById('ipWebcamStatus').style.color = 'var(--terracotta)';
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
      st.style.color = 'var(--terracotta)';
      return;
    }
    st.innerHTML = `✅ Captured ${capData.width}×${capData.height}. Running YOLOv8 snooker model...`;
    st.style.color = 'var(--primary-green)';

    await runDetection();
    st.innerHTML = `✅ Captured + Detection complete! View balls on table.`;
    showTab('tab-balls', document.querySelectorAll('.nav-btn')[3]);
  } catch (e) {
    st.textContent = '❌ Error: ' + e.message;
    st.style.color = 'var(--terracotta)';
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

window.onload = init;
</script>
</body>
</html>
"""

# ==============================================================
# 4. REST API ROUTES (PRESERVING ALL HARDWARE & CALIBRATION ENDPOINTS)
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
    add_log("E-STOP", "EMERGENCY STOP TRIGGERED")
    return jsonify({"success": True, "message": "Stopped"})

@app.route("/api/laser/toggle", methods=["POST"])
def api_laser_toggle():
    global laser_state
    laser_state = not laser_state
    if not simulation_mode:
        try: requests.get(f"{esp32_base_url}/laser", params={"state": "1" if laser_state else "0"}, timeout=1.5)
        except: pass
    add_log("LASER", f"Laser toggled {'ON' if laser_state else 'OFF'}")
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
        return jsonify({"detected": False})

    cx, cy, conf = spot
    mx, my = cx, cy
    if camera_homography.H_img_to_table is not None:
        table_pt = camera_homography.pixel_to_table(cx, cy)
        if table_pt:
            mx, my = table_pt

    ex = mx - tx
    ey = my - ty
    dist = math.hypot(ex, ey)
    add_log("SPOT", f"Target: ({tx:.1f}, {ty:.1f}) | Measured: ({mx:.1f}, {my:.1f}) | Error: {dist:.1f}mm")
    return jsonify({
        "detected": True,
        "pixel_x": cx,
        "pixel_y": cy,
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

    # If no frame has been manually uploaded/frozen yet, attempt to capture fresh snapshot from IP Webcam
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
        # IP Webcam serves snapshot at /photo.jpg
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
