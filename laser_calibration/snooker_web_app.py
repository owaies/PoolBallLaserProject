"""
Professional 3D Snooker Ball Identification & Laser Positioning System
Modern Nature Background, Capsule Buttons, Clean Minimal Typography (Inter),
Smooth Fluid Hover Animations, Glassmorphic Panels, and Interactive Three.js 3D Simulation.
"""

import os, sys, json, math, time, io, base64, socket, threading
from concurrent.futures import ThreadPoolExecutor
import requests
import requests.adapters
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify, send_file, Response, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser_calibration.snooker_geometry import SNOOKER, get_snooker_pockets, is_within_snooker_table
from laser_calibration.camera_calibration import SnookerCameraHomography
from laser_calibration.snooker_interpolator import SnookerLaserInterpolator
from laser_calibration.laser_spot_detector import detect_laser_spot
from laser_calibration.snooker_detection import SnookerBallDetector
from laser_calibration.foul_detector import FoulDetector, REGULATION_SPOTS

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

# Calibration Data
CALIB_FILE = os.path.join(os.path.dirname(__file__), "snooker_calibration.json")
with open(CALIB_FILE, "r") as f:
    calib_data = json.load(f)

interpolator = SnookerLaserInterpolator(calib_data["nodes"])
foul_detector = FoulDetector(interpolator)
camera_homography = SnookerCameraHomography()
detector = SnookerBallDetector("models/snooker/custom_table_best.pt")

# System State
esp32_base_url = "http://10.21.46.190"
simulation_mode = True
laser_state = False
current_pan = 10.0
current_tilt = 150.0
pan_trim = 0.0
tilt_trim = 0.0
emergency_stopped = False
ip_webcam_url = "http://10.21.46.181:8080"  # Default IP Webcam URL (update in UI)

# Persistent ESP32 HTTP Connection Session & Thread Safety Lock
esp32_session = requests.Session()
esp32_adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=10, max_retries=1)
esp32_session.mount("http://", esp32_adapter)
esp32_session.mount("https://", esp32_adapter)
esp32_lock = threading.Lock()

def call_esp32(endpoint: str, params=None, timeout=3.5, retries=1):
    """
    Thread-safe, keep-alive HTTP communication with ESP32 with automatic retry.
    Prevents socket thrashing and handles ESP32 Wi-Fi channel hops.
    """
    url = f"{esp32_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    last_err = None
    with esp32_lock:
        for attempt in range(retries + 1):
            try:
                resp = esp32_session.get(url, params=params, timeout=timeout)
                return resp
            except Exception as e:
                last_err = e
                time.sleep(0.12)
    if last_err:
        raise last_err

uploaded_frame_bgr = None
# Pre-load initial table frame from dataset so offline testing works immediately
default_sample = os.path.join(r"D:\Dataset", "photo58.jpg")
if not os.path.exists(default_sample):
    default_sample = os.path.join(r"D:\Dataset", "photo42.jpg")
if os.path.exists(default_sample):
    init_f = cv2.imread(default_sample)
    if init_f is not None:
        uploaded_frame_bgr = init_f
        _, buf = cv2.imencode(".jpg", init_f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

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
<script src="/static/js/three.min.js">
// ==============================================================
// 11. AI FOUL DETECTION & REFEREE REPOSITIONING LOGIC
// ==============================================================
let preShotBallCount = 0;

async function capturePreShotState() {
  const currentBalls = (typeof lastDetectedBalls !== "undefined" && lastDetectedBalls.length > 0) ? lastDetectedBalls : ((typeof detected_balls_cache !== "undefined" && detected_balls_cache.length > 0) ? detected_balls_cache : []);
  try {
    const res = await fetch('/api/foul/snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ balls: currentBalls })
    });
    const data = await res.json();
    if (data.success) {
      preShotBallCount = data.count;
      updatePreShotBadge(data.count);
      alert(`📸 Pre-shot reference state captured with ${data.count} balls! Proceed with your shot.`);
    }
  } catch (e) {
    alert(`Error capturing pre-shot state: ${e}`);
  }
}

function updatePreShotBadge(count) {
  const badge = document.getElementById('foulPreShotBadge');
  if (badge) {
    if (count > 0) {
      badge.className = 'tag-pill tag-online';
      badge.textContent = `Pre-Shot Reference: ${count} balls stored`;
    } else {
      badge.className = 'tag-pill';
      badge.textContent = 'Pre-Shot Reference: None';
    }
  }
}

async function analyzeShotForFoul() {
  const nominated = document.getElementById('selNominatedBall').value;
  try {
    const res = await fetch('/api/foul/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nominated: nominated, balls: (typeof lastDetectedBalls !== "undefined" ? lastDetectedBalls : []) })
    });
    const data = await res.json();
    if (data.success) {
      renderFoulReport(data.report);
    }
  } catch (e) {
    alert(`Shot analysis error: ${e}`);
  }
}

async function captureLiveAndAnalyzeFoul() {
  // Capture fresh frame from camera, run YOLO detection, then analyze foul
  try {
    await captureAndDetect();
    await analyzeShotForFoul();
  } catch (e) {
    alert(`Live cam shot analysis error: ${e}`);
  }
}

function renderFoulReport(rep) {
  const box = document.getElementById('foulVerdictBox');
  const icon = document.getElementById('foulVerdictIcon');
  const title = document.getElementById('foulVerdictTitle');
  const penaltyBadge = document.getElementById('foulPenaltyBadge');
  const desc = document.getElementById('foulVerdictDesc');
  const countBadge = document.getElementById('displacedCountBadge');
  const tb = document.getElementById('foulDisplacedTableBody');

  if (!box) return;
  box.style.display = 'block';

  if (rep.is_foul) {
    box.style.background = 'rgba(239, 68, 68, 0.12)';
    box.style.border = '1px solid rgba(239, 68, 68, 0.4)';
    icon.textContent = '🚨';
    title.textContent = `FOUL DETECTED: ${rep.foul_type.replace(/_/g, ' ')}`;
    title.style.color = '#dc2626';
    penaltyBadge.className = 'status-pill badge-off';
    penaltyBadge.textContent = `PENALTY: ${rep.penalty_points} PTS`;
    desc.innerHTML = `<strong>${rep.description}</strong><br><span style="font-size:0.75rem; color:var(--text-muted);">Nominated target was: <strong>${rep.nominated_target.toUpperCase()}</strong>. Displaced balls can be restored below using the physical laser pointer.</span>`;
  } else {
    box.style.background = 'rgba(34, 197, 94, 0.12)';
    box.style.border = '1px solid rgba(34, 197, 94, 0.4)';
    icon.textContent = '✅';
    title.textContent = 'LEGAL SHOT CONFIRMED';
    title.style.color = 'var(--primary-forest)';
    penaltyBadge.className = 'status-pill tag-online';
    penaltyBadge.textContent = 'NO FOUL';
    desc.textContent = rep.description;
  }

  // Render displaced balls table
  const disp = rep.displaced_balls || [];
  if (countBadge) countBadge.textContent = `${disp.length} balls displaced/potted`;

  if (tb) {
    if (disp.length === 0) {
      tb.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:14px;">No balls were displaced during this shot.</td></tr>';
    } else {
      tb.innerHTML = '';
      disp.forEach((d, idx) => {
        const isPotted = d.status === 'POTTED';
        const statusPill = isPotted 
          ? '<span class="status-pill badge-off" style="font-size:0.65rem;">POTTED</span>'
          : '<span class="status-pill badge-sim" style="font-size:0.65rem;">MOVED</span>';
        const postCoord = isPotted ? '<em>Pocketed</em>' : `(${d.post_x}, ${d.post_y})`;
        const shift = isPotted ? 'Off Table' : `${d.displacement_mm} mm`;

        tb.innerHTML += `<tr id="rowDisplaced_${idx}">
          <td><strong>${d.label}</strong></td>
          <td>${statusPill}</td>
          <td class="mono" style="color:var(--primary-green); font-weight:700;">(${d.pre_x}, ${d.pre_y}) mm</td>
          <td class="mono">${postCoord}</td>
          <td class="mono">${shift}</td>
          <td>
            <button class="btn btn-forest" style="padding:4px 10px; font-size:0.7rem;" onclick="aimLaserAtPreFoul(${d.pre_x}, ${d.pre_y}, '${d.label}')">
              🎯 Aim Laser to Restore
            </button>
          </td>
        </tr>`;
      });
    }
  }
}

async function aimLaserAtPreFoul(x, y, label) {
  setPreset(x, y);
  try {
    const res = await fetch('/api/foul/reposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: x, y: y })
    });
    const data = await res.json();
    if (data.success) {
      alert(`🎯 Physical laser aiming at pre-foul spot for ${label} (${x}mm, ${y}mm)! Place the ball under the red laser beam.`);
    }
  } catch (e) {
    alert(`Laser reposition error: ${e}`);
  }
}

async function aimAtRegulationSpot(spotKey) {
  try {
    const res = await fetch('/api/foul/regulation_spots');
    const data = await res.json();
    if (data.success && data.spots[spotKey]) {
      const sp = data.spots[spotKey];
      setPreset(sp.x_mm, sp.y_mm);
      await fetch('/api/foul/reposition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: sp.x_mm, y: sp.y_mm })
      });
      alert(`🎯 Physical laser aimed at ${sp.name} (${sp.x_mm}mm, ${sp.y_mm}mm) for color respotting.`);
    }
  } catch (e) {
    alert(`Error aiming at regulation spot: ${e}`);
  }
}


// ==============================================================
// 12. FULLY AUTONOMOUS LIVE REFEREE STATE MACHINE
// ==============================================================
let isAutoRefereeActive = false;
let autoRefereeState = 'STANDBY'; // 'STANDBY', 'READY', 'MOTION', 'SETTLING', 'ANALYZING', 'FOUL_HOLD'
let motionSamplerTimer = null;
let lastMotionLuma = null;
let motionDetectionCanvas = null;
let motionDetectionCtx = null;
let settleStartTime = null;
const MOTION_THRESHOLD = 3.2;      // Percent diff to register stroke
const SETTLE_DELAY_MS = 1400;      // 1.4s of calm to confirm balls stopped

let lastDisplacedBalls = [];
let currentRestoreIdx = 0;

function setRefereeMode(mode) {
  const autoPanel = document.getElementById('autoRefereePanel');
  const manualPanel = document.getElementById('manualRefereePanel');
  const btnManual = document.getElementById('btnModeManual');
  const tag = document.getElementById('refereeModeTag');

  if (mode === 'manual') {
    if (isAutoRefereeActive) toggleAutonomousReferee();
    if (autoPanel) autoPanel.style.display = 'none';
    if (manualPanel) manualPanel.style.display = 'block';
    if (btnManual) {
      btnManual.className = 'btn btn-forest';
      btnManual.textContent = '📸 Manual 2-Shot Mode: ON';
    }
    if (tag) tag.textContent = 'MANUAL 2-SHOT MODE';
  } else {
    if (autoPanel) autoPanel.style.display = 'block';
    if (manualPanel) manualPanel.style.display = 'none';
    if (btnManual) {
      btnManual.className = 'btn btn-secondary';
      btnManual.textContent = '📸 Manual 2-Shot Mode';
    }
    if (tag) tag.textContent = 'LIVE MOTION AI';
  }
}

let manualPreShotImgObj = null;
let manualCornerMode = false;
let manualPickedCorners = [];
let manualCalibratedCorners = [];

function drawManualPreShotCanvas() {
  const cvs = document.getElementById('cvsManualPreShot');
  const ph = document.getElementById('lblManualPrePlaceholder');
  if (!cvs || !manualPreShotImgObj) return;
  cvs.width = manualPreShotImgObj.naturalWidth || manualPreShotImgObj.width || 640;
  cvs.height = manualPreShotImgObj.naturalHeight || manualPreShotImgObj.height || 360;
  cvs.style.display = 'block';
  if (ph) ph.style.display = 'none';

  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.drawImage(manualPreShotImgObj, 0, 0, cvs.width, cvs.height);

  const pts = manualCornerMode ? manualPickedCorners : manualCalibratedCorners;
  if (pts && pts.length > 0) {
    if (pts.length === 4) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = Math.max(2, cvs.width / 320);
      ctx.stroke();
    }
    const labels = ['TR', 'TL', 'BL', 'BR'];
    pts.forEach((p, idx) => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], Math.max(6, cvs.width / 110), 0, Math.PI * 2);
      ctx.fillStyle = manualCornerMode ? '#f59e0b' : '#22c55e';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${Math.max(12, Math.round(cvs.width / 45))}px monospace`;
      ctx.fillText(labels[idx] || String(idx + 1), p[0] + 8, p[1] - 6);
    });
  }
}

function startManualCornerPick() {
  if (!manualPreShotImgObj) {
    alert('Please click "1. Take Pre-Shot Image" first.');
    return;
  }
  manualCornerMode = true;
  manualPickedCorners = [];
  const lbl = document.getElementById('lblManualHomogStatus');
  if (lbl) lbl.textContent = '🎯 Click Corner 1/4: Top-Right (TR) on Pre-Shot image';
  drawManualPreShotCanvas();
}

async function onManualPreShotCanvasClick(event) {
  if (!manualCornerMode || !manualPreShotImgObj) return;
  const cvs = document.getElementById('cvsManualPreShot');
  const rect = cvs.getBoundingClientRect();
  // Account for object-fit: contain scaling inside canvas box
  const scale = Math.min(rect.width / cvs.width, rect.height / cvs.height);
  const drawW = cvs.width * scale;
  const drawH = cvs.height * scale;
  const offX = (rect.width - drawW) / 2;
  const offY = (rect.height - drawH) / 2;
  const x = (event.clientX - rect.left - offX) / scale;
  const y = (event.clientY - rect.top - offY) / scale;
  if (x < 0 || y < 0 || x > cvs.width || y > cvs.height) return;

  manualPickedCorners.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
  drawManualPreShotCanvas();

  const names = ['Top-Right (TR)', 'Top-Left (TL)', 'Bottom-Left (BL)', 'Bottom-Right (BR)'];
  const lbl = document.getElementById('lblManualHomogStatus');
  if (manualPickedCorners.length < 4) {
    if (lbl) lbl.textContent = `🎯 Click Corner ${manualPickedCorners.length + 1}/4: ${names[manualPickedCorners.length]}`;
    return;
  }

  manualCornerMode = false;
  if (lbl) lbl.textContent = '⏳ Calibrating Homography & Re-detecting Pre-Shot...';
  try {
    const res = await fetch('/api/foul/manual_calibrate_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners: manualPickedCorners })
    });
    const data = await res.json();
    applyManualCalibrationResult(data);
  } catch (e) {
    alert(`Calibration error: ${e}`);
  }
}

async function autoCalibrateManualPreShot() {
  const lbl = document.getElementById('lblManualHomogStatus');
  if (lbl) lbl.textContent = '⏳ Auto-detecting table corners...';
  try {
    const res = await fetch('/api/foul/manual_calibrate_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ auto: true })
    });
    const data = await res.json();
    applyManualCalibrationResult(data);
  } catch (e) {
    alert(`Auto-calibration error: ${e}`);
  }
}

function applyManualCalibrationResult(data) {
  const lbl = document.getElementById('lblManualHomogStatus');
  const pill = document.getElementById('manualRefereePill');
  if (!data.success) {
    if (lbl) lbl.textContent = '⚠️ Calibration failed — click Pick 4 Corners';
    alert(data.error || 'Could not calibrate homography.');
    return;
  }
  manualCalibratedCorners = data.corners || [];
  drawManualPreShotCanvas();
  preShotBallCount = data.count || 0;
  if (data.balls && typeof update3DBalls === 'function') {
    lastDetectedBalls = data.balls;
    update3DBalls(data.balls);
  }
  if (lbl) lbl.textContent = `✅ Homography Calibrated (${preShotBallCount} balls locked)`;
  if (pill) {
    pill.className = preShotBallCount > 0 ? 'status-pill tag-online' : 'status-pill badge-sim';
    pill.textContent = preShotBallCount > 0
      ? `✅ CALIBRATED & PRE-SHOT LOCKED (${preShotBallCount} BALLS) — PLAY SHOT`
      : '⚠️ CALIBRATED BUT 0 BALLS DETECTED — CHECK TABLE/CAMERA';
  }
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function uploadManualPreShot(inputEl) {
  if (!inputEl || !inputEl.files || !inputEl.files[0]) return;
  try {
    const b64 = await readFileAsDataURL(inputEl.files[0]);
    inputEl.value = '';
    await captureManualPreShot(b64);
  } catch (e) {
    alert(`Failed to read Pre-Shot image: ${e}`);
  }
}

async function uploadManualPostShotAndCompare(inputEl) {
  if (!inputEl || !inputEl.files || !inputEl.files[0]) return;
  try {
    const b64 = await readFileAsDataURL(inputEl.files[0]);
    inputEl.value = '';
    await captureManualPostShotAndCompare(b64);
  } catch (e) {
    alert(`Failed to read Post-Shot image: ${e}`);
  }
}

async function captureManualPreShot(uploadedB64 = null) {
  const pill = document.getElementById('manualRefereePill');
  const lblHomog = document.getElementById('lblManualHomogStatus');
  if (pill) pill.textContent = '⏳ PROCESSING PRE-SHOT, CALIBRATING HOMOGRAPHY & DETECTING...';
  manualCornerMode = false;
  try {
    const payload = uploadedB64 ? { image_b64: uploadedB64 } : {};
    const res = await fetch('/api/foul/manual_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      alert(data.error || 'Failed to capture pre-shot image.');
      return;
    }
    preShotBallCount = data.count || 0;
    manualCalibratedCorners = data.corners || [];
    if (data.balls && typeof update3DBalls === 'function') {
      lastDetectedBalls = data.balls;
      update3DBalls(data.balls);
    }
    if (data.image_b64) {
      const img = new Image();
      img.onload = () => {
        manualPreShotImgObj = img;
        drawManualPreShotCanvas();
      };
      img.src = data.image_b64;
    }
    // Reset post-shot preview & restore bar for the new shot
    const postImg = document.getElementById('imgManualPostShot');
    const postPh = document.getElementById('lblManualPostPlaceholder');
    const restoreBar = document.getElementById('manualRestoreBar');
    if (postImg) { postImg.style.display = 'none'; postImg.src = ''; }
    if (postPh) postPh.style.display = 'block';
    if (restoreBar) restoreBar.style.display = 'none';

    if (lblHomog) {
      lblHomog.textContent = data.homography_calibrated
        ? `✅ Homography Calibrated (${preShotBallCount} balls)`
        : '⚠️ Homography Not Calibrated — Click "Pick 4 Corners"';
    }

    if (pill) {
      if (!data.homography_calibrated) {
        pill.className = 'status-pill badge-off';
        pill.textContent = '⚠️ CLICK "PICK 4 CORNERS" ON PRE-SHOT IMAGE TO CALIBRATE';
      } else if (preShotBallCount === 0) {
        pill.className = 'status-pill badge-off';
        pill.textContent = '⚠️ CALIBRATED BUT 0 BALLS FOUND — CHECK CAMERA / PICK CORNERS';
      } else {
        pill.className = 'status-pill tag-online';
        pill.textContent = `✅ CALIBRATED & PRE-SHOT LOCKED (${preShotBallCount} BALLS) — PLAY SHOT`;
      }
    }
  } catch (e) {
    alert(`Pre-shot capture error: ${e}`);
  }
}

async function captureManualPostShotAndCompare(uploadedB64 = null) {
  const pill = document.getElementById('manualRefereePill');
  const nominated = (document.getElementById('selManualNominatedBall') || document.getElementById('selNominatedBall')).value;
  if (pill) pill.textContent = '⏳ ANALYZING POST-SHOT IMAGE & COMPARING...';
  try {
    const payload = uploadedB64 ? { nominated: nominated, image_b64: uploadedB64 } : { nominated: nominated };
    const res = await fetch('/api/foul/manual_postshot_compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      if (pill) {
        pill.className = 'status-pill badge-sim';
        pill.textContent = '📸 STEP 1: TAKE PRE-SHOT IMAGE FIRST';
      }
      alert(data.error || 'Please take a Pre-Shot image first.');
      return;
    }

    if (data.balls && typeof update3DBalls === 'function') {
      lastDetectedBalls = data.balls;
      update3DBalls(data.balls);
    }
    if (data.image_b64) {
      const img = document.getElementById('imgManualPostShot');
      const ph = document.getElementById('lblManualPostPlaceholder');
      if (img) { img.src = data.image_b64; img.style.display = 'block'; }
      if (ph) ph.style.display = 'none';
    }

    renderFoulReport(data.report);
    lastDisplacedBalls = (data.report && data.report.displaced_balls) ? data.report.displaced_balls : [];
    currentRestoreIdx = 0;

    const restoreBar = document.getElementById('manualRestoreBar');
    const lblStatus = document.getElementById('lblManualRestoreStatus');
    if (lastDisplacedBalls.length > 0) {
      const first = lastDisplacedBalls[0];
      setPreset(first.pre_x, first.pre_y);
      if (restoreBar) restoreBar.style.display = 'flex';
      if (lblStatus) {
        lblStatus.textContent = `🎯 Pointing Ball 1 of ${lastDisplacedBalls.length}: ${first.label} (${first.pre_x}, ${first.pre_y} mm)`;
      }
      if (pill) {
        pill.className = 'status-pill badge-off';
        pill.textContent = `🚨 ${lastDisplacedBalls.length} DISPLACED BALL(S) — LASER POINTING TO RESTORE`;
      }
    } else {
      if (restoreBar) restoreBar.style.display = 'none';
      if (pill) {
        pill.className = 'status-pill tag-online';
        pill.textContent = '✅ NO DISPLACED BALLS DETECTED';
      }
    }
  } catch (e) {
    alert(`Post-shot comparison error: ${e}`);
  }
}

async function pointNextDisplacedBall() {
  if (!lastDisplacedBalls || lastDisplacedBalls.length === 0) return;
  currentRestoreIdx = (currentRestoreIdx + 1) % lastDisplacedBalls.length;
  const target = lastDisplacedBalls[currentRestoreIdx];
  const lblStatus = document.getElementById('lblManualRestoreStatus');
  if (lblStatus) {
    lblStatus.textContent = `🎯 Pointing Ball ${currentRestoreIdx + 1} of ${lastDisplacedBalls.length}: ${target.label} (${target.pre_x}, ${target.pre_y} mm)`;
  }
  setPreset(target.pre_x, target.pre_y);
  try {
    await fetch('/api/foul/reposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: target.pre_x, y: target.pre_y })
    });
  } catch (e) {
    console.error('Error pointing next displaced ball:', e);
  }
}

function resetManualRefereeRound() {
  lastDisplacedBalls = [];
  currentRestoreIdx = 0;
  manualPreShotImgObj = null;
  manualCornerMode = false;
  manualPickedCorners = [];
  manualCalibratedCorners = [];
  const preCvs = document.getElementById('cvsManualPreShot');
  const prePh = document.getElementById('lblManualPrePlaceholder');
  const postImg = document.getElementById('imgManualPostShot');
  const postPh = document.getElementById('lblManualPostPlaceholder');
  const restoreBar = document.getElementById('manualRestoreBar');
  const pill = document.getElementById('manualRefereePill');
  const lblHomog = document.getElementById('lblManualHomogStatus');
  if (preCvs) preCvs.style.display = 'none';
  if (prePh) prePh.style.display = 'block';
  if (postImg) { postImg.style.display = 'none'; postImg.src = ''; }
  if (postPh) postPh.style.display = 'block';
  if (restoreBar) restoreBar.style.display = 'none';
  if (lblHomog) lblHomog.textContent = '📐 Homography: Waiting for Pre-Shot';
  if (pill) {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '📸 STEP 1: TAKE PRE-SHOT IMAGE BEFORE PLAY';
  }
}

function toggleAutonomousReferee() {
  isAutoRefereeActive = !isAutoRefereeActive;
  const btn = document.getElementById('btnToggleAutoReferee');
  if (isAutoRefereeActive) {
    setRefereeMode('auto');
    if (btn) {
      btn.className = 'btn btn-forest';
      btn.textContent = '🟢 AUTONOMOUS REFEREE: ON';
    }
    startAutonomousRefereeLoop();
  } else {
    if (btn) {
      btn.className = 'btn btn-secondary';
      btn.textContent = '⚪ AUTONOMOUS REFEREE: OFF';
    }
    stopAutonomousRefereeLoop();
  }
}

function startAutonomousRefereeLoop() {
  reloadAutoRefereeStream();
  if (!motionDetectionCanvas) {
    motionDetectionCanvas = document.createElement('canvas');
    motionDetectionCanvas.width = 160;
    motionDetectionCanvas.height = 90;
    motionDetectionCtx = motionDetectionCanvas.getContext('2d', { willReadFrequently: true });
  }

  setAutoState('INITIALIZING');
  // Lock baseline safely
  fetch('/api/foul/snapshot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ balls: (typeof lastDetectedBalls !== 'undefined' ? lastDetectedBalls : []) })
  }).then(res => res.json()).then(data => {
    preShotBallCount = data.count || 0;
    updatePreShotBadge(data.count || 0);
    setAutoState('READY');
    lastMotionLuma = null;
    if (motionSamplerTimer) clearInterval(motionSamplerTimer);
    motionSamplerTimer = setInterval(sampleLiveCameraMotion, 120);
  }).catch(err => {
    console.warn('Init auto referee baseline error:', err);
    setAutoState('READY');
    if (motionSamplerTimer) clearInterval(motionSamplerTimer);
    motionSamplerTimer = setInterval(sampleLiveCameraMotion, 120);
  });
}

function stopAutonomousRefereeLoop() {
  if (motionSamplerTimer) {
    clearInterval(motionSamplerTimer);
    motionSamplerTimer = null;
  }
  setAutoState('STANDBY');
  const bar = document.getElementById('barMotion');
  const lbl = document.getElementById('lblMotionVal');
  if (bar) bar.style.width = '0%';
  if (lbl) lbl.textContent = 'Motion: 0.0% | Referee Standby';
}

function setAutoState(newState) {
  autoRefereeState = newState;
  const pill = document.getElementById('autoRefereePill');
  const btnResume = document.getElementById('btnResumeAutoAfterFoul');
  if (!pill) return;

  if (newState === 'STANDBY') {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '⚪ REFEREE STANDBY (TAP ENABLE ABOVE)';
    if (btnResume) btnResume.style.display = 'none';
  } else if (newState === 'INITIALIZING') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> LOCKING PRE-SHOT BASELINE...';
  } else if (newState === 'READY') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> 🟢 READY FOR SHOT (PRE-SHOT LOCKED)';
    if (btnResume) btnResume.style.display = 'none';
  } else if (newState === 'MOTION') {
    pill.className = 'status-pill badge-sim';
    pill.style.background = 'rgba(234, 179, 8, 0.2)';
    pill.style.color = '#ca8a04';
    pill.style.borderColor = '#eab308';
    pill.innerHTML = '<span class="live-pulsing-dot" style="background:#eab308;"></span> 🟡 BALLS IN MOTION (WATCHING SHOT...)';
  } else if (newState === 'SETTLING') {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '⏳ BALLS STOPPING... SETTLING';
  } else if (newState === 'ANALYZING') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> 🔵 ANALYZING SHOT & FOUL RULES...';
  } else if (newState === 'FOUL_HOLD') {
    pill.className = 'status-pill badge-off';
    pill.textContent = '🚨 FOUL DETECTED! LASER AIMING AT MISPLACED BALL';
    if (btnResume) btnResume.style.display = 'inline-block';
  }
}

function handleAutoRefereeStreamLoaded(img) {
  const warn = document.getElementById('autoRefereeCamWarning');
  if (warn) warn.style.display = 'none';
}

function handleAutoRefereeStreamError(img) {
  const warn = document.getElementById('autoRefereeCamWarning');
  if (warn) warn.style.display = 'flex';
}

function reloadAutoRefereeStream() {
  const img = document.getElementById('autoRefereeLiveImg');
  if (img) img.src = '/api/camera/stream?t=' + Date.now();
}

function sampleLiveCameraMotion() {
  if (!isAutoRefereeActive || autoRefereeState === 'ANALYZING' || autoRefereeState === 'FOUL_HOLD') {
    return;
  }

  // Sample from autoRefereeLiveImg (same-origin, never cross-origin tainted)
  const streamImg = document.getElementById('autoRefereeLiveImg') || document.getElementById('liveStreamView');
  if (!streamImg || !streamImg.complete || !streamImg.naturalWidth) {
    const lbl = document.getElementById('lblMotionVal');
    if (lbl && isAutoRefereeActive) {
      lbl.textContent = 'Connecting to camera video stream... (Check IP Webcam)';
    }
    return;
  }

  try {
    motionDetectionCtx.drawImage(streamImg, 0, 0, 160, 90);
    const imgData = motionDetectionCtx.getImageData(0, 0, 160, 90);
    const data = imgData.data;

    if (!lastMotionLuma) {
      lastMotionLuma = new Uint8Array(160 * 90);
      for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        lastMotionLuma[j] = (data[i] * 299 + data[i+1] * 587 + data[i+2] * 114) >> 10;
      }
      return;
    }

    let diffSum = 0;
    const currLuma = new Uint8Array(160 * 90);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const luma = (data[i] * 299 + data[i+1] * 587 + data[i+2] * 114) >> 10;
      currLuma[j] = luma;
      const d = Math.abs(luma - lastMotionLuma[j]);
      if (d > 12) diffSum += d;
    }
    lastMotionLuma = currLuma;

    const motionPercent = Math.min(100.0, Math.round((diffSum / (160 * 90)) * 100) / 10);
    const bar = document.getElementById('barMotion');
    const lbl = document.getElementById('lblMotionVal');
    if (bar) bar.style.width = Math.min(100, motionPercent * 8) + '%';

    // State Transitions
    if (motionPercent >= MOTION_THRESHOLD) {
      // Balls or cue moving
      settleStartTime = null;
      if (autoRefereeState !== 'MOTION') {
        setAutoState('MOTION');
      }
      if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Balls Rolling...`;
    } else {
      // Calm table
      if (autoRefereeState === 'MOTION' || autoRefereeState === 'SETTLING') {
        if (!settleStartTime) {
          settleStartTime = Date.now();
          setAutoState('SETTLING');
        }
        const elapsed = Date.now() - settleStartTime;
        if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Settling: ${(elapsed/1000).toFixed(1)}s / ${(SETTLE_DELAY_MS/1000).toFixed(1)}s`;

        if (elapsed >= SETTLE_DELAY_MS) {
          // Balls have completely stopped! Trigger autonomous foul analysis
          settleStartTime = null;
          triggerAutonomousShotAnalysis();
        }
      } else {
        if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Table at Rest`;
      }
    }
  } catch (err) {
    // Canvas taint or cross-origin fallback
  }
}

async function triggerAutonomousShotAnalysis() {
  setAutoState('ANALYZING');
  try {
    const nominated = document.getElementById('selNominatedBall').value;
    const res = await fetch('/api/foul/auto_process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nominated: nominated })
    });
    const data = await res.json();
    if (!data.success) {
      setAutoState('READY');
      return;
    }

    const rep = data.report;
    renderFoulReport(rep);

    if (rep.is_foul) {
      // 🚨 FOUL DETECTED: Play warning alert tone
      playRefereeTone('foul');
      setAutoState('FOUL_HOLD');
      // Update UI with laser aiming feedback
      if (data.auto_laser_aimed) {
        document.getElementById('cmdFeedback').textContent = `🚨 FOUL: Laser auto-aimed at pre-foul spot for ${data.displaced_ball_aimed} (${data.laser_x}mm, ${data.laser_y}mm)!`;
      }
    } else {
      // ✅ CLEAN SHOT: Play clean chime, post-shot automatically becomes next pre-shot!
      playRefereeTone('clean');
      const box = document.getElementById('foulVerdictBox');
      if (box) {
        box.style.display = 'block';
        box.style.background = 'rgba(34, 197, 94, 0.14)';
        box.style.border = '1px solid rgba(34, 197, 94, 0.4)';
        document.getElementById('foulVerdictIcon').textContent = '✅';
        document.getElementById('foulVerdictTitle').textContent = 'LEGAL SHOT CONFIRMED';
        document.getElementById('foulVerdictTitle').style.color = 'var(--primary-forest)';
        document.getElementById('foulPenaltyBadge').className = 'status-pill tag-online';
        document.getElementById('foulPenaltyBadge').textContent = 'NO FOUL';
        document.getElementById('foulVerdictDesc').textContent = `${rep.description} • Rolled over to next pre-shot automatically!`;
      }
      // Update badge
      updatePreShotBadge(data.post_shot_ball_count);
      setTimeout(() => {
        if (isAutoRefereeActive) {
          setAutoState('READY');
        }
      }, 1500);
    }
  } catch (e) {
    console.error('Auto referee analysis error:', e);
    setAutoState('READY');
  }
}

function resumeAutonomousAfterFoul() {
  // Referee finished repositioning displaced balls: take fresh snapshot and resume auto mode
  captureAndDetect().then(() => {
    capturePreShotState().then(() => {
      setAutoState('READY');
      alert('✅ Table restored! Autonomous referee ready for the next shot.');
    });
  });
}

function playRefereeTone(type) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === 'foul') {
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.35);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } else {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, ctx.currentTime);
      osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.12);
      osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.24);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.45);
      osc.start();
      osc.stop(ctx.currentTime + 0.45);
    }
  } catch (e) {}
}

</script>
<script src="/static/js/OrbitControls.js">
// ==============================================================
// 11. AI FOUL DETECTION & REFEREE REPOSITIONING LOGIC
// ==============================================================
let preShotBallCount = 0;

async function capturePreShotState() {
  const currentBalls = (typeof lastDetectedBalls !== "undefined" && lastDetectedBalls.length > 0) ? lastDetectedBalls : ((typeof detected_balls_cache !== "undefined" && detected_balls_cache.length > 0) ? detected_balls_cache : []);
  try {
    const res = await fetch('/api/foul/snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ balls: currentBalls })
    });
    const data = await res.json();
    if (data.success) {
      preShotBallCount = data.count;
      updatePreShotBadge(data.count);
      alert(`📸 Pre-shot reference state captured with ${data.count} balls! Proceed with your shot.`);
    }
  } catch (e) {
    alert(`Error capturing pre-shot state: ${e}`);
  }
}

function updatePreShotBadge(count) {
  const badge = document.getElementById('foulPreShotBadge');
  if (badge) {
    if (count > 0) {
      badge.className = 'tag-pill tag-online';
      badge.textContent = `Pre-Shot Reference: ${count} balls stored`;
    } else {
      badge.className = 'tag-pill';
      badge.textContent = 'Pre-Shot Reference: None';
    }
  }
}

async function analyzeShotForFoul() {
  const nominated = document.getElementById('selNominatedBall').value;
  try {
    const res = await fetch('/api/foul/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nominated: nominated, balls: (typeof lastDetectedBalls !== "undefined" ? lastDetectedBalls : []) })
    });
    const data = await res.json();
    if (data.success) {
      renderFoulReport(data.report);
    }
  } catch (e) {
    alert(`Shot analysis error: ${e}`);
  }
}

async function captureLiveAndAnalyzeFoul() {
  // Capture fresh frame from camera, run YOLO detection, then analyze foul
  try {
    await captureAndDetect();
    await analyzeShotForFoul();
  } catch (e) {
    alert(`Live cam shot analysis error: ${e}`);
  }
}

function renderFoulReport(rep) {
  const box = document.getElementById('foulVerdictBox');
  const icon = document.getElementById('foulVerdictIcon');
  const title = document.getElementById('foulVerdictTitle');
  const penaltyBadge = document.getElementById('foulPenaltyBadge');
  const desc = document.getElementById('foulVerdictDesc');
  const countBadge = document.getElementById('displacedCountBadge');
  const tb = document.getElementById('foulDisplacedTableBody');

  if (!box) return;
  box.style.display = 'block';

  if (rep.is_foul) {
    box.style.background = 'rgba(239, 68, 68, 0.12)';
    box.style.border = '1px solid rgba(239, 68, 68, 0.4)';
    icon.textContent = '🚨';
    title.textContent = `FOUL DETECTED: ${rep.foul_type.replace(/_/g, ' ')}`;
    title.style.color = '#dc2626';
    penaltyBadge.className = 'status-pill badge-off';
    penaltyBadge.textContent = `PENALTY: ${rep.penalty_points} PTS`;
    desc.innerHTML = `<strong>${rep.description}</strong><br><span style="font-size:0.75rem; color:var(--text-muted);">Nominated target was: <strong>${rep.nominated_target.toUpperCase()}</strong>. Displaced balls can be restored below using the physical laser pointer.</span>`;
  } else {
    box.style.background = 'rgba(34, 197, 94, 0.12)';
    box.style.border = '1px solid rgba(34, 197, 94, 0.4)';
    icon.textContent = '✅';
    title.textContent = 'LEGAL SHOT CONFIRMED';
    title.style.color = 'var(--primary-forest)';
    penaltyBadge.className = 'status-pill tag-online';
    penaltyBadge.textContent = 'NO FOUL';
    desc.textContent = rep.description;
  }

  // Render displaced balls table
  const disp = rep.displaced_balls || [];
  if (countBadge) countBadge.textContent = `${disp.length} balls displaced/potted`;

  if (tb) {
    if (disp.length === 0) {
      tb.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:14px;">No balls were displaced during this shot.</td></tr>';
    } else {
      tb.innerHTML = '';
      disp.forEach((d, idx) => {
        const isPotted = d.status === 'POTTED';
        const statusPill = isPotted 
          ? '<span class="status-pill badge-off" style="font-size:0.65rem;">POTTED</span>'
          : '<span class="status-pill badge-sim" style="font-size:0.65rem;">MOVED</span>';
        const postCoord = isPotted ? '<em>Pocketed</em>' : `(${d.post_x}, ${d.post_y})`;
        const shift = isPotted ? 'Off Table' : `${d.displacement_mm} mm`;

        tb.innerHTML += `<tr id="rowDisplaced_${idx}">
          <td><strong>${d.label}</strong></td>
          <td>${statusPill}</td>
          <td class="mono" style="color:var(--primary-green); font-weight:700;">(${d.pre_x}, ${d.pre_y}) mm</td>
          <td class="mono">${postCoord}</td>
          <td class="mono">${shift}</td>
          <td>
            <button class="btn btn-forest" style="padding:4px 10px; font-size:0.7rem;" onclick="aimLaserAtPreFoul(${d.pre_x}, ${d.pre_y}, '${d.label}')">
              🎯 Aim Laser to Restore
            </button>
          </td>
        </tr>`;
      });
    }
  }
}

async function aimLaserAtPreFoul(x, y, label) {
  setPreset(x, y);
  try {
    const res = await fetch('/api/foul/reposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: x, y: y })
    });
    const data = await res.json();
    if (data.success) {
      alert(`🎯 Physical laser aiming at pre-foul spot for ${label} (${x}mm, ${y}mm)! Place the ball under the red laser beam.`);
    }
  } catch (e) {
    alert(`Laser reposition error: ${e}`);
  }
}

async function aimAtRegulationSpot(spotKey) {
  try {
    const res = await fetch('/api/foul/regulation_spots');
    const data = await res.json();
    if (data.success && data.spots[spotKey]) {
      const sp = data.spots[spotKey];
      setPreset(sp.x_mm, sp.y_mm);
      await fetch('/api/foul/reposition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: sp.x_mm, y: sp.y_mm })
      });
      alert(`🎯 Physical laser aimed at ${sp.name} (${sp.x_mm}mm, ${sp.y_mm}mm) for color respotting.`);
    }
  } catch (e) {
    alert(`Error aiming at regulation spot: ${e}`);
  }
}


// ==============================================================
// 12. FULLY AUTONOMOUS LIVE REFEREE STATE MACHINE
// ==============================================================
let isAutoRefereeActive = false;
let autoRefereeState = 'STANDBY'; // 'STANDBY', 'READY', 'MOTION', 'SETTLING', 'ANALYZING', 'FOUL_HOLD'
let motionSamplerTimer = null;
let lastMotionLuma = null;
let motionDetectionCanvas = null;
let motionDetectionCtx = null;
let settleStartTime = null;
const MOTION_THRESHOLD = 3.2;      // Percent diff to register stroke
const SETTLE_DELAY_MS = 1400;      // 1.4s of calm to confirm balls stopped

let lastDisplacedBalls = [];
let currentRestoreIdx = 0;

function setRefereeMode(mode) {
  const autoPanel = document.getElementById('autoRefereePanel');
  const manualPanel = document.getElementById('manualRefereePanel');
  const btnManual = document.getElementById('btnModeManual');
  const tag = document.getElementById('refereeModeTag');

  if (mode === 'manual') {
    if (isAutoRefereeActive) toggleAutonomousReferee();
    if (autoPanel) autoPanel.style.display = 'none';
    if (manualPanel) manualPanel.style.display = 'block';
    if (btnManual) {
      btnManual.className = 'btn btn-forest';
      btnManual.textContent = '📸 Manual 2-Shot Mode: ON';
    }
    if (tag) tag.textContent = 'MANUAL 2-SHOT MODE';
  } else {
    if (autoPanel) autoPanel.style.display = 'block';
    if (manualPanel) manualPanel.style.display = 'none';
    if (btnManual) {
      btnManual.className = 'btn btn-secondary';
      btnManual.textContent = '📸 Manual 2-Shot Mode';
    }
    if (tag) tag.textContent = 'LIVE MOTION AI';
  }
}

let manualPreShotImgObj = null;
let manualCornerMode = false;
let manualPickedCorners = [];
let manualCalibratedCorners = [];

function drawManualPreShotCanvas() {
  const cvs = document.getElementById('cvsManualPreShot');
  const ph = document.getElementById('lblManualPrePlaceholder');
  if (!cvs || !manualPreShotImgObj) return;
  cvs.width = manualPreShotImgObj.naturalWidth || manualPreShotImgObj.width || 640;
  cvs.height = manualPreShotImgObj.naturalHeight || manualPreShotImgObj.height || 360;
  cvs.style.display = 'block';
  if (ph) ph.style.display = 'none';

  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.drawImage(manualPreShotImgObj, 0, 0, cvs.width, cvs.height);

  const pts = manualCornerMode ? manualPickedCorners : manualCalibratedCorners;
  if (pts && pts.length > 0) {
    if (pts.length === 4) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = Math.max(2, cvs.width / 320);
      ctx.stroke();
    }
    const labels = ['TR', 'TL', 'BL', 'BR'];
    pts.forEach((p, idx) => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], Math.max(6, cvs.width / 110), 0, Math.PI * 2);
      ctx.fillStyle = manualCornerMode ? '#f59e0b' : '#22c55e';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${Math.max(12, Math.round(cvs.width / 45))}px monospace`;
      ctx.fillText(labels[idx] || String(idx + 1), p[0] + 8, p[1] - 6);
    });
  }
}

function startManualCornerPick() {
  if (!manualPreShotImgObj) {
    alert('Please click "1. Take Pre-Shot Image" first.');
    return;
  }
  manualCornerMode = true;
  manualPickedCorners = [];
  const lbl = document.getElementById('lblManualHomogStatus');
  if (lbl) lbl.textContent = '🎯 Click Corner 1/4: Top-Right (TR) on Pre-Shot image';
  drawManualPreShotCanvas();
}

async function onManualPreShotCanvasClick(event) {
  if (!manualCornerMode || !manualPreShotImgObj) return;
  const cvs = document.getElementById('cvsManualPreShot');
  const rect = cvs.getBoundingClientRect();
  // Account for object-fit: contain scaling inside canvas box
  const scale = Math.min(rect.width / cvs.width, rect.height / cvs.height);
  const drawW = cvs.width * scale;
  const drawH = cvs.height * scale;
  const offX = (rect.width - drawW) / 2;
  const offY = (rect.height - drawH) / 2;
  const x = (event.clientX - rect.left - offX) / scale;
  const y = (event.clientY - rect.top - offY) / scale;
  if (x < 0 || y < 0 || x > cvs.width || y > cvs.height) return;

  manualPickedCorners.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
  drawManualPreShotCanvas();

  const names = ['Top-Right (TR)', 'Top-Left (TL)', 'Bottom-Left (BL)', 'Bottom-Right (BR)'];
  const lbl = document.getElementById('lblManualHomogStatus');
  if (manualPickedCorners.length < 4) {
    if (lbl) lbl.textContent = `🎯 Click Corner ${manualPickedCorners.length + 1}/4: ${names[manualPickedCorners.length]}`;
    return;
  }

  manualCornerMode = false;
  if (lbl) lbl.textContent = '⏳ Calibrating Homography & Re-detecting Pre-Shot...';
  try {
    const res = await fetch('/api/foul/manual_calibrate_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners: manualPickedCorners })
    });
    const data = await res.json();
    applyManualCalibrationResult(data);
  } catch (e) {
    alert(`Calibration error: ${e}`);
  }
}

async function autoCalibrateManualPreShot() {
  const lbl = document.getElementById('lblManualHomogStatus');
  if (lbl) lbl.textContent = '⏳ Auto-detecting table corners...';
  try {
    const res = await fetch('/api/foul/manual_calibrate_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ auto: true })
    });
    const data = await res.json();
    applyManualCalibrationResult(data);
  } catch (e) {
    alert(`Auto-calibration error: ${e}`);
  }
}

function applyManualCalibrationResult(data) {
  const lbl = document.getElementById('lblManualHomogStatus');
  const pill = document.getElementById('manualRefereePill');
  if (!data.success) {
    if (lbl) lbl.textContent = '⚠️ Calibration failed — click Pick 4 Corners';
    alert(data.error || 'Could not calibrate homography.');
    return;
  }
  manualCalibratedCorners = data.corners || [];
  drawManualPreShotCanvas();
  preShotBallCount = data.count || 0;
  if (data.balls && typeof update3DBalls === 'function') {
    lastDetectedBalls = data.balls;
    update3DBalls(data.balls);
  }
  if (lbl) lbl.textContent = `✅ Homography Calibrated (${preShotBallCount} balls locked)`;
  if (pill) {
    pill.className = preShotBallCount > 0 ? 'status-pill tag-online' : 'status-pill badge-sim';
    pill.textContent = preShotBallCount > 0
      ? `✅ CALIBRATED & PRE-SHOT LOCKED (${preShotBallCount} BALLS) — PLAY SHOT`
      : '⚠️ CALIBRATED BUT 0 BALLS DETECTED — CHECK TABLE/CAMERA';
  }
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function uploadManualPreShot(inputEl) {
  if (!inputEl || !inputEl.files || !inputEl.files[0]) return;
  try {
    const b64 = await readFileAsDataURL(inputEl.files[0]);
    inputEl.value = '';
    await captureManualPreShot(b64);
  } catch (e) {
    alert(`Failed to read Pre-Shot image: ${e}`);
  }
}

async function uploadManualPostShotAndCompare(inputEl) {
  if (!inputEl || !inputEl.files || !inputEl.files[0]) return;
  try {
    const b64 = await readFileAsDataURL(inputEl.files[0]);
    inputEl.value = '';
    await captureManualPostShotAndCompare(b64);
  } catch (e) {
    alert(`Failed to read Post-Shot image: ${e}`);
  }
}

async function captureManualPreShot(uploadedB64 = null) {
  const pill = document.getElementById('manualRefereePill');
  const lblHomog = document.getElementById('lblManualHomogStatus');
  if (pill) pill.textContent = '⏳ PROCESSING PRE-SHOT, CALIBRATING HOMOGRAPHY & DETECTING...';
  manualCornerMode = false;
  try {
    const payload = uploadedB64 ? { image_b64: uploadedB64 } : {};
    const res = await fetch('/api/foul/manual_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      alert(data.error || 'Failed to capture pre-shot image.');
      return;
    }
    preShotBallCount = data.count || 0;
    manualCalibratedCorners = data.corners || [];
    if (data.balls && typeof update3DBalls === 'function') {
      lastDetectedBalls = data.balls;
      update3DBalls(data.balls);
    }
    if (data.image_b64) {
      const img = new Image();
      img.onload = () => {
        manualPreShotImgObj = img;
        drawManualPreShotCanvas();
      };
      img.src = data.image_b64;
    }
    // Reset post-shot preview & restore bar for the new shot
    const postImg = document.getElementById('imgManualPostShot');
    const postPh = document.getElementById('lblManualPostPlaceholder');
    const restoreBar = document.getElementById('manualRestoreBar');
    if (postImg) { postImg.style.display = 'none'; postImg.src = ''; }
    if (postPh) postPh.style.display = 'block';
    if (restoreBar) restoreBar.style.display = 'none';

    if (lblHomog) {
      lblHomog.textContent = data.homography_calibrated
        ? `✅ Homography Calibrated (${preShotBallCount} balls)`
        : '⚠️ Homography Not Calibrated — Click "Pick 4 Corners"';
    }

    if (pill) {
      if (!data.homography_calibrated) {
        pill.className = 'status-pill badge-off';
        pill.textContent = '⚠️ CLICK "PICK 4 CORNERS" ON PRE-SHOT IMAGE TO CALIBRATE';
      } else if (preShotBallCount === 0) {
        pill.className = 'status-pill badge-off';
        pill.textContent = '⚠️ CALIBRATED BUT 0 BALLS FOUND — CHECK CAMERA / PICK CORNERS';
      } else {
        pill.className = 'status-pill tag-online';
        pill.textContent = `✅ CALIBRATED & PRE-SHOT LOCKED (${preShotBallCount} BALLS) — PLAY SHOT`;
      }
    }
  } catch (e) {
    alert(`Pre-shot capture error: ${e}`);
  }
}

async function captureManualPostShotAndCompare(uploadedB64 = null) {
  const pill = document.getElementById('manualRefereePill');
  const nominated = (document.getElementById('selManualNominatedBall') || document.getElementById('selNominatedBall')).value;
  if (pill) pill.textContent = '⏳ ANALYZING POST-SHOT IMAGE & COMPARING...';
  try {
    const payload = uploadedB64 ? { nominated: nominated, image_b64: uploadedB64 } : { nominated: nominated };
    const res = await fetch('/api/foul/manual_postshot_compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      if (pill) {
        pill.className = 'status-pill badge-sim';
        pill.textContent = '📸 STEP 1: TAKE PRE-SHOT IMAGE FIRST';
      }
      alert(data.error || 'Please take a Pre-Shot image first.');
      return;
    }

    if (data.balls && typeof update3DBalls === 'function') {
      lastDetectedBalls = data.balls;
      update3DBalls(data.balls);
    }
    if (data.image_b64) {
      const img = document.getElementById('imgManualPostShot');
      const ph = document.getElementById('lblManualPostPlaceholder');
      if (img) { img.src = data.image_b64; img.style.display = 'block'; }
      if (ph) ph.style.display = 'none';
    }

    renderFoulReport(data.report);
    lastDisplacedBalls = (data.report && data.report.displaced_balls) ? data.report.displaced_balls : [];
    currentRestoreIdx = 0;

    const restoreBar = document.getElementById('manualRestoreBar');
    const lblStatus = document.getElementById('lblManualRestoreStatus');
    if (lastDisplacedBalls.length > 0) {
      const first = lastDisplacedBalls[0];
      setPreset(first.pre_x, first.pre_y);
      if (restoreBar) restoreBar.style.display = 'flex';
      if (lblStatus) {
        lblStatus.textContent = `🎯 Pointing Ball 1 of ${lastDisplacedBalls.length}: ${first.label} (${first.pre_x}, ${first.pre_y} mm)`;
      }
      if (pill) {
        pill.className = 'status-pill badge-off';
        pill.textContent = `🚨 ${lastDisplacedBalls.length} DISPLACED BALL(S) — LASER POINTING TO RESTORE`;
      }
    } else {
      if (restoreBar) restoreBar.style.display = 'none';
      if (pill) {
        pill.className = 'status-pill tag-online';
        pill.textContent = '✅ NO DISPLACED BALLS DETECTED';
      }
    }
  } catch (e) {
    alert(`Post-shot comparison error: ${e}`);
  }
}

async function pointNextDisplacedBall() {
  if (!lastDisplacedBalls || lastDisplacedBalls.length === 0) return;
  currentRestoreIdx = (currentRestoreIdx + 1) % lastDisplacedBalls.length;
  const target = lastDisplacedBalls[currentRestoreIdx];
  const lblStatus = document.getElementById('lblManualRestoreStatus');
  if (lblStatus) {
    lblStatus.textContent = `🎯 Pointing Ball ${currentRestoreIdx + 1} of ${lastDisplacedBalls.length}: ${target.label} (${target.pre_x}, ${target.pre_y} mm)`;
  }
  setPreset(target.pre_x, target.pre_y);
  try {
    await fetch('/api/foul/reposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: target.pre_x, y: target.pre_y })
    });
  } catch (e) {
    console.error('Error pointing next displaced ball:', e);
  }
}

function resetManualRefereeRound() {
  lastDisplacedBalls = [];
  currentRestoreIdx = 0;
  manualPreShotImgObj = null;
  manualCornerMode = false;
  manualPickedCorners = [];
  manualCalibratedCorners = [];
  const preCvs = document.getElementById('cvsManualPreShot');
  const prePh = document.getElementById('lblManualPrePlaceholder');
  const postImg = document.getElementById('imgManualPostShot');
  const postPh = document.getElementById('lblManualPostPlaceholder');
  const restoreBar = document.getElementById('manualRestoreBar');
  const pill = document.getElementById('manualRefereePill');
  const lblHomog = document.getElementById('lblManualHomogStatus');
  if (preCvs) preCvs.style.display = 'none';
  if (prePh) prePh.style.display = 'block';
  if (postImg) { postImg.style.display = 'none'; postImg.src = ''; }
  if (postPh) postPh.style.display = 'block';
  if (restoreBar) restoreBar.style.display = 'none';
  if (lblHomog) lblHomog.textContent = '📐 Homography: Waiting for Pre-Shot';
  if (pill) {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '📸 STEP 1: TAKE PRE-SHOT IMAGE BEFORE PLAY';
  }
}

function toggleAutonomousReferee() {
  isAutoRefereeActive = !isAutoRefereeActive;
  const btn = document.getElementById('btnToggleAutoReferee');
  if (isAutoRefereeActive) {
    setRefereeMode('auto');
    if (btn) {
      btn.className = 'btn btn-forest';
      btn.textContent = '🟢 AUTONOMOUS REFEREE: ON';
    }
    startAutonomousRefereeLoop();
  } else {
    if (btn) {
      btn.className = 'btn btn-secondary';
      btn.textContent = '⚪ AUTONOMOUS REFEREE: OFF';
    }
    stopAutonomousRefereeLoop();
  }
}

function startAutonomousRefereeLoop() {
  reloadAutoRefereeStream();
  if (!motionDetectionCanvas) {
    motionDetectionCanvas = document.createElement('canvas');
    motionDetectionCanvas.width = 160;
    motionDetectionCanvas.height = 90;
    motionDetectionCtx = motionDetectionCanvas.getContext('2d', { willReadFrequently: true });
  }

  setAutoState('INITIALIZING');
  // Lock baseline safely
  fetch('/api/foul/snapshot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ balls: (typeof lastDetectedBalls !== 'undefined' ? lastDetectedBalls : []) })
  }).then(res => res.json()).then(data => {
    preShotBallCount = data.count || 0;
    updatePreShotBadge(data.count || 0);
    setAutoState('READY');
    lastMotionLuma = null;
    if (motionSamplerTimer) clearInterval(motionSamplerTimer);
    motionSamplerTimer = setInterval(sampleLiveCameraMotion, 120);
  }).catch(err => {
    console.warn('Init auto referee baseline error:', err);
    setAutoState('READY');
    if (motionSamplerTimer) clearInterval(motionSamplerTimer);
    motionSamplerTimer = setInterval(sampleLiveCameraMotion, 120);
  });
}

function stopAutonomousRefereeLoop() {
  if (motionSamplerTimer) {
    clearInterval(motionSamplerTimer);
    motionSamplerTimer = null;
  }
  setAutoState('STANDBY');
  const bar = document.getElementById('barMotion');
  const lbl = document.getElementById('lblMotionVal');
  if (bar) bar.style.width = '0%';
  if (lbl) lbl.textContent = 'Motion: 0.0% | Referee Standby';
}

function setAutoState(newState) {
  autoRefereeState = newState;
  const pill = document.getElementById('autoRefereePill');
  const btnResume = document.getElementById('btnResumeAutoAfterFoul');
  if (!pill) return;

  if (newState === 'STANDBY') {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '⚪ REFEREE STANDBY (TAP ENABLE ABOVE)';
    if (btnResume) btnResume.style.display = 'none';
  } else if (newState === 'INITIALIZING') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> LOCKING PRE-SHOT BASELINE...';
  } else if (newState === 'READY') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> 🟢 READY FOR SHOT (PRE-SHOT LOCKED)';
    if (btnResume) btnResume.style.display = 'none';
  } else if (newState === 'MOTION') {
    pill.className = 'status-pill badge-sim';
    pill.style.background = 'rgba(234, 179, 8, 0.2)';
    pill.style.color = '#ca8a04';
    pill.style.borderColor = '#eab308';
    pill.innerHTML = '<span class="live-pulsing-dot" style="background:#eab308;"></span> 🟡 BALLS IN MOTION (WATCHING SHOT...)';
  } else if (newState === 'SETTLING') {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '⏳ BALLS STOPPING... SETTLING';
  } else if (newState === 'ANALYZING') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> 🔵 ANALYZING SHOT & FOUL RULES...';
  } else if (newState === 'FOUL_HOLD') {
    pill.className = 'status-pill badge-off';
    pill.textContent = '🚨 FOUL DETECTED! LASER AIMING AT MISPLACED BALL';
    if (btnResume) btnResume.style.display = 'inline-block';
  }
}

function handleAutoRefereeStreamLoaded(img) {
  const warn = document.getElementById('autoRefereeCamWarning');
  if (warn) warn.style.display = 'none';
}

function handleAutoRefereeStreamError(img) {
  const warn = document.getElementById('autoRefereeCamWarning');
  if (warn) warn.style.display = 'flex';
}

function reloadAutoRefereeStream() {
  const img = document.getElementById('autoRefereeLiveImg');
  if (img) img.src = '/api/camera/stream?t=' + Date.now();
}

function sampleLiveCameraMotion() {
  if (!isAutoRefereeActive || autoRefereeState === 'ANALYZING' || autoRefereeState === 'FOUL_HOLD') {
    return;
  }

  // Sample from autoRefereeLiveImg (same-origin, never cross-origin tainted)
  const streamImg = document.getElementById('autoRefereeLiveImg') || document.getElementById('liveStreamView');
  if (!streamImg || !streamImg.complete || !streamImg.naturalWidth) {
    const lbl = document.getElementById('lblMotionVal');
    if (lbl && isAutoRefereeActive) {
      lbl.textContent = 'Connecting to camera video stream... (Check IP Webcam)';
    }
    return;
  }

  try {
    motionDetectionCtx.drawImage(streamImg, 0, 0, 160, 90);
    const imgData = motionDetectionCtx.getImageData(0, 0, 160, 90);
    const data = imgData.data;

    if (!lastMotionLuma) {
      lastMotionLuma = new Uint8Array(160 * 90);
      for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        lastMotionLuma[j] = (data[i] * 299 + data[i+1] * 587 + data[i+2] * 114) >> 10;
      }
      return;
    }

    let diffSum = 0;
    const currLuma = new Uint8Array(160 * 90);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const luma = (data[i] * 299 + data[i+1] * 587 + data[i+2] * 114) >> 10;
      currLuma[j] = luma;
      const d = Math.abs(luma - lastMotionLuma[j]);
      if (d > 12) diffSum += d;
    }
    lastMotionLuma = currLuma;

    const motionPercent = Math.min(100.0, Math.round((diffSum / (160 * 90)) * 100) / 10);
    const bar = document.getElementById('barMotion');
    const lbl = document.getElementById('lblMotionVal');
    if (bar) bar.style.width = Math.min(100, motionPercent * 8) + '%';

    // State Transitions
    if (motionPercent >= MOTION_THRESHOLD) {
      // Balls or cue moving
      settleStartTime = null;
      if (autoRefereeState !== 'MOTION') {
        setAutoState('MOTION');
      }
      if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Balls Rolling...`;
    } else {
      // Calm table
      if (autoRefereeState === 'MOTION' || autoRefereeState === 'SETTLING') {
        if (!settleStartTime) {
          settleStartTime = Date.now();
          setAutoState('SETTLING');
        }
        const elapsed = Date.now() - settleStartTime;
        if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Settling: ${(elapsed/1000).toFixed(1)}s / ${(SETTLE_DELAY_MS/1000).toFixed(1)}s`;

        if (elapsed >= SETTLE_DELAY_MS) {
          // Balls have completely stopped! Trigger autonomous foul analysis
          settleStartTime = null;
          triggerAutonomousShotAnalysis();
        }
      } else {
        if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Table at Rest`;
      }
    }
  } catch (err) {
    // Canvas taint or cross-origin fallback
  }
}

async function triggerAutonomousShotAnalysis() {
  setAutoState('ANALYZING');
  try {
    const nominated = document.getElementById('selNominatedBall').value;
    const res = await fetch('/api/foul/auto_process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nominated: nominated })
    });
    const data = await res.json();
    if (!data.success) {
      setAutoState('READY');
      return;
    }

    const rep = data.report;
    renderFoulReport(rep);

    if (rep.is_foul) {
      // 🚨 FOUL DETECTED: Play warning alert tone
      playRefereeTone('foul');
      setAutoState('FOUL_HOLD');
      // Update UI with laser aiming feedback
      if (data.auto_laser_aimed) {
        document.getElementById('cmdFeedback').textContent = `🚨 FOUL: Laser auto-aimed at pre-foul spot for ${data.displaced_ball_aimed} (${data.laser_x}mm, ${data.laser_y}mm)!`;
      }
    } else {
      // ✅ CLEAN SHOT: Play clean chime, post-shot automatically becomes next pre-shot!
      playRefereeTone('clean');
      const box = document.getElementById('foulVerdictBox');
      if (box) {
        box.style.display = 'block';
        box.style.background = 'rgba(34, 197, 94, 0.14)';
        box.style.border = '1px solid rgba(34, 197, 94, 0.4)';
        document.getElementById('foulVerdictIcon').textContent = '✅';
        document.getElementById('foulVerdictTitle').textContent = 'LEGAL SHOT CONFIRMED';
        document.getElementById('foulVerdictTitle').style.color = 'var(--primary-forest)';
        document.getElementById('foulPenaltyBadge').className = 'status-pill tag-online';
        document.getElementById('foulPenaltyBadge').textContent = 'NO FOUL';
        document.getElementById('foulVerdictDesc').textContent = `${rep.description} • Rolled over to next pre-shot automatically!`;
      }
      // Update badge
      updatePreShotBadge(data.post_shot_ball_count);
      setTimeout(() => {
        if (isAutoRefereeActive) {
          setAutoState('READY');
        }
      }, 1500);
    }
  } catch (e) {
    console.error('Auto referee analysis error:', e);
    setAutoState('READY');
  }
}

function resumeAutonomousAfterFoul() {
  // Referee finished repositioning displaced balls: take fresh snapshot and resume auto mode
  captureAndDetect().then(() => {
    capturePreShotState().then(() => {
      setAutoState('READY');
      alert('✅ Table restored! Autonomous referee ready for the next shot.');
    });
  });
}

function playRefereeTone(type) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === 'foul') {
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.35);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } else {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, ctx.currentTime);
      osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.12);
      osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.24);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.45);
      osc.start();
      osc.stop(ctx.currentTime + 0.45);
    }
  } catch (e) {}
}

</script>
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
      Ball Detection
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
    <button id="btnHeaderLaser" class="btn" style="padding:6px 14px; font-size:0.75rem; border-radius:999px; font-weight:700; background:rgba(239,68,68,0.15); color:#ef4444; border:1px solid #ef4444; cursor:pointer;" onclick="toggleLaser()">
      🔴 LASER: OFF
    </button>
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

      <!-- LIVE LASER FINE-TUNING / NUDGE CONTROLS -->
      <div style="background:rgba(45,106,79,0.06); border:1px solid rgba(45,106,79,0.22); border-radius:14px; padding:12px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.8rem; font-weight:700; color:var(--primary-forest);">🎯 Live Laser Fine-Tuning (Nudge Spot)</span>
          <span class="mono lbl-global-trim" style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">Trim: Pan 0.0° | Tilt 0.0°</span>
        </div>
        <div style="display:flex; gap:6px; align-items:center; justify-content:center; flex-wrap:wrap;">
          <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.75rem;" onclick="nudgeTrim('pan', -1)">◀ Left (-1°)</button>
          <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.75rem;" onclick="nudgeTrim('pan', 1)">Right (+1°) ▶</button>
          <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.75rem;" onclick="nudgeTrim('tilt', -1)">▲ Up (-1°)</button>
          <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.75rem;" onclick="nudgeTrim('tilt', 1)">Down (+1°) ▼</button>
          <button class="btn btn-secondary" style="padding:5px 10px; font-size:0.72rem; color:#b91c1c;" onclick="resetTrim()">Reset Trim</button>
        </div>
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
        <div id="cameraViewportContainer" style="width:100%; height:460px; min-height:400px; max-height:65vh; background:#0b1410; border-radius:14px; border:1px solid rgba(45,106,79,0.3); overflow:hidden; position:relative; display:flex; align-items:center; justify-content:center; user-select:none;">
          <img id="liveStreamView" src="" alt="Live Overhead Stream" style="max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:block;" onerror="handleStreamError(this)" onload="handleStreamLoaded(this)">
          <canvas id="cameraCanvas" width="640" height="360" style="display:none; max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; cursor:crosshair; border-radius:6px; box-shadow:0 0 25px rgba(0,0,0,0.4);"></canvas>

          <div id="streamLoadingOverlay" style="position:absolute; display:none; flex-direction:column; align-items:center; justify-content:center; gap:8px; color:#e2e8f0; font-size:0.8rem; background:rgba(11,20,16,0.92); inset:0;">
            <div style="font-size:1.8rem;">📡</div>
            <div id="streamLoadingMsg">Connecting to IP Webcam stream...</div>
            <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 12px;" onclick="reloadLiveStream()">Retry Stream</button>
          </div>

          <div id="cameraModeBadge" style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.65); backdrop-filter:blur(6px); color:#4ade80; border:1px solid rgba(74,222,128,0.4); padding:3px 10px; border-radius:999px; font-size:0.68rem; font-weight:700;">
            ● LIVE STREAM
          </div>

          <div id="canvasCrosshairHud" class="mono" style="position:absolute; bottom:10px; right:12px; background:rgba(11,20,16,0.85); backdrop-filter:blur(4px); color:#22d3ee; border:1px solid rgba(34,211,238,0.3); padding:3px 10px; border-radius:6px; font-size:0.7rem; display:none;">
            Pixel: X: 0, Y: 0
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; flex-wrap:wrap; gap:8px;">
          <div id="cornerStatus" class="mono" style="font-size:0.75rem; color:#b45309; font-weight:600;">
            Status: Live Video streaming. Click "Freeze & Calibrate" to set 4 table corners.
          </div>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.72rem;" onclick="reloadLiveStream()">🔄 Refresh Stream</button>
            <button class="btn btn-secondary" style="padding:5px 12px; font-size:0.72rem;" onclick="undoLastCorner()">↩ Undo Corner</button>
            <button class="btn btn-danger" style="padding:5px 12px; font-size:0.72rem;" onclick="resetCorners()">Reset 4 Corners</button>
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
          <input type="text" id="inpIpWebcam" value="http://10.21.46.181:8080" style="flex:1;" placeholder="http://10.21.46.181:8080">
          <button class="btn btn-forest" style="font-size:0.74rem; white-space:nowrap; padding:9px 13px;" onclick="saveIpWebcam()">Update URL</button>
          <button class="btn btn-secondary" style="font-size:0.74rem; white-space:nowrap; padding:9px 13px;" onclick="setWebcamPreset('http://10.21.46.181:8080')">Hotspot Phone (10.21.46.181:8080)</button>
          <button class="btn btn-secondary" style="font-size:0.74rem; white-space:nowrap; padding:9px 13px;" onclick="autoScanPhoneIP()">🔍 Scan</button>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:6px;">
          <button class="btn btn-forest" style="flex:1; padding:10px;" onclick="freezeFrameForCalibration()">📸 Freeze Frame for Homography</button>
          <button class="btn btn-secondary" style="flex:1; padding:10px;" onclick="captureAndDetect()">🎱 Detect Balls from Live Feed</button>
        </div>
        <div id="ipWebcamStatus" style="font-size:0.74rem; color:var(--text-muted); margin-top:6px; line-height:1.45;"></div>
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
          <input type="text" id="inpEspIp" value="http://10.21.46.190" style="flex:1;">
          <button class="btn btn-forest" onclick="saveEspIp()">Save</button>
          <button class="btn btn-secondary" onclick="pingEsp()">Ping</button>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
          <button class="btn btn-forest" style="font-size:0.74rem; flex:1;" onclick="setEspPreset('http://10.21.46.190')">Hotspot ESP32 (10.21.46.190)</button>
          <button class="btn btn-secondary" style="font-size:0.74rem; flex:1;" onclick="setEspPreset('http://192.168.31.2')">Home Wi-Fi (192.168.31.2)</button>
        </div>
      </div>

      <div class="form-group">
        <label>
          <span>Pan Servo (GPIO 18)</span>
          <span id="valPan" class="mono" style="color:var(--primary-green); font-weight:700;">10°</span>
        </label>
        <div style="display:flex; align-items:center; gap:8px;">
          <button type="button" class="btn btn-secondary" style="padding:4px 10px; font-weight:700; min-width:34px;" onclick="stepServo('pan', -1)" title="Decrease Pan by 1°">&minus;</button>
          <input type="range" id="rngPan" min="10" max="170" step="0.5" value="10" style="flex:1;" oninput="previewServo('pan', this.value)" onchange="sendServoAngle('pan', this.value)">
          <button type="button" class="btn btn-secondary" style="padding:4px 10px; font-weight:700; min-width:34px;" onclick="stepServo('pan', 1)" title="Increase Pan by 1°">+</button>
          <div style="display:flex; align-items:center; gap:2px;">
            <input type="number" id="inpPan" class="mono" min="10" max="170" step="0.5" value="10" style="width:68px; text-align:center; padding:4px 6px;" onchange="applyManualServo('pan', this.value)" onkeydown="if(event.key==='Enter'){applyManualServo('pan', this.value);}">
            <span class="mono" style="font-size:0.82rem; color:var(--text-muted);">°</span>
          </div>
        </div>
      </div>

      <div class="form-group">
        <label>
          <span>Tilt Servo (GPIO 19)</span>
          <span id="valTilt" class="mono" style="color:var(--primary-green); font-weight:700;">150°</span>
        </label>
        <div style="display:flex; align-items:center; gap:8px;">
          <button type="button" class="btn btn-secondary" style="padding:4px 10px; font-weight:700; min-width:34px;" onclick="stepServo('tilt', -1)" title="Decrease Tilt by 1°">&minus;</button>
          <input type="range" id="rngTilt" min="60" max="160" step="0.5" value="150" style="flex:1;" oninput="previewServo('tilt', this.value)" onchange="sendServoAngle('tilt', this.value)">
          <button type="button" class="btn btn-secondary" style="padding:4px 10px; font-weight:700; min-width:34px;" onclick="stepServo('tilt', 1)" title="Increase Tilt by 1°">+</button>
          <div style="display:flex; align-items:center; gap:2px;">
            <input type="number" id="inpTilt" class="mono" min="60" max="160" step="0.5" value="150" style="width:68px; text-align:center; padding:4px 6px;" onchange="applyManualServo('tilt', this.value)" onkeydown="if(event.key==='Enter'){applyManualServo('tilt', this.value);}">
            <span class="mono" style="font-size:0.82rem; color:var(--text-muted);">°</span>
          </div>
        </div>
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
    
      <!-- QUICK LASER FINE-TUNING / TRIM -->
      <div style="background:rgba(45,106,79,0.08); border:1px solid rgba(45,106,79,0.25); border-radius:16px; padding:14px; margin-top:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--primary-forest);">🎯 Physical Laser Fine-Tuning (Nudge Spot)</span>
          <span id="trimBadge" class="mono" style="font-size:0.72rem; color:var(--primary-forest); font-weight:700;">Pan: +0.0° | Tilt: +0.0°</span>
        </div>
        <p style="font-size:0.73rem; color:var(--text-muted); margin-bottom:10px;">
          Nudge the physical laser beam in real-time to eliminate physical horn/tilt mounting offsets:
        </p>
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-bottom:8px;">
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('pan', -1.0)">◀ Pan Left (-1.0°)</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('pan', +1.0)">Pan Right (+1.0°) ▶</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('tilt', -1.0)">▲ Tilt Up (-1.0°)</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('tilt', +1.0)">Tilt Down (+1.0°) ▼</button>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="nudgeTrim('pan', -0.2)">◀ Micro Pan (-0.2°)</button>
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="nudgeTrim('pan', +0.2)">Micro Pan (+0.2°) ▶</button>
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="resetTrim()">🔄 Reset Trim (0°)</button>
        </div>
      </div>
    </div>
  
      <!-- QUICK LASER FINE-TUNING / TRIM -->
      <div style="background:rgba(45,106,79,0.08); border:1px solid rgba(45,106,79,0.25); border-radius:16px; padding:14px; margin-top:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--primary-forest);">🎯 Physical Laser Fine-Tuning (Nudge Spot)</span>
          <span id="trimBadge" class="mono" style="font-size:0.72rem; color:var(--primary-forest); font-weight:700;">Pan: +0.0° | Tilt: +0.0°</span>
        </div>
        <p style="font-size:0.73rem; color:var(--text-muted); margin-bottom:10px;">
          Nudge the physical laser beam in real-time to eliminate physical horn/tilt mounting offsets:
        </p>
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-bottom:8px;">
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('pan', -1.0)">◀ Pan Left (-1.0°)</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('pan', +1.0)">Pan Right (+1.0°) ▶</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('tilt', -1.0)">▲ Tilt Up (-1.0°)</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('tilt', +1.0)">Tilt Down (+1.0°) ▼</button>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="nudgeTrim('pan', -0.2)">◀ Micro Pan (-0.2°)</button>
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="nudgeTrim('pan', +0.2)">Micro Pan (+0.2°) ▶</button>
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="resetTrim()">🔄 Reset Trim (0°)</button>
        </div>
      </div>
    </div>

      <!-- QUICK LASER FINE-TUNING / TRIM -->
      <div style="background:rgba(45,106,79,0.08); border:1px solid rgba(45,106,79,0.25); border-radius:16px; padding:14px; margin-top:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--primary-forest);">🎯 Physical Laser Fine-Tuning (Nudge Spot)</span>
          <span id="trimBadge" class="mono" style="font-size:0.72rem; color:var(--primary-forest); font-weight:700;">Pan: +0.0° | Tilt: +0.0°</span>
        </div>
        <p style="font-size:0.73rem; color:var(--text-muted); margin-bottom:10px;">
          Nudge the physical laser beam in real-time to eliminate physical horn/tilt mounting offsets:
        </p>
        <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-bottom:8px;">
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('pan', -1.0)">◀ Pan Left (-1.0°)</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('pan', +1.0)">Pan Right (+1.0°) ▶</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('tilt', -1.0)">▲ Tilt Up (-1.0°)</button>
          <button class="btn btn-secondary" style="font-size:0.76rem; padding:8px 6px;" onclick="nudgeTrim('tilt', +1.0)">Tilt Down (+1.0°) ▼</button>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="nudgeTrim('pan', -0.2)">◀ Micro Pan (-0.2°)</button>
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="nudgeTrim('pan', +0.2)">Micro Pan (+0.2°) ▶</button>
          <button class="btn btn-secondary" style="flex:1; font-size:0.72rem; padding:6px 10px;" onclick="resetTrim()">🔄 Reset Trim (0°)</button>
        </div>
      </div>
    </div>

    <!-- TAB 4: BALL DETECTION & UPLOAD -->
    <div id="tab-balls" class="tab-pane">
      <div class="panel-header" style="margin-bottom:10px;">
        <span style="display:flex; align-items:center; gap:8px;">
          🎱 AI Ball Detection & Image Upload
          <span class="tag-pill tag-online" style="font-size:0.65rem;">YOLOv8s • 1024px</span>
        </span>
        <button class="btn btn-forest" style="padding:6px 14px; font-size:0.77rem;" onclick="syncDetectedBallsTo3D()">
          🔄 Sync to 3D Table
        </button>
      </div>

      <!-- UPLOAD & ACTION BAR -->
      <div style="background:rgba(255,255,255,0.92); border:1px solid rgba(45,106,79,0.25); border-radius:16px; padding:14px; margin-bottom:12px; box-shadow:0 4px 14px rgba(0,0,0,0.03);">
        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
          <input type="file" id="inpDetectUpload" accept="image/*" onchange="onDetectFileUpload(this)" style="display:none;">
          <button class="btn btn-forest" style="flex:1; min-width:160px; padding:10px 14px;" onclick="document.getElementById('inpDetectUpload').click()">
            📁 Upload Table Image
          </button>
          <button class="btn btn-secondary" style="flex:1; min-width:160px; padding:10px 14px;" onclick="detectFromLiveCamera()">
            📸 Use Current Camera Feed
          </button>
        </div>

        <!-- Sample Images Bar -->
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:0.75rem; color:var(--text-muted);">
          <span style="font-weight:600; color:var(--text-dark);">Quick Test Samples:</span>
          <button class="btn btn-secondary" style="padding:4px 10px; font-size:0.72rem;" onclick="loadDetectSample('photo42')">Sample Rack (photo42)</button>
          <button class="btn btn-secondary" style="padding:4px 10px; font-size:0.72rem;" onclick="loadDetectSample('photo34')">Colors Spread (photo34)</button>
          <button class="btn btn-secondary" style="padding:4px 10px; font-size:0.72rem;" onclick="loadDetectSample('photo4')">Full Set (photo4)</button>
        </div>
      </div>

      <!-- DETECTION PREVIEW & ANNOTATED VIEWPORT -->
      <div id="detectViewerContainer" style="background:#0b1410; border-radius:14px; border:1px solid rgba(45,106,79,0.3); overflow:hidden; position:relative; min-height:260px; max-height:440px; display:flex; align-items:center; justify-content:center; margin-bottom:12px;">
        <img id="detectAnnotatedImg" src="" alt="Detection Output" style="max-width:100%; max-height:420px; width:auto; height:auto; object-fit:contain; display:none;">
        <div id="detectPlaceholder" style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:32px;">
          <div style="font-size:2.4rem; margin-bottom:8px;">🎱</div>
          <div style="font-weight:600; color:#e2e8f0; margin-bottom:4px;">Upload an Image or Choose a Sample</div>
          <div>YOLOv8s will detect all colored balls with color-coded bounding boxes</div>
        </div>
        <div id="detectSpinner" style="display:none; position:absolute; inset:0; background:rgba(11,20,16,0.88); flex-direction:column; align-items:center; justify-content:center; gap:10px; color:#4ade80; font-size:0.85rem;">
          <div class="live-pulsing-dot" style="width:14px; height:14px;"></div>
          <span id="detectSpinnerText">Running YOLOv8s Detection on GPU...</span>
        </div>
      </div>

      <!-- DETECTION STATS & SUMMARY PILLS -->
      <div id="detectSummaryBox" style="display:none; background:rgba(255,255,255,0.88); border:1px solid rgba(45,106,79,0.2); border-radius:14px; padding:10px 14px; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--primary-forest);" id="detectTotalCount">Detected Balls (0)</span>
          <span class="mono" style="font-size:0.72rem; color:var(--text-muted);" id="detectLatency">Latency: 0 ms</span>
        </div>
        <div id="detectClassChips" style="display:flex; gap:6px; flex-wrap:wrap;"></div>
      </div>

            <!-- QUICK LASER FINE-TUNING FOR DETECTED BALLS -->
      <div style="background:rgba(45,106,79,0.08); border:1px solid rgba(45,106,79,0.22); border-radius:14px; padding:10px 14px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:0.78rem; font-weight:700; color:var(--primary-forest);">🎯 Nudge Physical Laser:</span>
          <span id="ballsTrimBadge" class="mono" style="font-size:0.72rem; color:var(--primary-green); font-weight:700;">Pan: +0.0° | Tilt: +0.0°</span>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
          <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 10px;" onclick="nudgeTrim('pan', -1.0)">◀ Left (-1°)</button>
          <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 10px;" onclick="nudgeTrim('pan', +1.0)">Right (+1°) ▶</button>
          <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 10px;" onclick="nudgeTrim('tilt', -1.0)">▲ Up (-1°)</button>
          <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 10px;" onclick="nudgeTrim('tilt', +1.0)">Down (+1°) ▼</button>
          <button class="btn btn-secondary" style="font-size:0.72rem; padding:4px 8px;" onclick="resetTrim()">Reset</button>
        </div>
      </div>

      <!-- DETECTED BALLS TABLE -->
      <div class="table-scroll" style="max-height:220px;">
        <table>
          <thead>
            <tr><th>#</th><th>Class</th><th>Conf</th><th>Pixel (X, Y)</th><th>Table (mm)</th><th>Action</th></tr>
          </thead>
          <tbody id="ballsTableBody">
            <tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:16px;">No balls detected. Upload an image above to start.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 5: AI FOUL DETECTION & AUTONOMOUS REFEREE SUITE -->
    <div id="tab-foul" class="tab-pane">
      <div class="panel-header" style="flex-wrap:wrap; gap:8px;">
        <span style="display:flex; align-items:center; gap:8px;">
          ⚖️ AI Snooker Foul Detection & Referee
          <span id="refereeModeTag" class="tag-pill tag-online" style="font-size:0.65rem;">LIVE MOTION AI</span>
        </span>
        <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
          <button id="btnModeManual" class="btn btn-secondary" style="font-size:0.74rem; font-weight:700; padding:6px 12px; border-radius:999px;" onclick="setRefereeMode('manual')">
            📸 Manual 2-Shot Mode
          </button>
          <button id="btnToggleAutoReferee" class="btn btn-secondary" style="font-size:0.74rem; font-weight:800; padding:6px 14px; border-radius:999px; border-color:var(--primary-green);" onclick="toggleAutonomousReferee()">
            🟢 AUTONOMOUS REFEREE: OFF
          </button>
        </div>
      </div>

      <!-- MANUAL 2-SHOT REFEREE PANEL -->
      <div id="manualRefereePanel" style="display:none; background:rgba(255,255,255,0.94); border:1px solid rgba(45,106,79,0.28); border-radius:18px; padding:16px; margin-bottom:14px; box-shadow:0 4px 16px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
          <span id="manualRefereePill" class="status-pill badge-sim" style="font-size:0.74rem; padding:5px 12px; font-weight:800;">
            📸 STEP 1: TAKE PRE-SHOT IMAGE BEFORE PLAY
          </span>
          <div style="display:flex; align-items:center; gap:8px;">
            <label style="font-size:0.78rem; font-weight:700; color:var(--primary-forest);">Target On:</label>
            <select id="selManualNominatedBall" onchange="document.getElementById('selNominatedBall').value=this.value" style="padding:5px 10px; font-size:0.75rem; border-radius:8px; border:1px solid rgba(45,106,79,0.3); font-weight:700; font-family:'JetBrains Mono',monospace;">
              <option value="orange" selected>🟠 Orange Ball (1 pt)</option>
              <option value="yellow">🟡 Yellow Ball (2 pts)</option>
              <option value="green">🟢 Green Ball (3 pts)</option>
              <option value="blue">🔵 Blue Ball (5 pts)</option>
              <option value="pink">🌸 Pink Ball (6 pts)</option>
            </select>
          </div>
        </div>

        <p style="font-size:0.75rem; color:var(--text-muted); margin-bottom:10px;">
          <strong>Manual 2-Shot Workflow:</strong> Take a <strong>Pre-Shot</strong> photo before the stroke. If a foul occurs, take the <strong>Post-Shot</strong> photo to compare and automatically point the laser at displaced balls for restoration.
        </p>

        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
          <button id="btnManualPreShot" class="btn btn-forest" style="flex:1; min-width:170px; padding:9px 12px; font-size:0.76rem; font-weight:700;" onclick="captureManualPreShot()">
            📸 1. Take Pre-Shot Image
          </button>
          <button id="btnManualPostShot" class="btn btn-terracotta" style="flex:1; min-width:170px; padding:9px 12px; font-size:0.76rem; font-weight:700;" onclick="captureManualPostShotAndCompare()">
            📷 2. Take Post-Shot &amp; Compare
          </button>
          <button class="btn btn-secondary" style="padding:9px 12px; font-size:0.74rem;" onclick="resetManualRefereeRound()">
            🔄 Reset
          </button>
        </div>

        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
          <input type="file" id="inpManualPreUpload" accept="image/*" style="display:none;" onchange="uploadManualPreShot(this)">
          <input type="file" id="inpManualPostUpload" accept="image/*" style="display:none;" onchange="uploadManualPostShotAndCompare(this)">
          <button class="btn btn-secondary" style="flex:1; min-width:170px; padding:7px 12px; font-size:0.73rem; font-weight:700;" onclick="document.getElementById('inpManualPreUpload').click()">
            📁 Upload Pre-Shot Image
          </button>
          <button class="btn btn-secondary" style="flex:1; min-width:170px; padding:7px 12px; font-size:0.73rem; font-weight:700;" onclick="document.getElementById('inpManualPostUpload').click()">
            📁 Upload Post-Shot &amp; Compare
          </button>
        </div>

        <!-- Side-by-Side Pre-Shot (with Interactive 4-Corner Homography) vs Post-Shot Preview -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px;">
          <div style="background:#0b1410; border-radius:12px; overflow:hidden; border:1px solid rgba(45,106,79,0.3); position:relative; height:160px; display:flex; align-items:center; justify-content:center;">
            <canvas id="cvsManualPreShot" width="640" height="360" onclick="onManualPreShotCanvasClick(event)" style="display:none; width:100%; height:100%; object-fit:contain; cursor:crosshair;"></canvas>
            <span id="lblManualPrePlaceholder" style="color:#94a3b8; font-size:0.72rem; text-align:center; padding:8px;">Pre-Shot Image (Not Captured)</span>
            <span style="position:absolute; top:6px; left:6px; background:rgba(0,0,0,0.75); color:#4ade80; padding:2px 8px; border-radius:999px; font-size:0.64rem; font-weight:700; pointer-events:none;">PRE-SHOT + HOMOGRAPHY</span>
          </div>
          <div style="background:#0b1410; border-radius:12px; overflow:hidden; border:1px solid rgba(45,106,79,0.3); position:relative; height:160px; display:flex; align-items:center; justify-content:center;">
            <img id="imgManualPostShot" alt="Post-Shot" style="display:none; width:100%; height:100%; object-fit:contain;">
            <span id="lblManualPostPlaceholder" style="color:#94a3b8; font-size:0.72rem; text-align:center; padding:8px;">Post-Shot Image (If Foul Occurs)</span>
            <span style="position:absolute; top:6px; left:6px; background:rgba(0,0,0,0.75); color:#fbbf24; padding:2px 8px; border-radius:999px; font-size:0.64rem; font-weight:700; pointer-events:none;">POST-SHOT</span>
          </div>
        </div>

        <!-- Integrated Homography Status & Corner Calibration Bar -->
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; background:rgba(45,106,79,0.07); border:1px solid rgba(45,106,79,0.2); border-radius:12px; padding:8px 12px; margin-bottom:10px;">
          <span id="lblManualHomogStatus" class="mono" style="font-size:0.72rem; font-weight:700; color:var(--primary-forest);">
            📐 Homography: Waiting for Pre-Shot
          </span>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:4px 10px;" onclick="startManualCornerPick()">🎯 Pick 4 Corners (TR→TL→BL→BR)</button>
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:4px 10px;" onclick="autoCalibrateManualPreShot()">🤖 Auto-Calibrate</button>
          </div>
        </div>

        <!-- Sequential Laser Restoration Stepper -->
        <div id="manualRestoreBar" style="display:none; background:rgba(45,106,79,0.1); border:1px solid rgba(45,106,79,0.3); border-radius:12px; padding:10px 12px; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
          <span id="lblManualRestoreStatus" class="mono" style="font-size:0.74rem; font-weight:700; color:var(--primary-forest);">
            🎯 Pointing Ball 1 of 1
          </span>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-forest" style="font-size:0.72rem; padding:5px 10px;" onclick="pointNextDisplacedBall()">
              ➡️ Point Next Displaced Ball
            </button>
          </div>
        </div>
      </div>

      <!-- AUTONOMOUS LIVE STATUS CARD -->
      <div id="autoRefereePanel" style="background:rgba(255,255,255,0.92); border:1px solid rgba(45,106,79,0.25); border-radius:18px; padding:16px; margin-bottom:14px; box-shadow:0 4px 16px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div style="display:flex; align-items:center; gap:10px;">
            <span id="autoRefereePill" class="status-pill badge-sim" style="font-size:0.75rem; padding:5px 14px; font-weight:800;">
              ⚪ REFEREE STANDBY (TAP ENABLE ABOVE)
            </span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <label style="font-size:0.78rem; font-weight:700; color:var(--primary-forest);">Target On:</label>
            <select id="selNominatedBall" style="padding:5px 12px; font-size:0.76rem; border-radius:8px; border:1px solid rgba(45,106,79,0.3); font-weight:700; font-family:'JetBrains Mono',monospace;">
              <option value="orange" selected>🟠 Orange Ball (1 pt)</option>
              <option value="yellow">🟡 Yellow Ball (2 pts)</option>
              <option value="green">🟢 Green Ball (3 pts)</option>
              <option value="blue">🔵 Blue Ball (5 pts)</option>
              <option value="pink">🌸 Pink Ball (6 pts)</option>
            </select>
          </div>
        </div>

        <!-- LIVE OVERHEAD VIDEO MONITOR ON TAB 5 -->
        <div style="position:relative; width:100%; height:180px; max-height:200px; background:#0b1410; border-radius:14px; overflow:hidden; margin-bottom:12px; border:1px solid rgba(45,106,79,0.3); display:flex; align-items:center; justify-content:center;">
          <img id="autoRefereeLiveImg" src="/api/camera/stream" crossorigin="anonymous" alt="Live Table Overhead Feed" style="width:100%; height:100%; object-fit:contain; display:block;" onerror="handleAutoRefereeStreamError(this)" onload="handleAutoRefereeStreamLoaded(this)">
          <div id="autoRefereeOverlayBadge" style="position:absolute; top:8px; left:10px; background:rgba(0,0,0,0.65); backdrop-filter:blur(4px); color:#4ade80; border:1px solid rgba(74,222,128,0.4); padding:3px 10px; border-radius:999px; font-size:0.68rem; font-weight:700;">
            ● LIVE OVERHEAD MONITOR
          </div>
          <div id="autoRefereeCamWarning" style="display:none; position:absolute; inset:0; background:rgba(11,20,16,0.92); flex-direction:column; align-items:center; justify-content:center; gap:6px; color:#f87171; font-size:0.75rem; padding:12px; text-align:center;">
            <div style="font-size:1.6rem;">📡</div>
            <div style="font-weight:700;">Camera Feed Not Connected</div>
            <div style="font-size:0.7rem; color:#94a3b8;">Ensure phone screen is ON and IP Webcam says 'Server running'</div>
            <button class="btn btn-secondary" style="font-size:0.68rem; padding:3px 10px; margin-top:4px;" onclick="reloadAutoRefereeStream()">🔄 Retry Feed</button>
          </div>
        </div>

        <!-- Real-Time Motion Sensor Progress Bar -->
        <div style="background:rgba(45,106,79,0.08); border-radius:12px; padding:10px 14px; margin-bottom:12px;">
          <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.72rem; margin-bottom:6px;">
            <span style="font-weight:700; color:var(--primary-forest);">📹 Live Video Motion Sensor:</span>
            <span id="lblMotionVal" class="mono" style="font-weight:700; color:var(--text-dark);">Motion: 0.0% | Status: Table at Rest</span>
          </div>
          <div style="width:100%; height:8px; background:rgba(0,0,0,0.08); border-radius:999px; overflow:hidden;">
            <div id="barMotion" style="width:0%; height:100%; background:linear-gradient(90deg, #22c55e, #eab308, #ef4444); border-radius:999px; transition:width 0.1s linear;"></div>
          </div>
        </div>

        <!-- Autonomous Workflow Controls & Resume Button -->
        <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
          <button class="btn btn-secondary" style="flex:1; min-width:140px; padding:8px 12px; font-size:0.74rem;" onclick="capturePreShotState()">
            📸 Lock Pre-Shot Now
          </button>
          <button class="btn btn-secondary" style="flex:1; min-width:140px; padding:8px 12px; font-size:0.74rem;" onclick="analyzeShotForFoul()">
            ⚡ Manual Trigger Analysis
          </button>
          <button id="btnResumeAutoAfterFoul" class="btn btn-forest" style="display:none; flex:2; min-width:200px; padding:8px 14px; font-size:0.76rem; font-weight:800;" onclick="resumeAutonomousAfterFoul()">
            ✅ Balls Restored / Ready for Next Shot
          </button>
        </div>
      </div>

      <!-- CARD 2: DYNAMIC FOUL VERDICT BANNER -->
      <div id="foulVerdictBox" style="display:none; border-radius:16px; padding:16px; margin-bottom:14px; transition:all 0.3s ease;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span id="foulVerdictIcon" style="font-size:1.4rem;">🚨</span>
            <span id="foulVerdictTitle" style="font-size:0.95rem; font-weight:800; letter-spacing:-0.01em;">FOUL DETECTED</span>
          </div>
          <span id="foulPenaltyBadge" class="status-pill" style="font-size:0.75rem; padding:4px 12px; font-weight:800;">PENALTY: 4 PTS</span>
        </div>
        <p id="foulVerdictDesc" style="font-size:0.82rem; margin:0; line-height:1.4;"></p>
      </div>

      <!-- CARD 3: DISPLACED BALLS REPOSITIONING (REFEREE ASSISTANT) -->
      <div id="foulDisplacedContainer" style="background:rgba(255,255,255,0.88); border:1px solid rgba(45,106,79,0.22); border-radius:18px; padding:14px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
          <span style="font-size:0.82rem; font-weight:700; color:var(--primary-forest);">🎯 Referee Laser Repositioning (Displaced Balls)</span>
          <span id="displacedCountBadge" class="mono" style="font-size:0.72rem; color:var(--text-muted);">0 balls displaced</span>
        </div>

        <div class="table-scroll" style="max-height:220px; margin-bottom:10px;">
          <table>
            <thead>
              <tr>
                <th>Ball</th>
                <th>Status</th>
                <th>Pre-Foul Spot (X, Y)</th>
                <th>Post-Shot Spot (X, Y)</th>
                <th>Displacement</th>
                <th>Laser Action</th>
              </tr>
            </thead>
            <tbody id="foulDisplacedTableBody">
              <tr>
                <td colspan="6" style="text-align:center; color:var(--text-muted); padding:16px;">
                  No foul or displacement recorded yet. Autonomous referee will track every shot automatically.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- QUICK LASER NUDGE FOR FOUL PLACEMENT -->
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; background:rgba(45,106,79,0.06); padding:8px 12px; border-radius:12px;">
          <span style="font-size:0.74rem; font-weight:700; color:var(--primary-forest);">Fine-Tune Laser Spot:</span>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:3px 8px;" onclick="nudgeTrim('pan', -1.0)">◀ Left (-1°)</button>
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:3px 8px;" onclick="nudgeTrim('pan', +1.0)">Right (+1°) ▶</button>
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:3px 8px;" onclick="nudgeTrim('tilt', -1.0)">▲ Up (-1°)</button>
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:3px 8px;" onclick="nudgeTrim('tilt', +1.0)">Down (+1°) ▼</button>
            <button class="btn btn-secondary" style="font-size:0.7rem; padding:3px 8px;" onclick="resetTrim()">Reset</button>
          </div>
        </div>
      </div>

      <!-- CARD 4: REGULATION SNOOKER SPOTS RESPOSTING -->
      <div style="background:rgba(255,255,255,0.75); border:1px solid rgba(45,106,79,0.2); border-radius:16px; padding:14px;">
        <div style="font-size:0.8rem; font-weight:700; color:var(--primary-forest); margin-bottom:8px;">
          📍 Regulation Snooker Re-Spotting (Quick Aim)
        </div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:6px;">
          <button class="btn btn-secondary" style="font-size:0.73rem; padding:8px 6px;" onclick="aimAtRegulationSpot('pink')">🌸 Pink Spot (1756mm)</button>
          <button class="btn btn-secondary" style="font-size:0.73rem; padding:8px 6px;" onclick="aimAtRegulationSpot('blue')">🔵 Blue Spot (1171mm)</button>
          <button class="btn btn-secondary" style="font-size:0.73rem; padding:8px 6px;" onclick="aimAtRegulationSpot('green')">🟢 Green Spot (Baulk L)</button>
          <button class="btn btn-secondary" style="font-size:0.73rem; padding:8px 6px;" onclick="aimAtRegulationSpot('yellow')">🟡 Yellow Spot (Baulk R)</button>
          <button class="btn btn-secondary" style="font-size:0.73rem; padding:8px 6px;" onclick="aimAtRegulationSpot('cue')">⚪ Cue Ball ('D' Center)</button>
        </div>
      </div>
    </div>

    <!-- TAB 6: 9-POINT CALIBRATION & HARDWARE TUNING WIZARD -->
    <div id="tab-calib" class="tab-pane">
      <div class="panel-header">
        <span>9-Point Empirical Calibration Wizard</span>
        <button class="btn btn-secondary" style="padding:4px 12px; font-size:0.72rem;" onclick="fetchNodes()">🔄 Refresh Nodes</button>
      </div>

      <!-- LIVE NODE CALIBRATOR CARD -->
      <div style="background:rgba(255,255,255,0.92); border:1px solid rgba(45,106,79,0.25); border-radius:18px; padding:16px; margin-bottom:16px; box-shadow:0 4px 16px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div>
            <div id="lblCalibNodeTitle" style="font-size:0.88rem; font-weight:800; color:var(--primary-forest);">Tuning: Centre (C) (584.2mm, 1170.9mm)</div>
            <div style="font-size:0.72rem; color:var(--text-muted);">Aim laser at each pocket/grid node, tune angles to hit the felt mark, then click Save.</div>
          </div>
          <div style="display:flex; gap:6px; align-items:center;">
            <label style="font-size:0.74rem; font-weight:700; color:var(--primary-green);">Select Node:</label>
            <select id="selCalibNode" style="padding:5px 10px; font-size:0.75rem; border-radius:8px; border:1px solid rgba(45,106,79,0.3); background:#fff; font-family:'JetBrains Mono',monospace;" onchange="onSelectNodeDropdown(this.value)">
            </select>
          </div>
        </div>

        <!-- Pan Tuning Slider -->
        <div class="form-group" style="margin-bottom:12px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <label style="font-size:0.76rem; font-weight:700; color:var(--primary-forest);">Pan Angle (GPIO 18):</label>
            <div style="display:flex; align-items:center; gap:8px;">
              <span id="valCalibPan" class="mono" style="font-size:0.85rem; font-weight:800; color:var(--primary-green);">35.0°</span>
              <div style="display:flex; gap:4px;">
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('pan', -1.0)">-1°</button>
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('pan', -0.2)">-0.2°</button>
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('pan', +0.2)">+0.2°</button>
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('pan', +1.0)">+1°</button>
              </div>
            </div>
          </div>
          <input type="range" id="rngCalibPan" min="10" max="170" step="0.1" value="35" oninput="onTuneSlider('pan', this.value)">
        </div>

        <!-- Tilt Tuning Slider -->
        <div class="form-group" style="margin-bottom:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <label style="font-size:0.76rem; font-weight:700; color:var(--primary-forest);">Tilt Angle (GPIO 19):</label>
            <div style="display:flex; align-items:center; gap:8px;">
              <span id="valCalibTilt" class="mono" style="font-size:0.85rem; font-weight:800; color:var(--primary-green);">102.0°</span>
              <div style="display:flex; gap:4px;">
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('tilt', -1.0)">-1°</button>
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('tilt', -0.2)">-0.2°</button>
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('tilt', +0.2)">+0.2°</button>
                <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.7rem;" onclick="nudgeTune('tilt', +1.0)">+1°</button>
              </div>
            </div>
          </div>
          <input type="range" id="rngCalibTilt" min="60" max="160" step="0.1" value="102" oninput="onTuneSlider('tilt', this.value)">
        </div>

        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button class="btn btn-forest" style="flex:2; padding:10px 16px; font-weight:700;" onclick="saveCurrentNodeCalib()">
            💾 Save Calibrated Angle to Database
          </button>
          <button class="btn btn-secondary" style="flex:1; padding:10px 14px;" onclick="reAimSelectedNode()">
            🎯 Re-Aim Node
          </button>
        </div>
      </div>

      <!-- 9-POINT DATABASE TABLE -->
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Label</th>
              <th>X (mm)</th>
              <th>Y (mm)</th>
              <th>Pan</th>
              <th>Tilt</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
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
// Global ball detection caches
let lastDetectedBalls = [];
let detected_balls_cache = [];

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
  "white": 0xfcfbfa,
  "orange": 0xee1c00,
  "red": 0xee1c00,
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
    { type: "white", color: SNOOKER_COLORS.white, x: 584.2, y: 1800.0 },
    { type: "pink", color: SNOOKER_COLORS.pink, x: 584.2, y: 1756.4 },
    { type: "blue", color: SNOOKER_COLORS.blue, x: 584.2, y: 1170.94 },
    { type: "green", color: SNOOKER_COLORS.green, x: 876.2, y: 737.0 },
    { type: "yellow", color: SNOOKER_COLORS.yellow, x: 292.2, y: 737.0 },
    { type: "orange", color: SNOOKER_COLORS.orange, x: 584.2, y: 1600.0 },
    { type: "orange", color: SNOOKER_COLORS.orange, x: 556.0, y: 1645.0 },
    { type: "orange", color: SNOOKER_COLORS.orange, x: 612.4, y: 1645.0 },
    { type: "orange", color: SNOOKER_COLORS.orange, x: 528.0, y: 1690.0 },
    { type: "orange", color: SNOOKER_COLORS.orange, x: 584.2, y: 1690.0 },
    { type: "orange", color: SNOOKER_COLORS.orange, x: 640.4, y: 1690.0 }
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
  setInterval(pollStatus, 4000);
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

async function aimAtCoordinates(x, y) {
  setPreset(x, y);
  await updatePred();
  await sendAimCommand();
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
  if (data.laser !== undefined) {
    updateLaserUi(data.laser);
  }
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
  try {
    const res = await fetch('/api/nodes');
    const data = await res.json();
    currentCalibNodes = data.nodes || [];
    const tb = document.getElementById('calibTableBody');
    if (tb) {
      tb.innerHTML = '';
      currentCalibNodes.forEach(n => {
        const isCalib = n.status === 'calibrated';
        const statusBadge = isCalib
          ? '<span class="tag-pill tag-online" style="font-size:0.65rem;">CALIBRATED</span>'
          : '<span class="tag-pill" style="font-size:0.65rem;">DEFAULT</span>';
        tb.innerHTML += `<tr>
          <td><strong>${n.id}</strong></td>
          <td>${n.label}</td>
          <td>${n.x_mm}</td>
          <td>${n.y_mm}</td>
          <td><strong style="color:var(--primary-green);">${n.pan_deg}°</strong></td>
          <td><strong style="color:var(--primary-green);">${n.tilt_deg}°</strong></td>
          <td>${statusBadge}</td>
          <td>
            <button class="btn btn-secondary" style="padding:3px 10px; font-size:0.7rem; border-color:var(--primary-green);" onclick="selectNodeForCalib('${n.id}')">
              🎯 Aim & Tune
            </button>
          </td>
        </tr>`;
      });
    }
    populateCalibNodeDropdown();
  } catch (e) {
    console.error('fetchNodes error:', e);
  }
}

// ==============================================================
// 10. LIVE FINE-TUNING & 9-POINT CALIBRATION LOGIC
// ==============================================================
let panTrim = 0.0, tiltTrim = 0.0;
let currentCalibNodes = [];
let selectedCalibNodeId = 'node_c';

async function nudgeTrim(axis, delta) {
  if (axis === 'pan') panTrim = Math.round((panTrim + delta) * 10) / 10;
  if (axis === 'tilt') tiltTrim = Math.round((tiltTrim + delta) * 10) / 10;
  updateTrimBadges();
  try {
    await fetch('/api/trim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pan_trim: panTrim, tilt_trim: tiltTrim })
    });
    await sendAimCommand();
  } catch (e) {
    console.error('Trim error:', e);
  }
}

async function resetTrim() {
  panTrim = 0.0;
  tiltTrim = 0.0;
  updateTrimBadges();
  try {
    await fetch('/api/trim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pan_trim: 0.0, tilt_trim: 0.0 })
    });
    await sendAimCommand();
  } catch (e) {
    console.error('Reset trim error:', e);
  }
}

function updateTrimBadges() {
  const text = `Pan: ${panTrim >= 0 ? '+' : ''}${panTrim.toFixed(1)}° | Tilt: ${tiltTrim >= 0 ? '+' : ''}${tiltTrim.toFixed(1)}°`;
  const b1 = document.getElementById('trimBadge');
  if (b1) b1.textContent = text;
  const b2 = document.getElementById('ballsTrimBadge');
  if (b2) b2.textContent = text;
}

function populateCalibNodeDropdown() {
  const sel = document.getElementById('selCalibNode');
  if (!sel) return;
  sel.innerHTML = '';
  currentCalibNodes.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n.id;
    opt.textContent = `${n.label} - Pan ${n.pan_deg}°, Tilt ${n.tilt_deg}°`;
    if (n.id === selectedCalibNodeId) opt.selected = true;
    sel.appendChild(opt);
  });
}

function onSelectNodeDropdown(nodeId) {
  selectNodeForCalib(nodeId);
}

async function selectNodeForCalib(nodeId) {
  selectedCalibNodeId = nodeId;
  const node = currentCalibNodes.find(n => n.id === nodeId);
  if (!node) return;

  const sel = document.getElementById('selCalibNode');
  if (sel) sel.value = nodeId;

  const lbl = document.getElementById('lblCalibNodeTitle');
  if (lbl) lbl.textContent = `Tuning: ${node.label} (${node.x_mm}mm, ${node.y_mm}mm)`;

  const rPan = document.getElementById('rngCalibPan');
  const vPan = document.getElementById('valCalibPan');
  const rTilt = document.getElementById('rngCalibTilt');
  const vTilt = document.getElementById('valCalibTilt');

  if (rPan) rPan.value = node.pan_deg;
  if (vPan) vPan.textContent = node.pan_deg.toFixed(1) + '°';
  if (rTilt) rTilt.value = node.tilt_deg;
  if (vTilt) vTilt.textContent = node.tilt_deg.toFixed(1) + '°';

  // Move real physical laser directly to this node's baseline angles
  try {
    await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pan: node.pan_deg, tilt: node.tilt_deg, apply_trim: false })
    });
  } catch (e) {
    console.error('Select node move error:', e);
  }
}

async function reAimSelectedNode() {
  const pan = parseFloat(document.getElementById('rngCalibPan').value);
  const tilt = parseFloat(document.getElementById('rngCalibTilt').value);
  await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pan: pan, tilt: tilt, apply_trim: false })
  });
}

async function onTuneSlider(axis, val) {
  const v = parseFloat(val);
  if (axis === 'pan') {
    const el = document.getElementById('valCalibPan');
    if (el) el.textContent = v.toFixed(1) + '°';
  } else {
    const el = document.getElementById('valCalibTilt');
    if (el) el.textContent = v.toFixed(1) + '°';
  }
  const pan = parseFloat(document.getElementById('rngCalibPan').value);
  const tilt = parseFloat(document.getElementById('rngCalibTilt').value);
  try {
    await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pan: pan, tilt: tilt, apply_trim: false })
    });
  } catch (e) {
    console.error('Tune move error:', e);
  }
}

async function nudgeTune(axis, delta) {
  const rng = axis === 'pan' ? document.getElementById('rngCalibPan') : document.getElementById('rngCalibTilt');
  if (!rng) return;
  let v = parseFloat(rng.value) + delta;
  v = Math.round(v * 10) / 10;
  rng.value = v;
  await onTuneSlider(axis, v);
}

async function saveCurrentNodeCalib() {
  const pan = parseFloat(document.getElementById('rngCalibPan').value);
  const tilt = parseFloat(document.getElementById('rngCalibTilt').value);
  try {
    const res = await fetch('/api/calib/update_node', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: selectedCalibNodeId, pan: pan, tilt: tilt })
    });
    const data = await res.json();
    if (data.success) {
      alert(`✅ Calibrated node ${selectedCalibNodeId} saved! Pan: ${pan.toFixed(1)}°, Tilt: ${tilt.toFixed(1)}° recorded to snooker_calibration.json.`);
      await fetchNodes();
      updatePred();
    } else {
      alert(`Failed to save: ${data.error || data.message}`);
    }
  } catch (e) {
    alert(`Communication error while saving: ${e}`);
  }
}


async function emergencyStop() {
  await fetch('/api/stop', { method: 'POST' });
  isLaserActive = false;
  alert("EMERGENCY STOP ACTIVATED: Laser shut off and servos halted.");
}

function syncServoUi(axis, val) {
  const num = parseFloat(val);
  if (Number.isNaN(num)) return;
  if (axis === 'pan') {
    document.getElementById('valPan').textContent = num + '°';
    document.getElementById('rngPan').value = num;
    const inp = document.getElementById('inpPan');
    if (inp) inp.value = num;
    predPan = num;
  } else {
    document.getElementById('valTilt').textContent = num + '°';
    document.getElementById('rngTilt').value = num;
    const inp = document.getElementById('inpTilt');
    if (inp) inp.value = num;
    predTilt = num;
  }
}

async function homeServos() {
  syncServoUi('pan', 10);
  syncServoUi('tilt', 150);
  setPreset(0.0, 0.0);
  sendAimCommand();
}

async function centerServos() {
  syncServoUi('pan', 90);
  syncServoUi('tilt', 90);
  await fetch('/api/center', { method: 'POST' });
}

async function previewServo(axis, val) {
  syncServoUi(axis, val);
}

function clampServo(axis, val) {
  const min = axis === 'pan' ? 10 : 60;
  const max = axis === 'pan' ? 170 : 160;
  const parsed = parseFloat(val);
  if (Number.isNaN(parsed)) return axis === 'pan' ? predPan : predTilt;
  return Math.round(Math.min(max, Math.max(min, parsed)) * 10) / 10;
}

async function stepServo(axis, delta) {
  const rng = document.getElementById(axis === 'pan' ? 'rngPan' : 'rngTilt');
  const current = parseFloat(rng.value);
  const nextVal = clampServo(axis, current + delta);
  syncServoUi(axis, nextVal);
  await sendServoAngle(axis, nextVal);
}

async function applyManualServo(axis, rawVal) {
  const clamped = clampServo(axis, rawVal);
  syncServoUi(axis, clamped);
  await sendServoAngle(axis, clamped);
}

async function sendServoAngle(axis, val) {
  const pan = axis === 'pan' ? parseFloat(val) : parseFloat(document.getElementById('rngPan').value);
  const tilt = axis === 'tilt' ? parseFloat(val) : parseFloat(document.getElementById('rngTilt').value);
  syncServoUi('pan', pan);
  syncServoUi('tilt', tilt);
  await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pan: pan, tilt: tilt })
  });
}

function updateLaserUi(state) {
  isLaserActive = !!state;
  const hb = document.getElementById('btnHeaderLaser');
  if (hb) {
    if (isLaserActive) {
      hb.style.background = 'rgba(34, 197, 94, 0.2)';
      hb.style.color = '#15803d';
      hb.style.borderColor = '#22c55e';
      hb.innerHTML = '🟢 LASER: ON';
    } else {
      hb.style.background = 'rgba(239, 68, 68, 0.15)';
      hb.style.color = '#ef4444';
      hb.style.borderColor = '#ef4444';
      hb.innerHTML = '🔴 LASER: OFF';
    }
  }
  const tb = document.getElementById('btnLaserToggle');
  if (tb) {
    tb.textContent = isLaserActive ? "LASER IS ON (CLICK TO SHUT OFF)" : "TOGGLE LASER ON/OFF";
    tb.className = isLaserActive ? "btn btn-forest" : "btn btn-terracotta";
  }
}

async function toggleLaser() {
  const res = await fetch('/api/laser/toggle', { method: 'POST' });
  const data = await res.json();
  updateLaserUi(data.laser);
}

async function forceLaserOff() {
  await fetch('/api/laser/off', { method: 'POST' });
  updateLaserUi(false);
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

function setWebcamPreset(url) {
  const inp = document.getElementById('inpIpWebcam');
  if (inp) inp.value = url;
  saveIpWebcam();
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
  
  const customUrl = document.getElementById('inpIpWebcam') ? document.getElementById('inpIpWebcam').value.trim().replace(/\/+$/, '') : '';

  // 1. Try grabbing frame from backend
  try {
    const res = await fetch('/api/ipwebcam/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: customUrl })
    });
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
      return;
    }
  } catch (err) {
    console.warn("Backend capture attempt failed, trying browser canvas grab...", err);
  }

  // 2. Direct Browser Fallback: If video stream is visible in browser, freeze directly from <img> element!
  const streamImg = document.getElementById('liveStreamView');
  if (streamImg && (streamImg.naturalWidth > 0 || streamImg.complete)) {
    try {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = streamImg.naturalWidth || 1920;
      tempCanvas.height = streamImg.naturalHeight || 1080;
      const tctx = tempCanvas.getContext('2d');
      tctx.drawImage(streamImg, 0, 0, tempCanvas.width, tempCanvas.height);
      const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.92);

      // Upload frame to backend so homography & ball detection have it too
      fetch('/api/camera/upload_base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: dataUrl })
      });

      cleanFrozenImg = tempCanvas;
      const c = document.getElementById('cameraCanvas');
      c.width = tempCanvas.width;
      c.height = tempCanvas.height;
      homographyCorners = [];
      redrawCalibrationCanvas();
      setCameraDisplayMode('frozen');
      st.innerHTML = `✅ Frame frozen directly from visible stream (${c.width}×${c.height} px). Click 4 corners in order.`;
      st.style.color = 'var(--primary-green)';
      return;
    } catch (corsErr) {
      console.warn("Client canvas grab error (CORS):", corsErr);
    }
  }

  st.innerHTML = `❌ Cannot connect to ${customUrl || 'phone'}. Make sure IP Webcam is running and phone screen is unlocked.`;
  st.style.color = 'var(--terracotta)';
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

  // Check if user clicked near an existing corner to drag/adjust it
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

  // If clicking on new point and corners < 4
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
    el.innerHTML = `Corners set: <strong>${n} / 4</strong>. Next click: <strong style="color:var(--primary-green);">${labels[n]}</strong> <span style="font-size:0.7rem; color:var(--text-muted);">(You can drag any point to fine-tune)</span>`;
  } else {
    el.innerHTML = `<span style="color:var(--primary-green); font-weight:700;">✅ All 4 Corners Set!</span> <span style="font-size:0.7rem; color:var(--text-muted);">(Drag any corner to adjust, or click 'COMPUTE PLANAR HOMOGRAPHY')</span>`;
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

  const resScale = Math.max(1.0, c.width / 1280.0);
  const reticleR = 12 * resScale;
  const crossLen = 22 * resScale;
  const lineW = 3.5 * resScale;
  const fontPx = Math.round(16 * resScale);

  if (homographyCorners.length >= 2) {
    ctx.lineWidth = lineW;
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.95)';
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
    ctx.font = `bold ${fontPx}px ui-monospace, sans-serif`;
    const tag = names[i];
    const tw = ctx.measureText(tag).width;
    const badgeH = fontPx + 10 * resScale;
    const badgeW = tw + 14 * resScale;
    const badgeX = x + 14 * resScale;
    const badgeY = y - badgeH / 2;

    ctx.fillStyle = 'rgba(11, 20, 16, 0.92)';
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
    document.getElementById('ipWebcamStatus').style.color = d.success ? 'var(--primary-green)' : 'var(--terracotta)';
    reloadLiveStream();
  }).catch(e => {
    document.getElementById('ipWebcamStatus').textContent = '❌ Error: ' + e.message;
    document.getElementById('ipWebcamStatus').style.color = 'var(--terracotta)';
  });
}

async function autoScanPhoneIP() {
  const st = document.getElementById('ipWebcamStatus');
  st.innerHTML = '🔍 Scanning local network for your phone camera...';
  st.style.color = 'var(--text-muted)';
  try {
    const res = await fetch('/api/ipwebcam/scan', { method: 'POST' });
    const data = await res.json();
    if (data.found) {
      document.getElementById('inpIpWebcam').value = data.url;
      st.innerHTML = `✅ Found phone at <b>${data.url}</b>! Stream connected.`;
      st.style.color = 'var(--primary-green)';
      saveIpWebcam();
    } else {
      st.innerHTML = `❌ Phone camera not found on network.<br>` +
        `<b>Please verify:</b><br>` +
        `1. In IP Webcam app, scroll down and tap <b>"Start server"</b>.<br>` +
        `2. Keep the phone screen <b>awake & unlocked</b> (Android pauses background streams).<br>` +
        `3. Enter the exact URL displayed at the bottom of your phone screen.`;
      st.style.color = 'var(--terracotta)';
    }
  } catch (e) {
    st.innerHTML = '❌ Scan error: ' + e.message;
    st.style.color = 'var(--terracotta)';
  }
}

async function captureAndDetect() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Capturing snapshot + running YOLO detection...';
  st.style.color = 'var(--text-muted)';
  const customUrl = document.getElementById('inpIpWebcam') ? document.getElementById('inpIpWebcam').value.trim().replace(/\/+$/, '') : '';
  try {
    const capRes = await fetch('/api/ipwebcam/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: customUrl })
    });
    const capData = await capRes.json();
    if (!capData.success) {
      // Fallback: If live phone timed out, check if we have a frozen frame or cached image to detect from
      if (cleanFrozenImg || capData.has_cached_frame) {
        st.innerHTML = `⚠️ Phone camera timed out. Using current table frame for YOLO detection...`;
        st.style.color = 'var(--terracotta)';
        await runDetection();
        st.innerHTML = `✅ Detection complete using current table frame! View balls on table.`;
        st.style.color = 'var(--primary-green)';
        showTab('tab-balls', document.querySelectorAll('.nav-btn')[3]);
        return;
      }
      st.innerHTML = `❌ <b>Connection Timed Out to ${customUrl}</b><br>` +
        `<span style="color:var(--text-dark); font-weight:600;">Why this happens on Android phones:</span><br>` +
        `1. <b>Screen locked/asleep:</b> Android kills background camera streaming when the phone screen turns off. Wake your phone and keep IP Webcam open.<br>` +
        `2. <b>Server not started:</b> In IP Webcam app, scroll to the bottom and tap <b>"Start server"</b>.<br>` +
        `3. <b>Check IP:</b> Look at the URL printed at the bottom of your phone screen or click <b>"🔍 Auto-Detect Phone"</b>.`;
      st.style.color = 'var(--terracotta)';
      return;
    }
    st.innerHTML = `✅ Captured ${capData.width}×${capData.height}. Running YOLOv8 snooker model...`;
    st.style.color = 'var(--primary-green)';

    await runDetection();
    st.innerHTML = `✅ Captured + Detection complete! View balls on table.`;
    showTab('tab-balls', document.querySelectorAll('.nav-btn')[3]);
  } catch (e) {
    if (cleanFrozenImg) {
      st.innerHTML = `⚠️ Connection error. Running detection on current table frame...`;
      await runDetection();
      st.innerHTML = `✅ Detection complete using current table frame!`;
      st.style.color = 'var(--primary-green)';
      showTab('tab-balls', document.querySelectorAll('.nav-btn')[3]);
    } else {
      st.textContent = '❌ Error: ' + e.message;
      st.style.color = 'var(--terracotta)';
    }
  }
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    updateESPBadge(data.connected);
    if (data.laser !== undefined) {
      updateLaserUi(data.laser);
    }
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

// lastDetectedBalls declared globally at top of script

function setDetectLoading(isLoading, msg) {
  const sp = document.getElementById('detectSpinner');
  const txt = document.getElementById('detectSpinnerText');
  if (sp) sp.style.display = isLoading ? 'flex' : 'none';
  if (txt && msg) txt.textContent = msg;
}

function renderDetectionResult(data) {
  setDetectLoading(false);
  if (!data || !data.success) {
    alert(data && data.error ? data.error : "Detection failed");
    return;
  }

  // 1. Show annotated image
  const img = document.getElementById('detectAnnotatedImg');
  const ph = document.getElementById('detectPlaceholder');
  if (img && data.annotated_image) {
    img.src = data.annotated_image;
    img.style.display = 'block';
  }
  if (ph) ph.style.display = 'none';

  // 2. Summary stats
  const sumBox = document.getElementById('detectSummaryBox');
  const totalCount = document.getElementById('detectTotalCount');
  const lat = document.getElementById('detectLatency');
  const chips = document.getElementById('detectClassChips');

  if (sumBox) sumBox.style.display = 'block';
  if (totalCount) totalCount.textContent = `Detected Balls (${data.count || 0})`;
  if (lat) lat.textContent = `Latency: ${data.inference_ms || 0} ms • Model: YOLOv8s`;

  if (chips && data.summary) {
    chips.innerHTML = '';
    const chipColors = {
      'orange ball': '#ee1c00',
      'pink ball': '#f472b6',
      'white ball': '#f8fafc',
      'blue ball': '#38bdf8',
      'green ball': '#4ade80',
      'yellow ball': '#facc15'
    };
    for (const [cls, cnt] of Object.entries(data.summary)) {
      const c = chipColors[cls.toLowerCase()] || 'var(--primary-green)';
      chips.innerHTML += `<span style="display:inline-flex; align-items:center; gap:5px; background:rgba(0,0,0,0.06); border:1px solid rgba(0,0,0,0.12); border-radius:999px; padding:3px 10px; font-size:0.73rem; font-weight:600;">
        <span style="width:8px; height:8px; border-radius:50%; background:${c}; display:inline-block; border:1px solid rgba(0,0,0,0.2);"></span>
        ${cnt}x ${cls}
      </span>`;
    }
  }

  // 3. Table rows
  lastDetectedBalls = data.balls || [];
  detected_balls_cache = lastDetectedBalls;
  const tb = document.getElementById('ballsTableBody');
  if (tb) {
    tb.innerHTML = '';
    if (lastDetectedBalls.length === 0) {
      tb.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:16px;">No balls detected in this image.</td></tr>';
    } else {
      lastDetectedBalls.forEach(b => {
        const px = b.center_px ? `(${b.center_px[0]}, ${b.center_px[1]})` : '---';
        const tx = b.table_x_mm !== null && b.table_x_mm !== undefined ? b.table_x_mm.toFixed(1) : '---';
        const ty = b.table_y_mm !== null && b.table_y_mm !== undefined ? b.table_y_mm.toFixed(1) : '---';
        const tablePosStr = tx !== '---' ? `${tx}, ${ty}` : '<span style="color:var(--text-muted);">Calibrate corners</span>';
        tb.innerHTML += `<tr>
          <td>${b.id}</td>
          <td><strong>${b.label}</strong></td>
          <td>${(b.confidence * 100).toFixed(0)}%</td>
          <td class="mono" style="font-size:0.75rem;">${px}</td>
          <td class="mono" style="font-size:0.75rem;">${tablePosStr}</td>
          <td><button class="btn btn-forest" style="padding:5px 12px; font-size:0.74rem; font-weight:700;" onclick="aimAtCoordinates(${b.table_x_mm || 0}, ${b.table_y_mm || 0})">🎯 Aim Laser</button></td>
        </tr>`;
      });
    }
  }

  // 4. Automatically sync onto 3D table
  syncDetectedBallsTo3D();
}

async function onDetectFileUpload(input) {
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('image', file);

  setDetectLoading(true, `Uploading ${file.name} & running YOLOv8s...`);
  try {
    const res = await fetch('/api/balls/detect_image', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    renderDetectionResult(data);
  } catch (err) {
    setDetectLoading(false);
    alert('Upload detection error: ' + err.message);
  }
}

async function detectFromLiveCamera() {
  setDetectLoading(true, 'Grabbing frame from live camera & running YOLOv8s...');
  try {
    const res = await fetch('/api/balls/detect_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_current_frame: true })
    });
    const data = await res.json();
    renderDetectionResult(data);
  } catch (err) {
    setDetectLoading(false);
    alert('Camera detection error: ' + err.message);
  }
}

async function loadDetectSample(sampleId) {
  setDetectLoading(true, `Loading sample ${sampleId} & running YOLOv8s...`);
  try {
    const res = await fetch('/api/balls/detect_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId })
    });
    const data = await res.json();
    renderDetectionResult(data);
  } catch (err) {
    setDetectLoading(false);
    alert('Sample detection error: ' + err.message);
  }
}

function syncDetectedBallsTo3D() {
  if (!lastDetectedBalls || lastDetectedBalls.length === 0) return;
  clearBalls();
  lastDetectedBalls.forEach(b => {
    if (b.table_x_mm !== null && b.table_y_mm !== null) {
      let col = SNOOKER_COLORS.orange;
      const lbl = b.label.toLowerCase();
      if (lbl.includes("cue") || lbl.includes("white")) col = SNOOKER_COLORS.cue;
      else if (lbl.includes("orange")) col = SNOOKER_COLORS.orange;
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

async function runDetection() {
  detectFromLiveCamera();
}

function exportCsv() {
  window.location.href = '/api/logs/export';
}

function runInit() {
  try {
    initThree();
  } catch (err) {
    console.error("Three.js Init Error:", err);
  }
  fetchNodes();
  updatePred();
  setTimeout(initCameraModule, 200);
  setInterval(pollStatus, 4000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', runInit);
} else {
  runInit();
}

// ==============================================================
// 11. AI FOUL DETECTION & REFEREE REPOSITIONING LOGIC
// ==============================================================
let preShotBallCount = 0;

async function capturePreShotState() {
  const currentBalls = (typeof lastDetectedBalls !== "undefined" && lastDetectedBalls.length > 0) ? lastDetectedBalls : ((typeof detected_balls_cache !== "undefined" && detected_balls_cache.length > 0) ? detected_balls_cache : []);
  try {
    const res = await fetch('/api/foul/snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ balls: currentBalls })
    });
    const data = await res.json();
    if (data.success) {
      preShotBallCount = data.count;
      updatePreShotBadge(data.count);
      alert(`📸 Pre-shot reference state captured with ${data.count} balls! Proceed with your shot.`);
    }
  } catch (e) {
    alert(`Error capturing pre-shot state: ${e}`);
  }
}

function updatePreShotBadge(count) {
  const badge = document.getElementById('foulPreShotBadge');
  if (badge) {
    if (count > 0) {
      badge.className = 'tag-pill tag-online';
      badge.textContent = `Pre-Shot Reference: ${count} balls stored`;
    } else {
      badge.className = 'tag-pill';
      badge.textContent = 'Pre-Shot Reference: None';
    }
  }
}

async function analyzeShotForFoul() {
  const nominated = document.getElementById('selNominatedBall').value;
  try {
    const res = await fetch('/api/foul/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nominated: nominated, balls: (typeof lastDetectedBalls !== "undefined" ? lastDetectedBalls : []) })
    });
    const data = await res.json();
    if (data.success) {
      renderFoulReport(data.report);
    }
  } catch (e) {
    alert(`Shot analysis error: ${e}`);
  }
}

async function captureLiveAndAnalyzeFoul() {
  // Capture fresh frame from camera, run YOLO detection, then analyze foul
  try {
    await captureAndDetect();
    await analyzeShotForFoul();
  } catch (e) {
    alert(`Live cam shot analysis error: ${e}`);
  }
}

function renderFoulReport(rep) {
  const box = document.getElementById('foulVerdictBox');
  const icon = document.getElementById('foulVerdictIcon');
  const title = document.getElementById('foulVerdictTitle');
  const penaltyBadge = document.getElementById('foulPenaltyBadge');
  const desc = document.getElementById('foulVerdictDesc');
  const countBadge = document.getElementById('displacedCountBadge');
  const tb = document.getElementById('foulDisplacedTableBody');

  if (!box) return;
  box.style.display = 'block';

  if (rep.is_foul) {
    box.style.background = 'rgba(239, 68, 68, 0.12)';
    box.style.border = '1px solid rgba(239, 68, 68, 0.4)';
    icon.textContent = '🚨';
    title.textContent = `FOUL DETECTED: ${rep.foul_type.replace(/_/g, ' ')}`;
    title.style.color = '#dc2626';
    penaltyBadge.className = 'status-pill badge-off';
    penaltyBadge.textContent = `PENALTY: ${rep.penalty_points} PTS`;
    desc.innerHTML = `<strong>${rep.description}</strong><br><span style="font-size:0.75rem; color:var(--text-muted);">Nominated target was: <strong>${rep.nominated_target.toUpperCase()}</strong>. Displaced balls can be restored below using the physical laser pointer.</span>`;
  } else {
    box.style.background = 'rgba(34, 197, 94, 0.12)';
    box.style.border = '1px solid rgba(34, 197, 94, 0.4)';
    icon.textContent = '✅';
    title.textContent = 'LEGAL SHOT CONFIRMED';
    title.style.color = 'var(--primary-forest)';
    penaltyBadge.className = 'status-pill tag-online';
    penaltyBadge.textContent = 'NO FOUL';
    desc.textContent = rep.description;
  }

  // Render displaced balls table
  const disp = rep.displaced_balls || [];
  if (countBadge) countBadge.textContent = `${disp.length} balls displaced/potted`;

  if (tb) {
    if (disp.length === 0) {
      tb.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted); padding:14px;">No balls were displaced during this shot.</td></tr>';
    } else {
      tb.innerHTML = '';
      disp.forEach((d, idx) => {
        const isPotted = d.status === 'POTTED';
        const statusPill = isPotted 
          ? '<span class="status-pill badge-off" style="font-size:0.65rem;">POTTED</span>'
          : '<span class="status-pill badge-sim" style="font-size:0.65rem;">MOVED</span>';
        const postCoord = isPotted ? '<em>Pocketed</em>' : `(${d.post_x}, ${d.post_y})`;
        const shift = isPotted ? 'Off Table' : `${d.displacement_mm} mm`;

        tb.innerHTML += `<tr id="rowDisplaced_${idx}">
          <td><strong>${d.label}</strong></td>
          <td>${statusPill}</td>
          <td class="mono" style="color:var(--primary-green); font-weight:700;">(${d.pre_x}, ${d.pre_y}) mm</td>
          <td class="mono">${postCoord}</td>
          <td class="mono">${shift}</td>
          <td>
            <button class="btn btn-forest" style="padding:4px 10px; font-size:0.7rem;" onclick="aimLaserAtPreFoul(${d.pre_x}, ${d.pre_y}, '${d.label}')">
              🎯 Aim Laser to Restore
            </button>
          </td>
        </tr>`;
      });
    }
  }
}

async function aimLaserAtPreFoul(x, y, label) {
  setPreset(x, y);
  try {
    const res = await fetch('/api/foul/reposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: x, y: y })
    });
    const data = await res.json();
    if (data.success) {
      alert(`🎯 Physical laser aiming at pre-foul spot for ${label} (${x}mm, ${y}mm)! Place the ball under the red laser beam.`);
    }
  } catch (e) {
    alert(`Laser reposition error: ${e}`);
  }
}

async function aimAtRegulationSpot(spotKey) {
  try {
    const res = await fetch('/api/foul/regulation_spots');
    const data = await res.json();
    if (data.success && data.spots[spotKey]) {
      const sp = data.spots[spotKey];
      setPreset(sp.x_mm, sp.y_mm);
      await fetch('/api/foul/reposition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: sp.x_mm, y: sp.y_mm })
      });
      alert(`🎯 Physical laser aimed at ${sp.name} (${sp.x_mm}mm, ${sp.y_mm}mm) for color respotting.`);
    }
  } catch (e) {
    alert(`Error aiming at regulation spot: ${e}`);
  }
}


// ==============================================================
// 12. FULLY AUTONOMOUS LIVE REFEREE STATE MACHINE
// ==============================================================
let isAutoRefereeActive = false;
let autoRefereeState = 'STANDBY'; // 'STANDBY', 'READY', 'MOTION', 'SETTLING', 'ANALYZING', 'FOUL_HOLD'
let motionSamplerTimer = null;
let lastMotionLuma = null;
let motionDetectionCanvas = null;
let motionDetectionCtx = null;
let settleStartTime = null;
const MOTION_THRESHOLD = 3.2;      // Percent diff to register stroke
const SETTLE_DELAY_MS = 1400;      // 1.4s of calm to confirm balls stopped

let lastDisplacedBalls = [];
let currentRestoreIdx = 0;

function setRefereeMode(mode) {
  const autoPanel = document.getElementById('autoRefereePanel');
  const manualPanel = document.getElementById('manualRefereePanel');
  const btnManual = document.getElementById('btnModeManual');
  const tag = document.getElementById('refereeModeTag');

  if (mode === 'manual') {
    if (isAutoRefereeActive) toggleAutonomousReferee();
    if (autoPanel) autoPanel.style.display = 'none';
    if (manualPanel) manualPanel.style.display = 'block';
    if (btnManual) {
      btnManual.className = 'btn btn-forest';
      btnManual.textContent = '📸 Manual 2-Shot Mode: ON';
    }
    if (tag) tag.textContent = 'MANUAL 2-SHOT MODE';
  } else {
    if (autoPanel) autoPanel.style.display = 'block';
    if (manualPanel) manualPanel.style.display = 'none';
    if (btnManual) {
      btnManual.className = 'btn btn-secondary';
      btnManual.textContent = '📸 Manual 2-Shot Mode';
    }
    if (tag) tag.textContent = 'LIVE MOTION AI';
  }
}

let manualPreShotImgObj = null;
let manualCornerMode = false;
let manualPickedCorners = [];
let manualCalibratedCorners = [];

function drawManualPreShotCanvas() {
  const cvs = document.getElementById('cvsManualPreShot');
  const ph = document.getElementById('lblManualPrePlaceholder');
  if (!cvs || !manualPreShotImgObj) return;
  cvs.width = manualPreShotImgObj.naturalWidth || manualPreShotImgObj.width || 640;
  cvs.height = manualPreShotImgObj.naturalHeight || manualPreShotImgObj.height || 360;
  cvs.style.display = 'block';
  if (ph) ph.style.display = 'none';

  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.drawImage(manualPreShotImgObj, 0, 0, cvs.width, cvs.height);

  const pts = manualCornerMode ? manualPickedCorners : manualCalibratedCorners;
  if (pts && pts.length > 0) {
    if (pts.length === 4) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = Math.max(2, cvs.width / 320);
      ctx.stroke();
    }
    const labels = ['TR', 'TL', 'BL', 'BR'];
    pts.forEach((p, idx) => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], Math.max(6, cvs.width / 110), 0, Math.PI * 2);
      ctx.fillStyle = manualCornerMode ? '#f59e0b' : '#22c55e';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${Math.max(12, Math.round(cvs.width / 45))}px monospace`;
      ctx.fillText(labels[idx] || String(idx + 1), p[0] + 8, p[1] - 6);
    });
  }
}

function startManualCornerPick() {
  if (!manualPreShotImgObj) {
    alert('Please click "1. Take Pre-Shot Image" first.');
    return;
  }
  manualCornerMode = true;
  manualPickedCorners = [];
  const lbl = document.getElementById('lblManualHomogStatus');
  if (lbl) lbl.textContent = '🎯 Click Corner 1/4: Top-Right (TR) on Pre-Shot image';
  drawManualPreShotCanvas();
}

async function onManualPreShotCanvasClick(event) {
  if (!manualCornerMode || !manualPreShotImgObj) return;
  const cvs = document.getElementById('cvsManualPreShot');
  const rect = cvs.getBoundingClientRect();
  // Account for object-fit: contain scaling inside canvas box
  const scale = Math.min(rect.width / cvs.width, rect.height / cvs.height);
  const drawW = cvs.width * scale;
  const drawH = cvs.height * scale;
  const offX = (rect.width - drawW) / 2;
  const offY = (rect.height - drawH) / 2;
  const x = (event.clientX - rect.left - offX) / scale;
  const y = (event.clientY - rect.top - offY) / scale;
  if (x < 0 || y < 0 || x > cvs.width || y > cvs.height) return;

  manualPickedCorners.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
  drawManualPreShotCanvas();

  const names = ['Top-Right (TR)', 'Top-Left (TL)', 'Bottom-Left (BL)', 'Bottom-Right (BR)'];
  const lbl = document.getElementById('lblManualHomogStatus');
  if (manualPickedCorners.length < 4) {
    if (lbl) lbl.textContent = `🎯 Click Corner ${manualPickedCorners.length + 1}/4: ${names[manualPickedCorners.length]}`;
    return;
  }

  manualCornerMode = false;
  if (lbl) lbl.textContent = '⏳ Calibrating Homography & Re-detecting Pre-Shot...';
  try {
    const res = await fetch('/api/foul/manual_calibrate_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners: manualPickedCorners })
    });
    const data = await res.json();
    applyManualCalibrationResult(data);
  } catch (e) {
    alert(`Calibration error: ${e}`);
  }
}

async function autoCalibrateManualPreShot() {
  const lbl = document.getElementById('lblManualHomogStatus');
  if (lbl) lbl.textContent = '⏳ Auto-detecting table corners...';
  try {
    const res = await fetch('/api/foul/manual_calibrate_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ auto: true })
    });
    const data = await res.json();
    applyManualCalibrationResult(data);
  } catch (e) {
    alert(`Auto-calibration error: ${e}`);
  }
}

function applyManualCalibrationResult(data) {
  const lbl = document.getElementById('lblManualHomogStatus');
  const pill = document.getElementById('manualRefereePill');
  if (!data.success) {
    if (lbl) lbl.textContent = '⚠️ Calibration failed — click Pick 4 Corners';
    alert(data.error || 'Could not calibrate homography.');
    return;
  }
  manualCalibratedCorners = data.corners || [];
  drawManualPreShotCanvas();
  preShotBallCount = data.count || 0;
  if (data.balls && typeof update3DBalls === 'function') {
    lastDetectedBalls = data.balls;
    update3DBalls(data.balls);
  }
  if (lbl) lbl.textContent = `✅ Homography Calibrated (${preShotBallCount} balls locked)`;
  if (pill) {
    pill.className = preShotBallCount > 0 ? 'status-pill tag-online' : 'status-pill badge-sim';
    pill.textContent = preShotBallCount > 0
      ? `✅ CALIBRATED & PRE-SHOT LOCKED (${preShotBallCount} BALLS) — PLAY SHOT`
      : '⚠️ CALIBRATED BUT 0 BALLS DETECTED — CHECK TABLE/CAMERA';
  }
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function uploadManualPreShot(inputEl) {
  if (!inputEl || !inputEl.files || !inputEl.files[0]) return;
  try {
    const b64 = await readFileAsDataURL(inputEl.files[0]);
    inputEl.value = '';
    await captureManualPreShot(b64);
  } catch (e) {
    alert(`Failed to read Pre-Shot image: ${e}`);
  }
}

async function uploadManualPostShotAndCompare(inputEl) {
  if (!inputEl || !inputEl.files || !inputEl.files[0]) return;
  try {
    const b64 = await readFileAsDataURL(inputEl.files[0]);
    inputEl.value = '';
    await captureManualPostShotAndCompare(b64);
  } catch (e) {
    alert(`Failed to read Post-Shot image: ${e}`);
  }
}

async function captureManualPreShot(uploadedB64 = null) {
  const pill = document.getElementById('manualRefereePill');
  const lblHomog = document.getElementById('lblManualHomogStatus');
  if (pill) pill.textContent = '⏳ PROCESSING PRE-SHOT, CALIBRATING HOMOGRAPHY & DETECTING...';
  manualCornerMode = false;
  try {
    const payload = uploadedB64 ? { image_b64: uploadedB64 } : {};
    const res = await fetch('/api/foul/manual_preshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      alert(data.error || 'Failed to capture pre-shot image.');
      return;
    }
    preShotBallCount = data.count || 0;
    manualCalibratedCorners = data.corners || [];
    if (data.balls && typeof update3DBalls === 'function') {
      lastDetectedBalls = data.balls;
      update3DBalls(data.balls);
    }
    if (data.image_b64) {
      const img = new Image();
      img.onload = () => {
        manualPreShotImgObj = img;
        drawManualPreShotCanvas();
      };
      img.src = data.image_b64;
    }
    // Reset post-shot preview & restore bar for the new shot
    const postImg = document.getElementById('imgManualPostShot');
    const postPh = document.getElementById('lblManualPostPlaceholder');
    const restoreBar = document.getElementById('manualRestoreBar');
    if (postImg) { postImg.style.display = 'none'; postImg.src = ''; }
    if (postPh) postPh.style.display = 'block';
    if (restoreBar) restoreBar.style.display = 'none';

    if (lblHomog) {
      lblHomog.textContent = data.homography_calibrated
        ? `✅ Homography Calibrated (${preShotBallCount} balls)`
        : '⚠️ Homography Not Calibrated — Click "Pick 4 Corners"';
    }

    if (pill) {
      if (!data.homography_calibrated) {
        pill.className = 'status-pill badge-off';
        pill.textContent = '⚠️ CLICK "PICK 4 CORNERS" ON PRE-SHOT IMAGE TO CALIBRATE';
      } else if (preShotBallCount === 0) {
        pill.className = 'status-pill badge-off';
        pill.textContent = '⚠️ CALIBRATED BUT 0 BALLS FOUND — CHECK CAMERA / PICK CORNERS';
      } else {
        pill.className = 'status-pill tag-online';
        pill.textContent = `✅ CALIBRATED & PRE-SHOT LOCKED (${preShotBallCount} BALLS) — PLAY SHOT`;
      }
    }
  } catch (e) {
    alert(`Pre-shot capture error: ${e}`);
  }
}

async function captureManualPostShotAndCompare(uploadedB64 = null) {
  const pill = document.getElementById('manualRefereePill');
  const nominated = (document.getElementById('selManualNominatedBall') || document.getElementById('selNominatedBall')).value;
  if (pill) pill.textContent = '⏳ ANALYZING POST-SHOT IMAGE & COMPARING...';
  try {
    const payload = uploadedB64 ? { nominated: nominated, image_b64: uploadedB64 } : { nominated: nominated };
    const res = await fetch('/api/foul/manual_postshot_compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      if (pill) {
        pill.className = 'status-pill badge-sim';
        pill.textContent = '📸 STEP 1: TAKE PRE-SHOT IMAGE FIRST';
      }
      alert(data.error || 'Please take a Pre-Shot image first.');
      return;
    }

    if (data.balls && typeof update3DBalls === 'function') {
      lastDetectedBalls = data.balls;
      update3DBalls(data.balls);
    }
    if (data.image_b64) {
      const img = document.getElementById('imgManualPostShot');
      const ph = document.getElementById('lblManualPostPlaceholder');
      if (img) { img.src = data.image_b64; img.style.display = 'block'; }
      if (ph) ph.style.display = 'none';
    }

    renderFoulReport(data.report);
    lastDisplacedBalls = (data.report && data.report.displaced_balls) ? data.report.displaced_balls : [];
    currentRestoreIdx = 0;

    const restoreBar = document.getElementById('manualRestoreBar');
    const lblStatus = document.getElementById('lblManualRestoreStatus');
    if (lastDisplacedBalls.length > 0) {
      const first = lastDisplacedBalls[0];
      setPreset(first.pre_x, first.pre_y);
      if (restoreBar) restoreBar.style.display = 'flex';
      if (lblStatus) {
        lblStatus.textContent = `🎯 Pointing Ball 1 of ${lastDisplacedBalls.length}: ${first.label} (${first.pre_x}, ${first.pre_y} mm)`;
      }
      if (pill) {
        pill.className = 'status-pill badge-off';
        pill.textContent = `🚨 ${lastDisplacedBalls.length} DISPLACED BALL(S) — LASER POINTING TO RESTORE`;
      }
    } else {
      if (restoreBar) restoreBar.style.display = 'none';
      if (pill) {
        pill.className = 'status-pill tag-online';
        pill.textContent = '✅ NO DISPLACED BALLS DETECTED';
      }
    }
  } catch (e) {
    alert(`Post-shot comparison error: ${e}`);
  }
}

async function pointNextDisplacedBall() {
  if (!lastDisplacedBalls || lastDisplacedBalls.length === 0) return;
  currentRestoreIdx = (currentRestoreIdx + 1) % lastDisplacedBalls.length;
  const target = lastDisplacedBalls[currentRestoreIdx];
  const lblStatus = document.getElementById('lblManualRestoreStatus');
  if (lblStatus) {
    lblStatus.textContent = `🎯 Pointing Ball ${currentRestoreIdx + 1} of ${lastDisplacedBalls.length}: ${target.label} (${target.pre_x}, ${target.pre_y} mm)`;
  }
  setPreset(target.pre_x, target.pre_y);
  try {
    await fetch('/api/foul/reposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: target.pre_x, y: target.pre_y })
    });
  } catch (e) {
    console.error('Error pointing next displaced ball:', e);
  }
}

function resetManualRefereeRound() {
  lastDisplacedBalls = [];
  currentRestoreIdx = 0;
  manualPreShotImgObj = null;
  manualCornerMode = false;
  manualPickedCorners = [];
  manualCalibratedCorners = [];
  const preCvs = document.getElementById('cvsManualPreShot');
  const prePh = document.getElementById('lblManualPrePlaceholder');
  const postImg = document.getElementById('imgManualPostShot');
  const postPh = document.getElementById('lblManualPostPlaceholder');
  const restoreBar = document.getElementById('manualRestoreBar');
  const pill = document.getElementById('manualRefereePill');
  const lblHomog = document.getElementById('lblManualHomogStatus');
  if (preCvs) preCvs.style.display = 'none';
  if (prePh) prePh.style.display = 'block';
  if (postImg) { postImg.style.display = 'none'; postImg.src = ''; }
  if (postPh) postPh.style.display = 'block';
  if (restoreBar) restoreBar.style.display = 'none';
  if (lblHomog) lblHomog.textContent = '📐 Homography: Waiting for Pre-Shot';
  if (pill) {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '📸 STEP 1: TAKE PRE-SHOT IMAGE BEFORE PLAY';
  }
}

function toggleAutonomousReferee() {
  isAutoRefereeActive = !isAutoRefereeActive;
  const btn = document.getElementById('btnToggleAutoReferee');
  if (isAutoRefereeActive) {
    setRefereeMode('auto');
    if (btn) {
      btn.className = 'btn btn-forest';
      btn.textContent = '🟢 AUTONOMOUS REFEREE: ON';
    }
    startAutonomousRefereeLoop();
  } else {
    if (btn) {
      btn.className = 'btn btn-secondary';
      btn.textContent = '⚪ AUTONOMOUS REFEREE: OFF';
    }
    stopAutonomousRefereeLoop();
  }
}

function startAutonomousRefereeLoop() {
  reloadAutoRefereeStream();
  if (!motionDetectionCanvas) {
    motionDetectionCanvas = document.createElement('canvas');
    motionDetectionCanvas.width = 160;
    motionDetectionCanvas.height = 90;
    motionDetectionCtx = motionDetectionCanvas.getContext('2d', { willReadFrequently: true });
  }

  setAutoState('INITIALIZING');
  // Lock baseline safely
  fetch('/api/foul/snapshot', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ balls: (typeof lastDetectedBalls !== 'undefined' ? lastDetectedBalls : []) })
  }).then(res => res.json()).then(data => {
    preShotBallCount = data.count || 0;
    updatePreShotBadge(data.count || 0);
    setAutoState('READY');
    lastMotionLuma = null;
    if (motionSamplerTimer) clearInterval(motionSamplerTimer);
    motionSamplerTimer = setInterval(sampleLiveCameraMotion, 120);
  }).catch(err => {
    console.warn('Init auto referee baseline error:', err);
    setAutoState('READY');
    if (motionSamplerTimer) clearInterval(motionSamplerTimer);
    motionSamplerTimer = setInterval(sampleLiveCameraMotion, 120);
  });
}

function stopAutonomousRefereeLoop() {
  if (motionSamplerTimer) {
    clearInterval(motionSamplerTimer);
    motionSamplerTimer = null;
  }
  setAutoState('STANDBY');
  const bar = document.getElementById('barMotion');
  const lbl = document.getElementById('lblMotionVal');
  if (bar) bar.style.width = '0%';
  if (lbl) lbl.textContent = 'Motion: 0.0% | Referee Standby';
}

function setAutoState(newState) {
  autoRefereeState = newState;
  const pill = document.getElementById('autoRefereePill');
  const btnResume = document.getElementById('btnResumeAutoAfterFoul');
  if (!pill) return;

  if (newState === 'STANDBY') {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '⚪ REFEREE STANDBY (TAP ENABLE ABOVE)';
    if (btnResume) btnResume.style.display = 'none';
  } else if (newState === 'INITIALIZING') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> LOCKING PRE-SHOT BASELINE...';
  } else if (newState === 'READY') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> 🟢 READY FOR SHOT (PRE-SHOT LOCKED)';
    if (btnResume) btnResume.style.display = 'none';
  } else if (newState === 'MOTION') {
    pill.className = 'status-pill badge-sim';
    pill.style.background = 'rgba(234, 179, 8, 0.2)';
    pill.style.color = '#ca8a04';
    pill.style.borderColor = '#eab308';
    pill.innerHTML = '<span class="live-pulsing-dot" style="background:#eab308;"></span> 🟡 BALLS IN MOTION (WATCHING SHOT...)';
  } else if (newState === 'SETTLING') {
    pill.className = 'status-pill badge-sim';
    pill.textContent = '⏳ BALLS STOPPING... SETTLING';
  } else if (newState === 'ANALYZING') {
    pill.className = 'status-pill tag-online';
    pill.innerHTML = '<span class="live-pulsing-dot"></span> 🔵 ANALYZING SHOT & FOUL RULES...';
  } else if (newState === 'FOUL_HOLD') {
    pill.className = 'status-pill badge-off';
    pill.textContent = '🚨 FOUL DETECTED! LASER AIMING AT MISPLACED BALL';
    if (btnResume) btnResume.style.display = 'inline-block';
  }
}

function handleAutoRefereeStreamLoaded(img) {
  const warn = document.getElementById('autoRefereeCamWarning');
  if (warn) warn.style.display = 'none';
}

function handleAutoRefereeStreamError(img) {
  const warn = document.getElementById('autoRefereeCamWarning');
  if (warn) warn.style.display = 'flex';
}

function reloadAutoRefereeStream() {
  const img = document.getElementById('autoRefereeLiveImg');
  if (img) img.src = '/api/camera/stream?t=' + Date.now();
}

function sampleLiveCameraMotion() {
  if (!isAutoRefereeActive || autoRefereeState === 'ANALYZING' || autoRefereeState === 'FOUL_HOLD') {
    return;
  }

  // Sample from autoRefereeLiveImg (same-origin, never cross-origin tainted)
  const streamImg = document.getElementById('autoRefereeLiveImg') || document.getElementById('liveStreamView');
  if (!streamImg || !streamImg.complete || !streamImg.naturalWidth) {
    const lbl = document.getElementById('lblMotionVal');
    if (lbl && isAutoRefereeActive) {
      lbl.textContent = 'Connecting to camera video stream... (Check IP Webcam)';
    }
    return;
  }

  try {
    motionDetectionCtx.drawImage(streamImg, 0, 0, 160, 90);
    const imgData = motionDetectionCtx.getImageData(0, 0, 160, 90);
    const data = imgData.data;

    if (!lastMotionLuma) {
      lastMotionLuma = new Uint8Array(160 * 90);
      for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        lastMotionLuma[j] = (data[i] * 299 + data[i+1] * 587 + data[i+2] * 114) >> 10;
      }
      return;
    }

    let diffSum = 0;
    const currLuma = new Uint8Array(160 * 90);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const luma = (data[i] * 299 + data[i+1] * 587 + data[i+2] * 114) >> 10;
      currLuma[j] = luma;
      const d = Math.abs(luma - lastMotionLuma[j]);
      if (d > 12) diffSum += d;
    }
    lastMotionLuma = currLuma;

    const motionPercent = Math.min(100.0, Math.round((diffSum / (160 * 90)) * 100) / 10);
    const bar = document.getElementById('barMotion');
    const lbl = document.getElementById('lblMotionVal');
    if (bar) bar.style.width = Math.min(100, motionPercent * 8) + '%';

    // State Transitions
    if (motionPercent >= MOTION_THRESHOLD) {
      // Balls or cue moving
      settleStartTime = null;
      if (autoRefereeState !== 'MOTION') {
        setAutoState('MOTION');
      }
      if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Balls Rolling...`;
    } else {
      // Calm table
      if (autoRefereeState === 'MOTION' || autoRefereeState === 'SETTLING') {
        if (!settleStartTime) {
          settleStartTime = Date.now();
          setAutoState('SETTLING');
        }
        const elapsed = Date.now() - settleStartTime;
        if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Settling: ${(elapsed/1000).toFixed(1)}s / ${(SETTLE_DELAY_MS/1000).toFixed(1)}s`;

        if (elapsed >= SETTLE_DELAY_MS) {
          // Balls have completely stopped! Trigger autonomous foul analysis
          settleStartTime = null;
          triggerAutonomousShotAnalysis();
        }
      } else {
        if (lbl) lbl.textContent = `Motion: ${motionPercent}% | Table at Rest`;
      }
    }
  } catch (err) {
    // Canvas taint or cross-origin fallback
  }
}

async function triggerAutonomousShotAnalysis() {
  setAutoState('ANALYZING');
  try {
    const nominated = document.getElementById('selNominatedBall').value;
    const res = await fetch('/api/foul/auto_process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nominated: nominated })
    });
    const data = await res.json();
    if (!data.success) {
      setAutoState('READY');
      return;
    }

    const rep = data.report;
    renderFoulReport(rep);

    if (rep.is_foul) {
      // 🚨 FOUL DETECTED: Play warning alert tone
      playRefereeTone('foul');
      setAutoState('FOUL_HOLD');
      // Update UI with laser aiming feedback
      if (data.auto_laser_aimed) {
        document.getElementById('cmdFeedback').textContent = `🚨 FOUL: Laser auto-aimed at pre-foul spot for ${data.displaced_ball_aimed} (${data.laser_x}mm, ${data.laser_y}mm)!`;
      }
    } else {
      // ✅ CLEAN SHOT: Play clean chime, post-shot automatically becomes next pre-shot!
      playRefereeTone('clean');
      const box = document.getElementById('foulVerdictBox');
      if (box) {
        box.style.display = 'block';
        box.style.background = 'rgba(34, 197, 94, 0.14)';
        box.style.border = '1px solid rgba(34, 197, 94, 0.4)';
        document.getElementById('foulVerdictIcon').textContent = '✅';
        document.getElementById('foulVerdictTitle').textContent = 'LEGAL SHOT CONFIRMED';
        document.getElementById('foulVerdictTitle').style.color = 'var(--primary-forest)';
        document.getElementById('foulPenaltyBadge').className = 'status-pill tag-online';
        document.getElementById('foulPenaltyBadge').textContent = 'NO FOUL';
        document.getElementById('foulVerdictDesc').textContent = `${rep.description} • Rolled over to next pre-shot automatically!`;
      }
      // Update badge
      updatePreShotBadge(data.post_shot_ball_count);
      setTimeout(() => {
        if (isAutoRefereeActive) {
          setAutoState('READY');
        }
      }, 1500);
    }
  } catch (e) {
    console.error('Auto referee analysis error:', e);
    setAutoState('READY');
  }
}

function resumeAutonomousAfterFoul() {
  // Referee finished repositioning displaced balls: take fresh snapshot and resume auto mode
  captureAndDetect().then(() => {
    capturePreShotState().then(() => {
      setAutoState('READY');
      alert('✅ Table restored! Autonomous referee ready for the next shot.');
    });
  });
}

function playRefereeTone(type) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === 'foul') {
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.35);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } else {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, ctx.currentTime);
      osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.12);
      osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.24);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.45);
      osc.start();
      osc.stop(ctx.currentTime + 0.45);
    }
  } catch (e) {}
}

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
    clean_info = {k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float)) else str(v)) for k, v in info.items()}
    return jsonify({"pan": round(float(pan), 1), "tilt": round(float(tilt), 1), "info": clean_info})

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
        r = call_esp32("status", timeout=5.0, retries=2)
        dt = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            return jsonify({"success": True, "message": f"Successfully connected to ESP32 at {esp32_base_url} ({dt}ms latency)"})
        return jsonify({"success": False, "message": f"HTTP {r.status_code} received from {esp32_base_url}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Cannot reach ESP32 at {esp32_base_url}. Please verify ESP32 is powered on and connected to hotspot ({e})"})

@app.route("/api/status", methods=["GET"])
def api_status():
    if simulation_mode:
        return jsonify({"connected": True, "is_simulation": True, "pan": current_pan, "tilt": current_tilt, "laser": laser_state})
    
    # Try non-blocking acquisition of lock so polling doesn't stall if a move command is active
    acquired = esp32_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({"connected": True, "is_simulation": False, "pan": current_pan, "tilt": current_tilt, "laser": laser_state})
    try:
        try:
            resp = esp32_session.get(f"{esp32_base_url.rstrip('/')}/status", timeout=2.5)
            if resp.status_code == 200:
                r = resp.json()
                return jsonify({
                    "connected": True,
                    "is_simulation": False,
                    "pan": r.get("pan", current_pan),
                    "tilt": r.get("tilt", current_tilt),
                    "laser": r.get("laser", laser_state)
                })
        except Exception:
            pass
        return jsonify({"connected": False, "is_simulation": False, "pan": current_pan, "tilt": current_tilt, "laser": laser_state})
    finally:
        esp32_lock.release()

@app.route("/api/move", methods=["POST"])
def api_move():
    global current_pan, current_tilt, laser_state, laser_state
    data = request.json or {}
    pan = float(data.get("pan", 10.0))
    tilt = float(data.get("tilt", 150.0))
    apply_trim = bool(data.get("apply_trim", True))

    cmd_pan = round(max(10.0, min(170.0, pan + (pan_trim if apply_trim else 0.0))), 1)
    cmd_tilt = round(max(10.0, min(170.0, tilt + (tilt_trim if apply_trim else 0.0))), 1)
    current_pan = cmd_pan
    current_tilt = cmd_tilt

    if simulation_mode:
        msg = f"[SIMULATION] Commanded Pan: {cmd_pan:.1f}°, Tilt: {cmd_tilt:.1f}°"
        add_log("MOVE_SIM", msg)
        return jsonify({"success": True, "message": msg, "pan": cmd_pan, "tilt": cmd_tilt})

    try:
        call_esp32("move", params={"axis": "pan", "angle": int(cmd_pan)}, timeout=3.5, retries=1)
        call_esp32("move", params={"axis": "tilt", "angle": int(cmd_tilt)}, timeout=3.5, retries=1)
        if not emergency_stopped and not laser_state:
            try:
                call_esp32("laser", params={"state": "1"}, timeout=2.5, retries=1)
                laser_state = True
            except:
                pass
        msg = f"Servos positioned to Pan: {cmd_pan:.1f}°, Tilt: {cmd_tilt:.1f}° on ESP32 ({esp32_base_url})"
        add_log("MOVE_HW", msg)
        return jsonify({"success": True, "message": msg, "pan": cmd_pan, "tilt": cmd_tilt, "laser": laser_state})
    except Exception as e:
        msg = f"ESP32 communication failure: {e}"
        add_log("ERROR", msg)
        return jsonify({"success": False, "message": msg})

@app.route("/api/trim", methods=["POST"])
def api_trim():
    global pan_trim, tilt_trim
    data = request.json or {}
    pan_trim = float(data.get("pan_trim", pan_trim))
    tilt_trim = float(data.get("tilt_trim", tilt_trim))
    add_log("TRIM", f"Global trim offset set to Pan: {pan_trim:+.1f}°, Tilt: {tilt_trim:+.1f}°")
    return jsonify({"success": True, "pan_trim": pan_trim, "tilt_trim": tilt_trim})

@app.route("/api/calib/update_node", methods=["POST"])
def api_calib_update_node():
    global interpolator, calib_data
    data = request.json or {}
    node_id = data.get("id")
    pan = float(data.get("pan"))
    tilt = float(data.get("tilt"))

    found = False
    for n in calib_data["nodes"]:
        if n["id"] == node_id:
            n["pan_deg"] = pan
            n["tilt_deg"] = tilt
            n["status"] = "calibrated"
            found = True
            break

    if found:
        with open(CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)
        interpolator = SnookerLaserInterpolator(calib_data["nodes"])
        foul_detector.interpolator = interpolator
        add_log("CALIB", f"Node {node_id} calibrated to Pan: {pan:.1f}°, Tilt: {tilt:.1f}°")
        return jsonify({"success": True, "nodes": calib_data["nodes"]})
    return jsonify({"success": False, "error": "Node not found"}), 400

@app.route("/api/center", methods=["POST"])
def api_center():
    global current_pan, current_tilt, emergency_stopped
    current_pan = 90.0
    current_tilt = 90.0
    emergency_stopped = False
    if not simulation_mode:
        try:
            call_esp32("center", timeout=3.0, retries=1)
        except Exception as e:
            add_log("WARN", f"Center command: {e}")
    add_log("SERVO", "Centered to Pan: 90°, Tilt: 90°")
    return jsonify({"success": True, "message": "Centered (90°, 90°)"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    global emergency_stopped, laser_state
    emergency_stopped = True
    laser_state = False
    if not simulation_mode:
        try:
            call_esp32("stop", timeout=2.5, retries=1)
        except Exception as e:
            pass
    add_log("E-STOP", "EMERGENCY STOP TRIGGERED")
    return jsonify({"success": True, "message": "Stopped"})

@app.route("/api/laser/toggle", methods=["POST"])
def api_laser_toggle():
    global laser_state
    laser_state = not laser_state
    if not simulation_mode:
        try:
            call_esp32("laser", params={"state": "1" if laser_state else "0"}, timeout=3.0, retries=1)
        except Exception as e:
            add_log("WARN", f"Laser toggle: {e}")
    add_log("LASER", f"Laser toggled {'ON' if laser_state else 'OFF'}")
    return jsonify({"success": True, "laser": laser_state})

@app.route("/api/laser/off", methods=["POST"])
def api_laser_off():
    global laser_state
    laser_state = False
    if not simulation_mode:
        try:
            call_esp32("laser", params={"state": "0"}, timeout=2.5, retries=1)
        except Exception as e:
            pass
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
        for ep in ["/shot.jpg", "/photo.jpg"]:
            try:
                snap_url = ip_webcam_url.rstrip("/") + ep
                resp = requests.get(snap_url, timeout=2.5)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        uploaded_frame_bgr = decoded
                        frame = decoded
                        add_log("DETECTION", f"Auto-captured live frame from {snap_url} for detection")
                        break
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

@app.route("/api/balls/detect_image", methods=["POST"])
def api_balls_detect_image():
    global uploaded_frame_bgr, uploaded_frame_b64
    t0 = time.time()
    frame = None

    if "image" in request.files:
        file = request.files["image"]
        bytes_data = file.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        if data.get("use_current_frame") and uploaded_frame_bgr is not None:
            frame = uploaded_frame_bgr.copy()
        elif data.get("sample_id"):
            sample_path = os.path.join(r"D:\Dataset", f"{data['sample_id']}.jpg")
            if os.path.exists(sample_path):
                frame = cv2.imread(sample_path)
        elif data.get("image_b64"):
            raw_b64 = data["image_b64"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            bytes_data = base64.b64decode(raw_b64)
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        if uploaded_frame_bgr is not None:
            frame = uploaded_frame_bgr.copy()
        elif ip_webcam_url:
            for ep in ["/shot.jpg", "/photo.jpg"]:
                try:
                    resp = requests.get(ip_webcam_url.rstrip("/") + ep, timeout=2.5)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        nparr = np.frombuffer(resp.content, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            break
                except Exception:
                    pass

    if frame is None:
        return jsonify({"success": False, "error": "No image provided. Please upload an image or choose a sample."}), 400

    uploaded_frame_bgr = frame
    h, w = frame.shape[:2]

    # Run detection with custom YOLOv8s model
    detected = detector.detect(frame, homography=camera_homography, conf=0.20)
    inf_ms = round((time.time() - t0) * 1000, 1)

    annotated = frame.copy()
    bgr_colors = {
        "blue ball": (225, 130, 20),
        "green ball": (40, 190, 40),
        "orange ball": (0, 120, 255),
        "pink ball": (190, 110, 245),
        "white ball": (245, 245, 245),
        "yellow ball": (25, 225, 245)
    }

    balls_list = []
    counts = {}
    for b in detected:
        counts[b.label] = counts.get(b.label, 0) + 1
        x1, y1, x2, y2 = b.bbox
        col = bgr_colors.get(b.label.lower(), (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 3)

        tag = f"{b.label} {int(b.confidence * 100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 6, th + 6)
        cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (15, 25, 20), -1)
        cv2.putText(annotated, tag, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        balls_list.append({
            "id": b.id,
            "label": b.label,
            "confidence": b.confidence,
            "bbox": [x1, y1, x2, y2],
            "center_px": [round(b.center_px[0], 1), round(b.center_px[1], 1)],
            "table_x_mm": b.table_x_mm,
            "table_y_mm": b.table_y_mm
        })

    _, buf_ann = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ann_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_ann).decode('utf-8')}"

    _, buf_raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    raw_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_raw).decode('utf-8')}"
    uploaded_frame_b64 = raw_b64

    add_log("DETECTION", f"Ball Detection Page: Found {len(balls_list)} ball(s) in {inf_ms}ms")
    return jsonify({
        "success": True,
        "width": w,
        "height": h,
        "inference_ms": inf_ms,
        "count": len(balls_list),
        "summary": counts,
        "balls": balls_list,
        "annotated_image": ann_b64,
        "raw_image": raw_b64
    })

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
        req = requests.get(stream_url, stream=True, timeout=1.2)
        content_type = req.headers.get("content-type", "multipart/x-mixed-replace; boundary=Ba4oTvQMY8ew04N8dcnM")
        def generate():
            try:
                for chunk in req.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk
            except Exception:
                pass
        return Response(generate(), content_type=content_type)
    except Exception:
        # Graceful instant fallback frame when phone webcam is switched off
        if uploaded_frame_bgr is not None:
            synth = uploaded_frame_bgr.copy()
            cv2.rectangle(synth, (0, 0), (synth.shape[1], 44), (20, 35, 25), -1)
            cv2.putText(synth, "OFFLINE MODE: PHONE WEBCAM DISCONNECTED (STATIC FRAME ACTIVE)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 140), 2)
        else:
            synth = np.zeros((360, 640, 3), dtype=np.uint8)
            synth[:] = [20, 35, 25]
            cv2.putText(synth, "PHONE WEBCAM SWITCHED OFF", (120, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 160, 140), 2)
            cv2.putText(synth, "Offline testing & 3D Simulation Active", (140, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 120, 100), 1)
        _, buf = cv2.imencode(".jpg", synth)
        return Response(buf.tobytes(), mimetype="image/jpeg")

@app.route("/api/camera/upload_base64", methods=["POST"])
def api_camera_upload_base64():
    global uploaded_frame_bgr, uploaded_frame_b64
    try:
        data = request.get_json(force=True) or {}
        raw_b64 = data.get("image_data", "")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400
        uploaded_frame_bgr = frame
        h, w = frame.shape[:2]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        add_log("CAMERA", f"Client-captured frame saved: {w}x{h} px")
        return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ipwebcam/capture", methods=["POST"])
def api_ipwebcam_capture():
    global uploaded_frame_bgr, uploaded_frame_b64, ip_webcam_url
    data = request.get_json(silent=True) or {}
    custom_url = str(data.get("url", "")).strip().rstrip("/")
    target_url = custom_url if custom_url.startswith("http") else ip_webcam_url
    if target_url:
        ip_webcam_url = target_url  # automatically sync backend URL

    # 1. Prioritize /shot.jpg (instant hardware buffer frame without autofocus hang)
    endpoints = ["/shot.jpg", "/photo.jpg"]
    last_err = None
    for ep in endpoints:
        snap_url = target_url.rstrip("/") + ep
        try:
            resp = requests.get(snap_url, timeout=2.5)
            if resp.status_code == 200 and len(resp.content) > 1000:
                nparr = np.frombuffer(resp.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    uploaded_frame_bgr = frame
                    h, w = frame.shape[:2]
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                    add_log("CAMERA", f"Captured snapshot via {ep}: {w}x{h} px")
                    return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
        except Exception as e:
            last_err = str(e)

    # 2. Fallback: Quick attempt to read a frame from OpenCV VideoCapture
    try:
        host_part = target_url.replace("http://", "").replace("https://", "").split("/")[0]
        ip_host = host_part.split(":")[0]
        port_num = int(host_part.split(":")[1]) if ":" in host_part else 8080
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.4)
        s.connect((ip_host, port_num))
        s.close()

        stream_url = target_url.rstrip("/") + "/video"
        add_log("CAMERA", f"Attempting video stream frame grab from: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                uploaded_frame_bgr = frame
                h, w = frame.shape[:2]
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                add_log("CAMERA", f"Captured frame directly from MJPEG stream: {w}x{h} px")
                return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
    except Exception as e:
        last_err = str(e)

    # Fallback: When phone is switched off, auto-load real table photo from dataset
    latest_sample = os.path.join(r"D:\Dataset", "photo58.jpg")
    if not os.path.exists(latest_sample):
        latest_sample = os.path.join(r"D:\Dataset", "photo42.jpg")
    if os.path.exists(latest_sample):
        frame = cv2.imread(latest_sample)
        if frame is not None:
            uploaded_frame_bgr = frame
            h, w = frame.shape[:2]
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            add_log("CAMERA", f"Phone offline: Loaded test frame from {os.path.basename(latest_sample)}")
            return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64, "offline_mode": True})

    return jsonify({
        "success": False,
        "error": f"Cannot connect to {target_url}. Is IP Webcam running on your phone? ({last_err})",
        "has_cached_frame": uploaded_frame_bgr is not None
    }), 500

@app.route("/api/ipwebcam/scan", methods=["POST"])
def api_ipwebcam_scan():
    global ip_webcam_url
    subnets = set()
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith('127.'):
                subnets.add('.'.join(ip.split('.')[:3]))
    except Exception:
        pass
    subnets.update(['10.18.178', '192.168.137', '192.168.43', '192.168.1', '192.168.0'])

    def probe(ip_port):
        ip, port = ip_port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.25)
        try:
            s.connect((ip, port))
            s.close()
            r = requests.get(f'http://{ip}:{port}/', timeout=1.0)
            if 'IP Webcam' in r.headers.get('server', '') or 'IP Webcam' in r.text or r.status_code == 200:
                return f'http://{ip}:{port}'
        except Exception:
            pass
        return None

    targets = []
    for subnet in subnets:
        for i in range(1, 255):
            targets.append((f'{subnet}.{i}', 8080))

    with ThreadPoolExecutor(max_workers=80) as ex:
        results = list(filter(None, ex.map(probe, targets)))

    if results:
        found_url = results[0]
        ip_webcam_url = found_url
        add_log("CAMERA", f"Auto-detected phone camera at: {found_url}")
        return jsonify({"success": True, "found": True, "url": found_url})
    else:
        return jsonify({
            "success": True,
            "found": False,
            "message": "No IP Webcam server found on network. Please ensure phone screen is awake and 'Start server' is tapped."
        })

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


# ==============================================================
# FOUL DETECTION & REFEREE REPOSITIONING ROUTES
# ==============================================================

@app.route("/api/foul/snapshot", methods=["POST"])
def api_foul_snapshot():
    data = request.json or {}
    balls = data.get("balls", detected_balls_cache)
    count = foul_detector.set_reference_state(balls)
    add_log("FOUL", f"Stored pre-shot snapshot with {count} balls")
    return jsonify({"success": True, "count": count})

@app.route("/api/foul/analyze", methods=["POST"])
def api_foul_analyze():
    data = request.json or {}
    balls = data.get("balls", detected_balls_cache)
    nominated = str(data.get("nominated", "orange"))
    report = foul_detector.analyze_shot(balls, nominated_target=nominated)
    
    rep_dict = {
        "is_foul": report.is_foul,
        "foul_type": report.foul_type,
        "penalty_points": report.penalty_points,
        "description": report.description,
        "nominated_target": report.nominated_target,
        "potted_balls": report.potted_balls,
        "cue_ball_in_off": report.cue_ball_in_off,
        "displaced_balls": [
            {
                "ball_id": db.ball_id,
                "label": db.label,
                "norm_label": db.norm_label,
                "pre_x": db.pre_x,
                "pre_y": db.pre_y,
                "post_x": db.post_x,
                "post_y": db.post_y,
                "displacement_mm": db.displacement_mm,
                "status": db.status,
                "laser_pan": db.laser_pan,
                "laser_tilt": db.laser_tilt
            }
            for db in report.displaced_balls
        ],
        "details": report.details
    }
    log_cat = "FOUL" if report.is_foul else "SHOT"
    add_log(log_cat, report.description)
    return jsonify({"success": True, "report": rep_dict})

@app.route("/api/foul/reposition", methods=["POST"])
def api_foul_reposition():
    global current_pan, current_tilt, laser_state
    data = request.json or {}
    x = float(data.get("x", 584.2))
    y = float(data.get("y", 1170.94))
    pan, tilt, _ = interpolator.predict(x, y)
    
    cmd_pan = round(max(10.0, min(170.0, pan + pan_trim)), 1)
    cmd_tilt = round(max(10.0, min(170.0, tilt + tilt_trim)), 1)
    current_pan = cmd_pan
    current_tilt = cmd_tilt

    if not simulation_mode:
        try:
            call_esp32("move", params={"axis": "pan", "angle": int(cmd_pan)}, timeout=3.5, retries=1)
            call_esp32("move", params={"axis": "tilt", "angle": int(cmd_tilt)}, timeout=3.5, retries=1)
            call_esp32("laser", params={"state": "1"}, timeout=2.5, retries=1)
            laser_state = True
        except Exception as e:
            add_log("ERROR", f"Reposition ESP32 error: {e}")
            
    add_log("REFEREE", f"Aiming laser at pre-foul spot ({x:.1f}, {y:.1f}) mm -> Pan: {cmd_pan:.1f}°, Tilt: {cmd_tilt:.1f}°")
    return jsonify({"success": True, "x": x, "y": y, "pan": cmd_pan, "tilt": cmd_tilt, "laser": True})

@app.route("/api/foul/regulation_spots", methods=["GET"])
def api_foul_regulation_spots():
    res = {}
    for k, v in REGULATION_SPOTS.items():
        pan, tilt, _ = interpolator.predict(v["x_mm"], v["y_mm"])
        res[k] = {
            **v,
            "pan": round(pan, 1),
            "tilt": round(tilt, 1)
        }
    return jsonify({"success": True, "spots": res})


def _capture_and_detect_live_frame():
    """Capture a fresh frame from IP Webcam (falling back to cached frame) and run YOLO detection."""
    global uploaded_frame_bgr, uploaded_frame_b64, detected_balls_cache
    if ip_webcam_url:
        for ep in ["/shot.jpg", "/photo.jpg"]:
            try:
                snap_url = ip_webcam_url.rstrip("/") + ep
                resp = requests.get(snap_url, timeout=2.5)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        uploaded_frame_bgr = frame
                        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                        break
            except Exception:
                pass

    if uploaded_frame_bgr is None:
        fallback_sample = os.path.join(r"D:\Dataset", "photo58.jpg")
        if os.path.exists(fallback_sample):
            frame = cv2.imread(fallback_sample)
            if frame is not None:
                uploaded_frame_bgr = frame
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    if uploaded_frame_bgr is not None:
        try:
            detected_balls = detector.detect(uploaded_frame_bgr, homography=camera_homography, conf=0.18)
            detected_balls_cache = [
                {
                    "id": b.id,
                    "label": b.label,
                    "confidence": b.confidence,
                    "table_x_mm": b.table_x_mm,
                    "table_y_mm": b.table_y_mm,
                    "pixel_x": b.center_px[0],
                    "pixel_y": b.center_px[1]
                }
                for b in detected_balls
            ]
        except Exception as e:
            add_log("ERROR", f"Detection error: {e}")

    return detected_balls_cache


def _aim_laser_at_mm(x: float, y: float):
    """Predict spline angles and aim physical laser (when not in simulation mode)."""
    global current_pan, current_tilt, laser_state
    pan, tilt, _ = interpolator.predict(x, y)
    cmd_pan = round(max(10.0, min(170.0, pan + pan_trim)), 1)
    cmd_tilt = round(max(10.0, min(170.0, tilt + tilt_trim)), 1)
    current_pan = cmd_pan
    current_tilt = cmd_tilt

    if not simulation_mode:
        call_esp32("move", params={"axis": "pan", "angle": int(cmd_pan)}, timeout=3.5, retries=1)
        call_esp32("move", params={"axis": "tilt", "angle": int(cmd_tilt)}, timeout=3.5, retries=1)
        call_esp32("laser", params={"state": "1"}, timeout=2.5, retries=1)
        laser_state = True
    return cmd_pan, cmd_tilt


def _serialize_foul_report(report):
    return {
        "is_foul": report.is_foul,
        "foul_type": report.foul_type,
        "penalty_points": report.penalty_points,
        "description": report.description,
        "nominated_target": report.nominated_target,
        "potted_balls": report.potted_balls,
        "cue_ball_in_off": report.cue_ball_in_off,
        "displaced_balls": [
            {
                "ball_id": db.ball_id,
                "label": db.label,
                "norm_label": db.norm_label,
                "pre_x": db.pre_x,
                "pre_y": db.pre_y,
                "post_x": db.post_x,
                "post_y": db.post_y,
                "displacement_mm": db.displacement_mm,
                "status": db.status,
                "laser_pan": db.laser_pan,
                "laser_tilt": db.laser_tilt
            }
            for db in report.displaced_balls
        ],
        "details": report.details
    }


manual_preshot_bgr = None


def _load_uploaded_b64_frame(raw_b64: str) -> bool:
    """Decode a data URL / base64 image into uploaded_frame_bgr and uploaded_frame_b64."""
    global uploaded_frame_bgr, uploaded_frame_b64
    if not raw_b64:
        return False
    b64_part = raw_b64.split(",", 1)[1] if "," in raw_b64 else raw_b64
    nparr = np.frombuffer(base64.b64decode(b64_part), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return False
    uploaded_frame_bgr = frame
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
    return True


def _detect_on_current_frame():
    """Run YOLO detection on uploaded_frame_bgr using current camera_homography."""
    global detected_balls_cache
    if uploaded_frame_bgr is None:
        return detected_balls_cache
    detected_balls = detector.detect(uploaded_frame_bgr, homography=camera_homography, conf=0.18)
    detected_balls_cache = [
        {
            "id": b.id,
            "label": b.label,
            "confidence": b.confidence,
            "table_x_mm": b.table_x_mm,
            "table_y_mm": b.table_y_mm,
            "pixel_x": b.center_px[0],
            "pixel_y": b.center_px[1]
        }
        for b in detected_balls
    ]
    return detected_balls_cache


@app.route("/api/foul/manual_preshot", methods=["POST"])
def api_foul_manual_preshot():
    global manual_preshot_bgr
    data = request.json or {}
    if data.get("image_b64"):
        if not _load_uploaded_b64_frame(data["image_b64"]):
            return jsonify({"success": False, "error": "Invalid uploaded Pre-Shot image."}), 400
    else:
        _capture_and_detect_live_frame()

    if uploaded_frame_bgr is not None:
        manual_preshot_bgr = uploaded_frame_bgr.copy()
        if camera_homography.H_img_to_table is None:
            if camera_homography.auto_calibrate_from_frame(uploaded_frame_bgr):
                add_log("HOMOGRAPHY", "Auto-calibrated 4-corner homography on Manual Pre-Shot")
        balls = _detect_on_current_frame()
    else:
        balls = detected_balls_cache

    count = foul_detector.set_reference_state(balls)
    homog_ok = camera_homography.H_img_to_table is not None
    add_log("REFEREE", f"Manual 2-Shot Pre-Shot captured ({count} balls, calibrated={homog_ok})")
    return jsonify({
        "success": True,
        "count": count,
        "balls": balls,
        "image_b64": uploaded_frame_b64,
        "homography_calibrated": homog_ok,
        "corners": camera_homography.corners_px
    })


@app.route("/api/foul/manual_calibrate_preshot", methods=["POST"])
def api_foul_manual_calibrate_preshot():
    global detected_balls_cache
    data = request.json or {}
    target_frame = manual_preshot_bgr if manual_preshot_bgr is not None else uploaded_frame_bgr
    if target_frame is None:
        return jsonify({"success": False, "error": "No Pre-Shot image available to calibrate."}), 400

    if data.get("auto"):
        ok = camera_homography.auto_calibrate_from_frame(target_frame)
    else:
        corners = data.get("corners", [])
        if len(corners) != 4:
            return jsonify({"success": False, "error": "Exactly 4 corners required."}), 400
        ok = camera_homography.calibrate([(float(c[0]), float(c[1])) for c in corners])

    if not ok:
        return jsonify({"success": False, "error": "Could not calibrate homography from these corners."}), 400

    detected_balls = detector.detect(target_frame, homography=camera_homography, conf=0.18)
    detected_balls_cache = [
        {
            "id": b.id,
            "label": b.label,
            "confidence": b.confidence,
            "table_x_mm": b.table_x_mm,
            "table_y_mm": b.table_y_mm,
            "pixel_x": b.center_px[0],
            "pixel_y": b.center_px[1]
        }
        for b in detected_balls
    ]
    count = foul_detector.set_reference_state(detected_balls_cache)
    add_log("HOMOGRAPHY", f"Calibrated Pre-Shot homography and re-locked {count} balls")
    return jsonify({
        "success": True,
        "count": count,
        "balls": detected_balls_cache,
        "homography_calibrated": True,
        "corners": camera_homography.corners_px
    })


@app.route("/api/foul/manual_postshot_compare", methods=["POST"])
def api_foul_manual_postshot_compare():
    if not foul_detector.pre_shot_state:
        return jsonify({
            "success": False,
            "error": "No Pre-Shot image locked yet. Please take or upload a Pre-Shot image before play."
        }), 400

    data = request.json or {}
    nominated = str(data.get("nominated", "orange"))
    if data.get("image_b64"):
        if not _load_uploaded_b64_frame(data["image_b64"]):
            return jsonify({"success": False, "error": "Invalid uploaded Post-Shot image."}), 400
        balls = _detect_on_current_frame()
    else:
        balls = _capture_and_detect_live_frame()
    report = foul_detector.analyze_shot(balls, nominated_target=nominated)

    auto_laser_aimed = False
    displaced_ball_aimed = None
    laser_x, laser_y = None, None

    # In manual foul mode, point at the first displaced/potted ball regardless of rule heuristic
    if len(report.displaced_balls) > 0:
        first_displaced = report.displaced_balls[0]
        laser_x = first_displaced.pre_x
        laser_y = first_displaced.pre_y
        displaced_ball_aimed = first_displaced.label
        try:
            _aim_laser_at_mm(laser_x, laser_y)
            auto_laser_aimed = True
            add_log("REFEREE", f"Manual Foul Restore: Laser aimed at {displaced_ball_aimed} ({laser_x}, {laser_y})")
        except Exception as e:
            add_log("ERROR", f"Manual laser aim error: {e}")

    return jsonify({
        "success": True,
        "report": _serialize_foul_report(report),
        "auto_laser_aimed": auto_laser_aimed,
        "displaced_ball_aimed": displaced_ball_aimed,
        "laser_x": laser_x,
        "laser_y": laser_y,
        "balls": balls,
        "image_b64": uploaded_frame_b64,
        "post_shot_ball_count": len(balls)
    })


@app.route("/api/foul/auto_process", methods=["POST"])
def api_foul_auto_process():
    data = request.json or {}
    nominated = str(data.get("nominated", "orange"))

    balls = _capture_and_detect_live_frame()
    report = foul_detector.analyze_shot(balls, nominated_target=nominated)

    auto_laser_aimed = False
    displaced_ball_aimed = None
    laser_x, laser_y = None, None

    if report.is_foul and len(report.displaced_balls) > 0:
        first_displaced = report.displaced_balls[0]
        laser_x = first_displaced.pre_x
        laser_y = first_displaced.pre_y
        displaced_ball_aimed = first_displaced.label
        try:
            _aim_laser_at_mm(laser_x, laser_y)
            auto_laser_aimed = True
            add_log("AUTOREFEREE", f"FOUL: Laser automatically aimed at pre-foul spot for {displaced_ball_aimed} ({laser_x}, {laser_y})")
        except Exception as e:
            add_log("ERROR", f"Auto laser aim error: {e}")

    elif not report.is_foul:
        foul_detector.rollover_to_next_preshot(balls)
        add_log("AUTOREFEREE", f"CLEAN SHOT: Rolled over {len(balls)} balls as next pre-shot baseline")

    return jsonify({
        "success": True,
        "report": _serialize_foul_report(report),
        "auto_laser_aimed": auto_laser_aimed,
        "displaced_ball_aimed": displaced_ball_aimed,
        "laser_x": laser_x,
        "laser_y": laser_y,
        "post_shot_ball_count": len(balls)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
