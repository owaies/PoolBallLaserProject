# PoolBot AI Laser Calibration & Verification System

This application provides a complete, practical calibration and verification system for the final-year engineering project: **AI-Based Pool Ball Identification and Laser Positioning System**.

It enables clicking any coordinate on a virtual representation of the pool table on a laptop, calculating the required pan and tilt servo angles, commanding the ESP32 pan-tilt assembly, and verifying the physical laser spot accuracy using a phone camera.

---

## 1. Table Dimensions & Coordinate System

### Inner Playing Surface (Felt Boundary)
- **Length ($X$-axis):** `92.2 inches` = `2341.88 mm`
- **Breadth ($Y$-axis):** `46.0 inches` = `1168.40 mm`
- **Aspect Ratio:** $2341.88 / 1168.40 \approx 2.00435$ (Exact 2:1 standard pool ratio)

### Outer Wooden Rail Dimensions
- **Length:** `95.4 inches` = `2423.16 mm`
- **Breadth:** `47.7 inches` = `1211.58 mm`
- **Cushion Width ($X$):** `40.64 mm`
- **Cushion Width ($Y$):** `21.59 mm`

### Coordinate System Definition
- **Origin $(0, 0)$:** Top-Left inner corner of the green felt playing surface.
- **$+X$ Direction:** Increases along the 2341.88 mm length towards the right pockets.
- **$+Y$ Direction:** Increases along the 1168.40 mm breadth towards the bottom pockets.
- **Coordinate Bounds:** $X \in [0.0, 2341.88]\text{ mm}$, $Y \in [0.0, 1168.40]\text{ mm}$.

---

## 2. Initial Empirical Calibration Dataset (9 Samples)

The initial empirical observations recorded from the physical prototype are stored in [`calibration_points.json`](file:///d:/Final%20Year%20Project/AIPOOLBALLLASERPROJECT/laser_calibration/calibration_points.json):

| Grid Position | Table $X$ (mm) | Table $Y$ (mm) | Commanded Pan | Commanded Tilt | Description |
|---|---:|---:|---:|---:|---|
| **Top-left** | 0.00 | 0.00 | 129° | 61° | Empirical corner sample |
| **Top-centre** | 1170.94 | 0.00 | 109° | 81° | Empirical top rail side sample |
| **Top-right** | 2341.88 | 0.00 | 104° | 94° | Empirical corner sample |
| **Middle-left** | 0.00 | 584.20 | 154° | 70° | Empirical head rail midpoint sample |
| **Centre** | 1170.94 | 584.20 | 125° | 84° | Empirical table center sample |
| **Middle-right** | 2341.88 | 584.20 | 115° | 94° | Empirical foot rail midpoint sample |
| **Bottom-left** | 0.00 | 1168.40 | 167° | 79° | Empirical corner sample |
| **Bottom-centre** | 1170.94 | 1168.40 | 140° | 87° | Empirical bottom rail side sample |
| **Bottom-right** | 2341.88 | 1168.40 | 125° | 94° | Empirical corner sample |

*Note: These points are preserved as editable calibration data and are not silently assumed to be pocket centers.*

---

## 3. Verified ESP32 Communication & Firmware Endpoints

Inspected and verified against firmware source in [`esp32_laser_pantilt.ino`](file:///d:/Final%20Year%20Project/PoolBallLaserProject/firmware/esp32_laser_pantilt/esp32_laser_pantilt.ino):

- **Wi-Fi Access Point:** `PoolBot`
- **Wi-Fi Password:** `PoolBot123`
- **ESP32 IP Address:** `http://192.168.4.1`
- **Hardware Pins:**
  - `GPIO 18`: Pan servo (MG996R)
  - `GPIO 19`: Tilt servo (MG996R)
  - `GPIO 23`: Laser control via IRLZ44N MOSFET

### Supported HTTP Endpoints:
1. `GET /status`
   - Returns JSON: `{"pan": <int>, "tilt": <int>, "laser": <bool>, "stopped": <bool>}`
2. `GET /move?axis=pan&angle=<int>`
   - Commands pan angle (clamped to soft limits, e.g. 15°–175°).
3. `GET /move?axis=tilt&angle=<int>`
   - Commands tilt angle (clamped to soft limits, e.g. 25°–155°).
4. `GET /center`
   - Moves both servos to 90°, clears `emergencyStopped`, turns off laser.
5. `GET /stop`
   - Freezes servo targets at current positions and forces laser pin LOW.
   - *Important:* When stopped, `/move` calls return HTTP 423 until `/center` is called.
6. `GET /laser?state=1` / `GET /laser?state=0`
   - Sets laser gate pin HIGH/LOW. Blocked if `emergencyStopped` is active.

---

## 4. Safety & Mechanical Protection Architecture

1. **Laser OFF by Default:**
   - The laser is **NEVER** activated automatically by Python, by the UI, or during target selection.
   - Manual activation requires an explicit safety acknowledgment dialog.
2. **Simulation Mode:**
   - Clearly labeled in the UI header with a amber status pill.
   - Enabled by default so users can test clicks, angles, and homography without accidental hardware motion.
3. **Explicit Move Confirmation:**
   - In physical hardware mode, a confirmation dialog pops up showing target $(X, Y)$ and commanded angles before physical motion occurs.
4. **Configurable Soft Limits:**
   - Pan: `[15°, 175°]` (Firmware allows 10°–180°).
   - Tilt: `[25°, 155°]` (Firmware allows 10°–180°; dashboard uses 20°–160°).
5. **Emergency Stop Notice:**
   - The `/stop` route sets `emergencyStopped = true` and turns off the laser pin, holding current servo positions.
   - It **does not isolate electrical power**. Power isolation requires unplugging the battery/adapter.
6. **Thread-Safe Serial Command Execution:**
   - Uses thread locks and sequential requests to prevent overlapping, re-ordered, or stale network packets.

---

## 5. Phone-Camera Verification & Homography Workflow

To verify laser spot accuracy using phone images:

1. **Camera Setup:** Mount the phone camera in a fixed, stable position with an unobstructed view of the entire felt surface. If the phone moves, re-run corner calibration.
2. **Load Photo:** In the "Camera Verification" tab, click `1. Load Photo...` and select your phone photograph.
3. **Calibrate 4 Inner Felt Corners:**
   Click `2. Mark 4 Inner Corners` and click the 4 corners in order:
   - Corner 1: **Top-Left** $(0, 0)\text{ mm}$
   - Corner 2: **Top-Right** $(2341.88, 0)\text{ mm}$
   - Corner 3: **Bottom-Right** $(2341.88, 1168.40)\text{ mm}$
   - Corner 4: **Bottom-Left** $(0, 1168.40)\text{ mm}$
4. **Homography Projection:**
   The software computes the perspective matrix $H$ mapping image pixels $(u, v) \leftrightarrow$ physical table coordinates $(X, Y)\text{ mm}$.
5. **Measure Laser Spot:**
   Turn the laser on, photograph the table, load the image, and click `3. Click Laser Spot on Photo` (or click `Auto-Detect Red Spot`).
6. **Error Reporting:**
   For requested target $(x_r, y_r)$ and measured spot $(x_m, y_m)$:
   $$e_x = x_m - x_r$$
   $$e_y = y_m - y_r$$
   $$e = \sqrt{e_x^2 + e_y^2}$$
7. **Session Logging & Export:**
   Click `Record Spot Measurement to Session Log`. View summary stats (Mean, Median, Max, RMS) in the "Test Results & CSV" tab and export to CSV.

---

## 6. Project Architecture

```
d:\Final Year Project\AIPOOLBALLLASERPROJECT\
├── launch_calibration_app.py      # Main desktop app launcher
├── laser_calibration/             # Core Calibration Package
│   ├── __init__.py
│   ├── table_geometry.py          # Exact dimensions, boundaries, aspect ratio, canvas scaling
│   ├── calibration_data.py        # CalibrationNode CRUD, JSON serialization
│   ├── calibration_points.json    # Initial 9 empirical observations
│   ├── interpolator.py            # Bilinear grid, Scattered IDW, Geometric model
│   ├── esp32_client.py            # HTTP client, soft limits, safety checks, simulation mode
│   ├── camera_homography.py       # 4-point perspective transform & red spot detector
│   ├── spot_measurement.py        # Error metrics (ex, ey, e, RMS, Mean), CSV export
│   ├── app_gui.py                 # Tkinter GUI application
│   ├── demo_simulation.py         # End-to-end simulation demonstration script
│   └── requirements.txt           # Dependencies
└── tests/
    └── test_laser_calibration.py  # 21 Unit tests covering all modules
```

---

## 7. Installation & Quick Start (Windows)

### Requirements
- Python 3.10+ (Tested on Python 3.11)
- Packages: `numpy`, `opencv-python`, `Pillow`, `requests`

### Install Dependencies
```powershell
pip install -r laser_calibration/requirements.txt
```

### Run Unit Tests
```powershell
python tests/test_laser_calibration.py
```
*(All 21 tests should pass in under 0.1s)*

### Run Simulation Demonstration
```powershell
python laser_calibration/demo_simulation.py
```

### Launch Desktop Calibration Application
```powershell
python launch_calibration_app.py
```

---

## 8. Physical Measurements Still Needed

The following physical measurements are currently provisional and must be verified on the actual hardware:

1. **Plate-to-Pivot Height:** Distance from the underside of the mounting plate to the center of rotation of the pan servo pivot.
2. **Mounting Pivot 3D Offsets:** Exact physical horizontal $(X, Y)$ and vertical $(Z)$ distance from the $(0, 0)$ inner felt corner to the pan pivot axis.
3. **Tilt Axis Offset:** Vertical and horizontal offset between the pan rotation axis and the tilt rotation axis.
4. **Laser Optical Axis Alignment:** Distance and angle between the laser diode center beam and the tilt servo mounting horn.
5. **Driver Circuit Electrical Check:** Verify IRLZ44N gate voltage and diode current draw under full battery/adapter load before prolonged laser activation.

---

## 9. Statement of Testing: Simulation vs. Physical Hardware

- **Tested in Simulation:**
  - Table coordinate conversion, aspect ratio scaling, and felt boundary clamping.
  - Piecewise Bilinear Grid angle interpolation against the 9 empirical nodes and quadrant midpoints.
  - Scattered 2D IDW and Geometric kinematic model predictions.
  - ESP32 client state management, soft limit clamping, confirmation blocking, and emergency stop logic.
  - Mocked HTTP requests verifying exact query strings for all ESP32 routes (`/status`, `/move`, `/center`, `/stop`, `/laser`).
  - Perspective homography forward and inverse coordinate mapping and synthetic red spot detection.
  - Spot measurement error calculations ($e_x, e_y, e$, Mean, Median, Max, RMS) and CSV export.
  - Tkinter GUI rendering, canvas reticle interactions, node editing, and tab transitions.

- **To Be Completed on Physical Hardware by User:**
  - Establishing connection to the live `PoolBot` Wi-Fi AP (`192.168.4.1`) from the laptop.
  - Physical servo motion verification under battery power.
  - Taking actual phone photos of the pool table and selecting physical felt corners.
  - Physical laser spot error measurement across multiple table targets.
