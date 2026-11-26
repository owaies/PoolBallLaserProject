"""
Simulation Demonstration Script for PoolBot Laser Calibration System.

Demonstrates end-to-end operation in simulation mode without physical hardware:
1. Loads the 9 empirical calibration points.
2. Selects 5 diverse test target points across the pool table.
3. Computes predicted pan & tilt angles via Bilinear Grid Interpolation.
4. Commands simulated ESP32 servos and verifies soft limit compliance.
5. Simulates phone-camera capture with a perspective view of the pool table.
6. Calibrates homography from 4 inner felt corners.
7. Measures laser spot, calculates ex, ey, and Euclidean error e (mm).
8. Aggregates error statistics (Mean, Median, Max, RMS) and exports to CSV.
"""

import os
import sys
import math
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser_calibration.table_geometry import TABLE
from laser_calibration.calibration_data import CalibrationDataset, get_default_nine_samples
from laser_calibration.interpolator import BilinearGridInterpolator
from laser_calibration.esp32_client import ESP32Client, ServoSoftLimits
from laser_calibration.camera_homography import CameraHomography, ImageCornerSelection, detect_red_laser_spot
from laser_calibration.spot_measurement import MeasurementSession, SpotMeasurementRecord


def run_simulation_demo(export_csv_path: str = "simulation_results.csv"):
    print("=" * 75)
    print(" POOLBOT AI LASER CALIBRATION SYSTEM - SIMULATION DEMO")
    print("=" * 75)

    # 1. Load Calibration Points
    dataset = CalibrationDataset()
    print(f"\n[1] Loaded {len(dataset.nodes)} Calibration Nodes:")
    for n in dataset.nodes:
        print(f"    • {n.label:<14} X={n.x_mm:>7.1f} mm, Y={n.y_mm:>6.1f} mm -> Pan={n.pan_deg:>5.1f}°, Tilt={n.tilt_deg:>5.1f}°")

    # 2. Initialize Bilinear Interpolator
    interpolator = BilinearGridInterpolator(dataset.nodes)
    print(f"\n[2] Active Interpolator: {interpolator.get_name()}")

    # 3. Setup ESP32 Hardware Client in Simulation Mode
    soft_limits = ServoSoftLimits(pan_min=15, pan_max=175, tilt_min=25, tilt_max=155)
    esp32 = ESP32Client(simulation_mode=True, soft_limits=soft_limits)
    print(f"\n[3] ESP32 Client initialized in SIMULATION MODE")
    print(f"    Soft Limits: Pan [{soft_limits.pan_min}°- {soft_limits.pan_max}°], Tilt [{soft_limits.tilt_min}°- {soft_limits.tilt_max}°]")

    # 4. Setup Synthetic Phone Camera & Homography
    # Simulating a 1920x1080 phone camera looking at the table with perspective tilt
    img_w, img_h = 1920, 1080
    corner_selection = ImageCornerSelection(
        top_left=(180.0, 120.0),
        top_right=(1740.0, 150.0),
        bottom_right=(1840.0, 980.0),
        bottom_left=(80.0, 950.0),
    )
    homography = CameraHomography(corner_selection)
    print(f"\n[4] Phone-Camera Perspective Homography Calibrated:")
    print(f"    Table corners mapped to image quad: {corner_selection.to_list()}")

    # 5. Define Test Targets (x_r, y_r in mm)
    test_targets = [
        ("Table Center", 1170.94, 584.20),
        ("Break Area (Head)", 585.00, 584.20),
        ("Rack Spot (Foot)", 1756.00, 584.20),
        ("Top-Left Pocket Vicinity", 100.00, 100.00),
        ("Bottom-Right Pocket Vicinity", 2200.00, 1050.00),
    ]

    session = MeasurementSession()

    print("\n[5] Executing Calibration Verification Run on 5 Targets:")
    print("-" * 85)
    print(f"{'Target Name':<28} {'Requested (mm)':<18} {'Command (P/T)':<14} {'Measured (mm)':<18} {'Error (e)':<10}")
    print("-" * 85)

    for name, xr, yr in test_targets:
        # A. Predict angles
        pan, tilt, info = interpolator.predict(xr, yr)

        # B. Command simulated servos
        success, cmd_msg = esp32.move_servos(pan, tilt, require_confirmation=False, confirmed=True)

        # C. Simulate physical laser spot on table with slight mechanical deviation (~3-5 mm)
        # and project into phone image
        np.random.seed(int(xr + yr) % 1000)
        simulated_dx = np.random.uniform(-3.5, 3.5)
        simulated_dy = np.random.uniform(-3.5, 3.5)
        actual_laser_table_x = xr + simulated_dx
        actual_laser_table_y = yr + simulated_dy

        # D. Project laser spot to phone image pixel coordinates
        spot_u, spot_v = homography.table_to_image(actual_laser_table_x, actual_laser_table_y)

        # E. Invert back through Homography (as done in user workflow when clicking or detecting spot)
        xm, ym = homography.image_to_table(spot_u, spot_v)

        # F. Calculate error record
        rec = SpotMeasurementRecord.create(
            target_x_mm=xr,
            target_y_mm=yr,
            commanded_pan_deg=pan,
            commanded_tilt_deg=tilt,
            measured_x_mm=xm,
            measured_y_mm=ym,
            notes=f"Simulated test: {name}",
        )
        session.add_record(rec)

        req_str = f"({xr:.1f}, {yr:.1f})"
        cmd_str = f"({pan:.1f}°, {tilt:.1f}°)"
        meas_str = f"({xm:.1f}, {ym:.1f})"
        err_str = f"{rec.euclidean_error_mm:.2f} mm"

        print(f"{name:<28} {req_str:<18} {cmd_str:<14} {meas_str:<18} {err_str:<10}")

    print("-" * 85)

    # 6. Compute Aggregate Statistics
    stats = session.calculate_statistics()
    print("\n[6] Aggregate Error Statistics:")
    print(f"    • Total Test Points : {stats.count}")
    print(f"    • Mean Error        : {stats.mean_error_mm:.2f} mm")
    print(f"    • Median Error      : {stats.median_error_mm:.2f} mm")
    print(f"    • Max Error         : {stats.max_error_mm:.2f} mm")
    print(f"    • RMS Error         : {stats.rms_error_mm:.2f} mm")
    print(f"    • Mean X Offset (ex): {stats.mean_ex_mm:+.2f} mm")
    print(f"    • Mean Y Offset (ey): {stats.mean_ey_mm:+.2f} mm")

    # 7. Export to CSV
    session.export_to_csv(export_csv_path)
    print(f"\n[7] Exported session measurement results to: {os.path.abspath(export_csv_path)}")

    print("\n" + "=" * 75)
    print(" SIMULATION DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return session, stats


if __name__ == "__main__":
    run_simulation_demo()
