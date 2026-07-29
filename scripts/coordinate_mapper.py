"""
Coordinate Mapping Script.
Converts 2D image coordinates from YOLO detections to real-world 3D coordinates (mm)
using camera calibration and homography.
"""
import os
import sys
import cv2
import csv
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CALIB_DIR = BASE_DIR / "calibration" / "camera_matrix"
INPUT_IMG_DIR = BASE_DIR / "input_images"
RESULTS_DIR = BASE_DIR / "results"
MAPPING_DIR = RESULTS_DIR / "mapping"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

# Table standard dimensions (7-foot pool table play area in mm)
TABLE_WIDTH_MM = 1981.0
TABLE_HEIGHT_MM = 990.0

def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        if name == 'map_logger':
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(stdout_handler)
    return logger

logger = setup_logger('map_logger', LOGS_DIR / 'mapping.log')
error_logger = setup_logger('error_logger', LOGS_DIR / 'error.log', level=logging.ERROR)

def load_calibration() -> Tuple[np.ndarray, np.ndarray]:
    mtx_file = CALIB_DIR / "camera_matrix.npy"
    dist_file = CALIB_DIR / "dist_coeffs.npy"
    
    if not (mtx_file.exists() and dist_file.exists()):
        raise FileNotFoundError("Calibration files not found. Run Phase 5 first.")
        
    mtx = np.load(str(mtx_file))
    dist = np.load(str(dist_file))
    return mtx, dist

def compute_homography() -> np.ndarray:
    """
    Computes a mock homography matrix assuming the pool table corners 
    were located at specific pixel coordinates in the image.
    In a full production environment, these pixel coordinates are obtained
    by clicking the four corners of the table during a setup phase.
    """
    # Source points (Pixel coordinates in the image of the 4 table corners)
    # Using dummy coordinates representing a typical overhead view.
    src_pts = np.array([
        [100, 100],            # Top-Left
        [700, 100],            # Top-Right
        [700, 500],            # Bottom-Right
        [100, 500]             # Bottom-Left
    ], dtype=np.float32)

    # Destination points (Real-world coordinates in mm)
    dst_pts = np.array([
        [0, 0],                                  # Top-Left
        [TABLE_WIDTH_MM, 0],                     # Top-Right
        [TABLE_WIDTH_MM, TABLE_HEIGHT_MM],       # Bottom-Right
        [0, TABLE_HEIGHT_MM]                     # Bottom-Left
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def undistort_and_map_point(x: float, y: float, mtx: np.ndarray, dist: np.ndarray, H: np.ndarray) -> Tuple[float, float]:
    """Applies lens undistortion and then homography transformation."""
    # 1. Undistort the pixel
    pts = np.array([[[x, y]]], dtype=np.float32)
    undistorted_pts = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    
    # 2. Apply Homography
    world_pts = cv2.perspectiveTransform(undistorted_pts, H)
    wx, wy = world_pts[0][0]
    return float(wx), float(wy)

def draw_grid_and_points(img_path: Path, points: List[Dict], mtx: np.ndarray, dist: np.ndarray, H: np.ndarray):
    """Draws a visualization of the mapping process on the original image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
        
    # Undistort the image so the visual mapping aligns
    img = cv2.undistort(img, mtx, dist)
    
    # Draw table bounds based on the mock homography inverse
    H_inv = np.linalg.inv(H)
    dst_pts = np.array([
        [0, 0], [TABLE_WIDTH_MM, 0], 
        [TABLE_WIDTH_MM, TABLE_HEIGHT_MM], [0, TABLE_HEIGHT_MM]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    table_corners_px = cv2.perspectiveTransform(dst_pts, H_inv)
    cv2.polylines(img, [np.int32(table_corners_px)], True, (255, 0, 0), 2)
    
    # Draw mapped points
    for pt in points:
        cx = float(pt["Pixel X"])
        cy = float(pt["Pixel Y"])
        wx = float(pt["World X (mm)"])
        wy = float(pt["World Y (mm)"])
        
        # Undistort point for drawing
        px_u = cv2.undistortPoints(np.array([[[cx, cy]]], dtype=np.float32), mtx, dist, P=mtx)
        draw_x, draw_y = int(px_u[0][0][0]), int(px_u[0][0][1])
        
        cv2.circle(img, (draw_x, draw_y), 5, (0, 255, 255), -1)
        label = f"{pt['Class']}: ({wx:.1f}, {wy:.1f})mm"
        cv2.putText(img, label, (draw_x + 10, draw_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
    out_path = MAPPING_DIR / img_path.name
    cv2.imwrite(str(out_path), img)

def generate_report(H: np.ndarray, num_mapped: int):
    """Generates the markdown mapping report."""
    report = [
        "# Coordinate Mapping Report",
        "",
        "## Overview",
        "This report details the conversion of YOLO 2D pixel coordinates into 3D real-world coordinates (mm) on the pool table surface.",
        "",
        "## Methodology",
        "1. **Lens Undistortion:** `cv2.undistortPoints` uses the `camera_matrix` and `dist_coeffs` from Phase 5 to eliminate barrel/pincushion distortion.",
        "2. **Perspective Transformation:** A Homography matrix (`H`) is calculated mapping the 4 corners of the table in pixel space to the physical table dimensions.",
        "3. **Pixel-to-Millimeter Conversion:** `cv2.perspectiveTransform(pts, H)` executes the planar 2D-to-2D projection.",
        "",
        "## Homography Matrix",
        "```python",
        np.array2string(H, precision=6, separator=', ', suppress_small=True),
        "```",
        "",
        "## Execution Summary",
        f"- **Total Points Mapped:** {num_mapped}",
        f"- **Table Dimensions Configured:** {TABLE_WIDTH_MM}mm x {TABLE_HEIGHT_MM}mm",
        "",
        "## Limitations & Future Work",
        "- The current homography relies on a static hardcoded array of source pixel corners. In a live production environment, an interactive corner selection tool should be used.",
        "- Lens distortion assumes the camera is stationary; moving the camera requires recalibration."
    ]
    
    with open(DOCS_DIR / 'coordinate_mapping_report.md', 'w') as f:
        f.write("\n".join(report))

def main():
    logger.info("Initializing Coordinate Mapping Pipeline...")
    
    try:
        mtx, dist = load_calibration()
    except Exception as e:
        error_logger.error(f"Calibration load failed: {e}")
        return
        
    H = compute_homography()
    logger.info("Calculated Homography Matrix.")
    
    csv_in = RESULTS_DIR / "detections.csv"
    if not csv_in.exists():
        error_logger.error(f"Detection input missing: {csv_in}")
        return
        
    world_results = []
    
    # Process detections
    with open(csv_in, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cx, cy = float(row["Center X"]), float(row["Center Y"])
            wx, wy = undistort_and_map_point(cx, cy, mtx, dist, H)
            
            world_results.append({
                "Image": row["Image Name"],
                "Ball ID": row["Detection ID"],
                "Class": row["Class"],
                "Pixel X": cx,
                "Pixel Y": cy,
                "World X (mm)": wx,
                "World Y (mm)": wy
            })
            
    logger.info(f"Mapped {len(world_results)} detections to real-world coordinates.")

    # Save to CSV
    csv_out = RESULTS_DIR / "world_coordinates.csv"
    if world_results:
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=world_results[0].keys())
            writer.writeheader()
            writer.writerows(world_results)
            
    # Save to JSON
    json_out = RESULTS_DIR / "world_coordinates.json"
    with open(json_out, 'w') as f:
        json.dump(world_results, f, indent=4)
        
    # Visualizations
    logger.info("Generating visualizations...")
    # Group points by image
    img_points = {}
    for pt in world_results:
        img_name = pt["Image"]
        if img_name not in img_points:
            img_points[img_name] = []
        img_points[img_name].append(pt)
        
    for img_name, points in img_points.items():
        img_path = INPUT_IMG_DIR / img_name
        if img_path.exists():
            draw_grid_and_points(img_path, points, mtx, dist, H)
            
    # Documentation
    generate_report(H, len(world_results))
    logger.info("Coordinate Mapping Complete. Check results/mapping and docs/coordinate_mapping_report.md")

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    main()
