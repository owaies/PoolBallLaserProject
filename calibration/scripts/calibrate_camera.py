"""
Camera calibration script using checkerboard patterns.
Calculates and exports the camera matrix and distortion coefficients.
"""
import os
import sys
import cv2
import yaml
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CALIB_DIR = BASE_DIR / "calibration"
IMAGES_DIR = CALIB_DIR / "images"
MATRIX_DIR = CALIB_DIR / "camera_matrix"
RESULTS_DIR = CALIB_DIR / "results"
DOCS_DIR = CALIB_DIR / "docs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [MATRIX_DIR, RESULTS_DIR, DOCS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logger setup
def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        if name == 'calib_logger':
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(stdout_handler)
    return logger

logger = setup_logger('calib_logger', LOGS_DIR / 'calibration.log')
error_logger = setup_logger('error_logger', LOGS_DIR / 'error.log', level=logging.ERROR)

def calibrate_camera(checkerboard_size=(9, 6), square_size=24.0):
    """
    Calibrates the camera.
    checkerboard_size: Tuple (inner corners per row, inner corners per column).
    square_size: Size of a square in your defined unit (e.g., mm).
    """
    logger.info("Starting camera calibration process...")
    
    # termination criteria for corner subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    image_files = list(IMAGES_DIR.glob('*.jpg')) + list(IMAGES_DIR.glob('*.png'))
    if not image_files:
        error_logger.error(f"No images found in {IMAGES_DIR}")
        return

    logger.info(f"Found {len(image_files)} images for calibration.")

    img_shape = None
    successful_images = 0

    for img_path in tqdm(image_files, desc="Finding corners"):
        img = cv2.imread(str(img_path))
        if img is None:
            error_logger.warning(f"Could not read {img_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_images += 1
        else:
            logger.warning(f"Checkerboard not found in {img_path.name}")

    if successful_images == 0:
        error_logger.error("No checkerboards found in any images. Calibration failed.")
        return

    logger.info(f"Successfully found corners in {successful_images}/{len(image_files)} images. Computing matrices...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Compute Reprojection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    logger.info(f"Calibration successful! Mean reprojection error: {mean_error:.4f}")

    # Save Arrays
    np.save(str(MATRIX_DIR / 'camera_matrix.npy'), mtx)
    np.save(str(MATRIX_DIR / 'dist_coeffs.npy'), dist)
    
    # Save YAML
    yaml_data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.tolist(),
        'mean_reprojection_error': float(mean_error),
        'image_resolution': list(img_shape)
    }
    with open(MATRIX_DIR / 'camera_parameters.yaml', 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    logger.info(f"Saved calibration data to {MATRIX_DIR}")

    # Validation (Undistort a sample image)
    sample_img_path = image_files[0]
    img = cv2.imread(str(sample_img_path))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w_roi, h_roi = roi
    # dst = dst[y:y+h_roi, x:x+w_roi] # Optional cropping
    
    cv2.imwrite(str(RESULTS_DIR / 'original.jpg'), img)
    cv2.imwrite(str(RESULTS_DIR / 'undistorted.jpg'), dst)
    
    # Difference comparison
    diff = cv2.absdiff(img, dst)
    cv2.imwrite(str(RESULTS_DIR / 'difference.jpg'), diff)
    logger.info("Saved validation sample images to results/")

    # Generate Markdown Report
    report_paths = [
        DOCS_DIR / 'camera_calibration_report.md',
        BASE_DIR / 'docs' / 'camera_calibration_report.md'
    ]
    
    for report_path in report_paths:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write("# Camera Calibration Report\n\n")
            f.write("## Settings\n")
            f.write(f"- **Checkerboard Inner Corners:** {checkerboard_size}\n")
            f.write(f"- **Square Size:** {square_size} mm\n")
            f.write(f"- **Images Attempted:** {len(image_files)}\n")
            f.write(f"- **Images Used (Valid Corners):** {successful_images}\n\n")
            
            f.write("## Calibration Results\n")
            f.write(f"- **Mean Reprojection Error:** `{mean_error:.4f}` pixels\n\n")
            
            f.write("### Camera Matrix\n```python\n")
            f.write(np.array2string(mtx, precision=4, separator=', ', suppress_small=True))
            f.write("\n```\n\n")
            
            f.write("### Distortion Coefficients\n```python\n")
            f.write(np.array2string(dist, precision=4, separator=', ', suppress_small=True))
            f.write("\n```\n\n")
            
            f.write("## Recommendations\n")
            if mean_error < 0.5:
                f.write("- Calibration error is excellent (< 0.5px). Proceed to coordinate transformation.\n")
            elif mean_error < 1.0:
                f.write("- Calibration error is acceptable. You may improve it by capturing more diverse angles.\n")
            else:
                f.write("- Calibration error is high (> 1.0px). It is strongly recommended to capture a new set of checkerboard images ensuring the board covers different regions of the camera view and has no motion blur.\n")
        logger.info(f"Generated report at {report_path}")

if __name__ == '__main__':
    os.chdir(BASE_DIR)
    calibrate_camera()
