"""
Script to capture checkerboard images for camera calibration.
"""
import os
import cv2
import argparse
import logging
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CALIBRATION_DIR = BASE_DIR / "calibration"
IMAGES_DIR = CALIBRATION_DIR / "images"
LOGS_DIR = BASE_DIR / "logs"

def setup_logger(name: str, log_file: Path) -> logging.Logger:
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    return logger

logger = setup_logger('capture_logger', LOGS_DIR / 'calibration.log')
error_logger = setup_logger('capture_error_logger', LOGS_DIR / 'error.log')

def main(num_images: int, camera_id: int):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        error_logger.error(f"Cannot open camera ID {camera_id}.")
        return

    logger.info(f"Press 'c' to capture an image. Press 'q' to quit early. Target: {num_images} images.")
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            error_logger.error("Failed to grab frame.")
            break
            
        cv2.imshow('Calibration Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            img_path = IMAGES_DIR / f"calib_{count:03d}.jpg"
            cv2.imwrite(str(img_path), frame)
            logger.info(f"Captured {img_path.name} ({count+1}/{num_images})")
            count += 1
        elif key == ord('q'):
            logger.info("Capture aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Capture session ended. Total captured: {count}/{num_images}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture checkerboard images for calibration.")
    parser.add_argument("--num_images", type=int, default=20, help="Number of images to capture")
    parser.add_argument("--camera_id", type=int, default=0, help="USB Camera ID")
    args = parser.parse_args()
    main(args.num_images, args.camera_id)
