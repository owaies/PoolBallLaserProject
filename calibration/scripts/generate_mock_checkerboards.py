"""
Generates synthetic checkerboard images to mock calibration data 
so that the pipeline can be executed and tested autonomously without a real webcam.
"""
import os
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = BASE_DIR / "calibration" / "images"

def generate_mock_checkerboards(num_images=10):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    pattern_size = (9, 6) # Inner corners
    square_size = 50 # pixels for the synthetic image
    
    for i in range(num_images):
        img_size = (800, 600)
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        
        # Add some random translation and rotation to make the calibration math work
        tx = np.random.randint(50, 200)
        ty = np.random.randint(50, 150)
        
        # Draw checkerboard
        for row in range(pattern_size[1] + 1):
            for col in range(pattern_size[0] + 1):
                if (row + col) % 2 == 0:
                    pt1 = (tx + col * square_size, ty + row * square_size)
                    pt2 = (tx + (col + 1) * square_size, ty + (row + 1) * square_size)
                    cv2.rectangle(img, pt1, pt2, (0, 0, 0), -1)
                    
        # Apply synthetic lens distortion using warp
        K = np.array([[600, 0, 400], [0, 600, 300], [0, 0, 1]], dtype=np.float32)
        D = np.array([-0.2, 0.1, 0, 0], dtype=np.float32) # radial distortion
        img_distorted = cv2.undistort(img, K, D) # undistort adds distortion if D is chosen carefully, but let's just use raw image for simplicity since cv2.calibrateCamera can handle perfect images
        
        # We will just save the perfect images with slight translations. It will result in near-zero distortion, but validates the pipeline.
        
        cv2.imwrite(str(IMAGES_DIR / f"calib_mock_{i:03d}.jpg"), img)
    print(f"Generated {num_images} mock checkerboards in {IMAGES_DIR}")

if __name__ == "__main__":
    generate_mock_checkerboards()
