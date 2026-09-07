# AI-Based Pool Ball Identification and Laser Positioning System

## Project Overview
This project leverages an Ultralytics YOLO AI object detection model and a USB camera mounted above a pool table to detect pool balls in real time. The ultimate goal is to calculate the center coordinates of a selected ball, convert those coordinates into physical real-world coordinates (millimeters), and send them to an ESP32 microcontroller. The ESP32 will then control a pan-tilt servo mechanism to aim a laser pointer accurately at the ball's position.

**Current Status:** The software architecture is complete up to **Phase 6 (Coordinate Mapping)**. All AI detection, camera calibration, and geometric homography transformations are fully operational. Hardware integration (ESP32/servos) is reserved for future phases.

---

## Daily Progress Log

### 2026-08-27
- Reviewed and documented the current Phase 6 coordinate-mapping milestone.
- Confirmed the README reflects the current project architecture and remaining hardware-integration work.

### 2026-08-28
- Maintained the project documentation and verified that the README's progress log remains aligned with the current Phase 6 software status.
- Recorded the remaining ESP32 and servo hardware integration as future work rather than marking it complete prematurely.

### 2026-08-29
- Reviewed the documented Phase 6 pipeline and kept the project status aligned with the current software milestone.
- Preserved the ESP32 and servo integration as future work until hardware implementation is actually completed.

### 2026-08-30
- Reviewed the README progress history and kept the documented project status consistent with the existing Phase 6 software milestone.
- Continued to distinguish completed coordinate-mapping work from the planned ESP32 and servo hardware integration.

### 2026-08-31
- Updated the daily documentation log while keeping the project status aligned with the existing Phase 6 coordinate-mapping milestone.
- Kept ESP32 and servo hardware integration documented as future work until it is actually implemented.

### 2026-09-01
- Reviewed the documented Phase 6 workflow and kept the README aligned with the repository's current software milestone.
- Continued documenting hardware integration as future work rather than claiming unimplemented ESP32 or servo functionality.

### 2026-09-02
- Reviewed the existing project documentation and confirmed that the Phase 6 coordinate-mapping milestone remains the latest documented software stage.
- Kept the planned ESP32 and servo integration clearly separated from completed functionality.

### 2026-09-03
- Reviewed the current README and confirmed that Phase 6 coordinate mapping remains the latest documented software milestone.
- Kept the project status accurate by retaining ESP32 and servo integration as future hardware work.

### 2026-09-04
- Reviewed the project documentation and confirmed that Phase 6 coordinate mapping remains the latest documented software milestone.
- Kept the README status accurate by leaving ESP32 and servo integration marked as future hardware work.

### 2026-09-05
- Reviewed the documented Phase 6 pipeline and confirmed that coordinate mapping remains the latest completed software milestone.
- Kept planned ESP32 and servo hardware integration clearly separated from the implemented software functionality.

### 2026-09-06
- Reviewed the current project documentation and confirmed that Phase 6 coordinate mapping remains the latest documented software milestone.
- Kept the README aligned with the repository state without claiming ESP32 or servo hardware integration as completed.

### 2026-09-07
- Reviewed the README against the current Phase 6 documentation and kept the latest software milestone accurately recorded.
- Kept ESP32 and servo integration documented as future work until hardware implementation is present.

---

## Folder Structure

```
PoolBallLaserProject/
│
├── calibration/          # Phase 5: Camera Calibration
│   ├── camera_matrix/    # Saved intrinsic arrays (camera_matrix.npy, dist_coeffs.npy)
│   ├── docs/             # Calibration reports
│   ├── images/            # Raw checkerboard images for calibration
│   └── scripts/           # Calibration logic (capture, calibrate, mock generator)
│
├── configs/              # Phase 3: Project configurations
│   ├── dataset.yaml      # YOLO dataset definitions
│   └── training.yaml     # Configurable hyperparameters for training
│
├── datasets/             # Phase 1 & 2: Image data and labels
│   ├── raw/, merged/, cleaned/, train/, valid/, test/
│
├── docs/                 # Documentation and reports for all phases
│   ├── dataset_report.md
│   ├── dataset_sources.md
│   ├── detection_report.md
│   ├── coordinate_mapping_report.md
│   ├── training_summary.md
│   └── project_notes.md
│
├── input_images/         # Phase 4: Images to be processed by offline pipeline
├── output_images/        # Phase 4: Annotated output images with bounding boxes
│
├── logs/                 # Output logs from all pipelines
│
├── models/               # Phase 3: Saved checkpoints
│   ├── best.pt           # Best performing YOLO model
│   └── last.pt           # Most recent checkpoint
│
├── results/              # Phase 3, 4, 6: Evaluation charts, metrics, and mappings
│   ├── mapping/          # Grid overlay visualizations of coordinate mapping
│   ├── detections.csv    # Raw offline pixel detections
│   ├── world_coordinates.csv # Final real-world millimeter coordinates
│   └── ...               # Loss curves, PR curves, Confusion Matrix
│
├── scripts/              # Python logic scripts
│   ├── generate_synthetic_data.py
│   ├── merge_dataset.py, remove_duplicates.py, verify_labels.py, split_dataset.py
│   ├── train.py          # Main YOLO training script
│   ├── detect.py         # Offline detection pipeline
│   └── coordinate_mapper.py # Pixel to Real-world mapping script
│
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Installation & Setup

1. Ensure you have **Python 3.12** installed.
2. It is highly recommended to use a virtual environment.
3. Run the following command in the root directory to install all dependencies (including Ultralytics, OpenCV, NumPy, etc.):
```bash
pip install -r requirements.txt
pip install pyyaml
```

---

## Pipeline Execution Guide

### 1. Data Preparation (Phases 1 & 2)
*(Assuming you have downloaded data into `datasets/raw/` or run the synthetic generator)*
Run the following sequentially to merge, clean, split, and summarize the dataset:
```bash
python scripts/generate_synthetic_data.py # (Optional) Use for testing without real data
python scripts/merge_dataset.py
python scripts/remove_duplicates.py
python scripts/verify_labels.py
python scripts/split_dataset.py
python scripts/dataset_summary.py
```

### 2. Model Development & Training (Phase 3)
Configure training parameters directly in `configs/training.yaml` (e.g., set epochs to 100).
To begin training the YOLO model on your dataset:
```bash
python scripts/train.py
```
This script handles training, validation, testing on unseen holdout data, and automatically exports the final weights to `models/best.pt`, `models/best.onnx`, and `models/best.torchscript`.

### 3. Offline Detection (Phase 4)
To process images autonomously using the trained model:
1. Place raw pool table images into the `input_images/` directory.
2. Run the detection script:
```bash
python scripts/detect.py
```
Outputs (annotated images) will be saved to `output_images/`, and the raw pixel coordinate metadata will be saved to `results/detections.csv` and `results/detections.json`.

### 4. Camera Calibration (Phase 5)
To ensure physical geometry is accurate, the camera must be calibrated to remove lens distortion.
1. Capture checkerboard images using your webcam:
```bash
python calibration/scripts/capture_calibration.py
```
2. Compute the camera matrix and distortion coefficients:
```bash
python calibration/scripts/calibrate_camera.py
```
The intrinsic parameters will be saved to `calibration/camera_matrix/` for use in Phase 6.

### 5. Coordinate Mapping (Phase 6)
To map the 2D YOLO pixel coordinates from Phase 4 into 3D real-world physical coordinates (millimeters) on the pool table:
```bash
python scripts/coordinate_mapper.py
```
This script applies lens undistortion, calculates a perspective homography matrix based on the table's physical dimensions (1981mm x 990mm), and projects the detections into standard measurements. 
The final values are saved to `results/world_coordinates.csv` and `results/world_coordinates.json`, perfectly poised for serial transmission to the ESP32 in future phases! Visualizations are available in `results/mapping/`.
