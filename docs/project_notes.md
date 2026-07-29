# Project Notes

## Phase 1 & 2 Constraints
- The current repository setup strictly handles dataset sourcing, merging, cleaning, and preparation.
- Model training, inference, and hardware integration (ESP32) are explicitly excluded from this phase.

## Dependencies
Ensure the following Python packages are installed before running the preparation scripts:
- `tqdm` (for progress bars)
- `Pillow` (for image validation)

Run: `pip install tqdm pillow`

## Future Development Hooks
- **Hardware (ESP32):** In future phases, coordinate extraction from YOLO will be translated to servo motor angles.
- **Laser Calibration:** The camera-to-table coordinate system will require a homography transformation matrix to map 2D image coordinates to physical motor angles.
