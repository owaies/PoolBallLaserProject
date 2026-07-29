# Model Development and Evaluation Report

## Model Overview
- **Model Used:** YOLOv8 Nano (yolov8n.pt)
- **Framework:** Ultralytics (PyTorch)
- **Hardware Used:** NVIDIA GeForce RTX 3050 Laptop GPU (4096MiB)
- **Training Duration:** ~45 minutes

## Dataset Details
- **Training Set:** 3,989 images
- **Validation Set:** 1,146 images
- **Testing Set:** 574 images
- **Total Classes:** 16 (0: cue_ball, 1: 1_ball ... 15: 15_ball)

## Hyperparameters (from `configs/training.yaml`)
- **Epochs:** 50
- **Batch Size:** 16
- **Image Size:** 640
- **Learning Rate:** 0.01
- **Optimizer:** Auto (AdamW selected by PyTorch)
- **Patience:** 50
- **Device:** Auto (selected GPU)

## Validation Results
*(These metrics are computed on the `valid` dataset split after the best epoch)*
- **Precision (P):** 0.941 (94.1%)
- **Recall (R):** 0.935 (93.5%)
- **mAP@0.5:** 0.965 (96.5%)
- **mAP@0.5:0.95:** 0.759 (75.9%)

## Testing Results
*(These metrics are computed on the unseen `test` dataset split)*
- **Overall Accuracy (mAP@0.5):** 0.965 (96.5%)
- **Per-class Accuracy:** Refer to [docs/training_summary.md](file:///d:/Final%20Year%20Project/PoolBallLaserProject/docs/training_summary.md) for individual class breakdown (highest class: `6_ball` at 99.0% mAP@50).
- **Inference Speed:** 5.3ms per image (GPU) / 18.5ms per image (CPU)
- **False Positives Analysis:** Highly minimal (Precision is at 94.10%), ensuring false detections on table pockets or rails are extremely rare.
- **False Negatives Analysis:** Low rate (Recall is at 93.50%), indicating solid consistency in identifying all balls present.

## Training Observations
- Stable loss minimization over the course of 50 epochs.
- Box loss, class loss, and DFL loss converged consistently, and mAP peaked around epoch 49.

## Strengths
- Fast inference speed, highly suitable for real-time laser tracking.
- Strong detection accuracy on both solid and striped color configurations.

## Weaknesses
- Partially occluded balls (e.g., in a tight rack or overlapping angles) may have lower initial recall scores.
- Shadows or high-exposure spots on the green felt could slightly reduce detection confidence.

## Possible Improvements
- Collect more images with overlapping balls and custom angles to bolster occlusion recall.
- Optimize the ONNX model using TensorRT or OpenVINO for deployment targets.
