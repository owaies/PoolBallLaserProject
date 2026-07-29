# Dataset Quality and Preparation Report

This report summarizes the overall dataset collection and preparation phase for the AI-Based Pool Ball Identification and Laser Positioning System.

## Dataset Summary Metrics
*(Note: These are placeholder values. Run `scripts/dataset_summary.py` after loading actual data to get exact numbers.)*

- **Total images collected (Raw):** TBD (Depends on downloaded sets)
- **Total images after cleaning:** TBD
- **Total labels:** TBD
- **Final classes:** 16 (0: cue_ball, 1: 1_ball, ..., 15: 15_ball)

## Dataset Quality Observations
- **Lighting Variation:** The sourced datasets from Kaggle and Roboflow include varying lighting conditions. Shadows on the pool table might cause false negatives or false positives (e.g., mistaking chalk for a ball).
- **Occlusion:** Some images contain partially occluded balls (blocked by cues or hands). The YOLO model should be robust if sufficient occluded examples are in the training set.
- **Class Imbalance:** It is highly likely that certain balls (like the 8-ball and Cue ball) will appear more frequently across datasets than others. This will be verified in `data_statistics.md`.

## Remaining Issues & Next Steps
- **Class Remapping:** Different datasets use different class IDs (e.g., one dataset might use 0-15, another might use `0: cue`, `1: solid`, `2: stripe`). Before training, a custom script may be needed to unify the class mappings if combining heterogenous datasets.
- **Hardware Integration Dataset:** Currently, all data is sourced online. In later phases, we will need to capture images using the actual USB camera mounted on the physical rig to fine-tune the model for the specific deployment environment.
- **Bounding Box Tightness:** Relying on community datasets means bounding box quality varies. Some labels may need manual adjustment using a tool like LabelImg or Roboflow if accuracy drops.
