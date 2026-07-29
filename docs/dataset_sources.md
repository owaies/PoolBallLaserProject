# Dataset Sources

This document contains information about publicly available pool ball datasets sourced from various platforms for the AI-Based Pool Ball Identification and Laser Positioning System project.

## 1. Pool Table Balls Classification (Kaggle)
- **Dataset Name:** Pool Table Balls Classification
- **Source:** Kaggle
- **URL:** [Kaggle Dataset](https://www.kaggle.com/) (Search "Pool Table Balls Classification")
- **License:** CC0: Public Domain / Variable
- **Number of Images:** ~Variable
- **Annotation Format:** YOLO / Pascal VOC (depending on specific upload)
- **Classes:** 15 pool balls + cue ball
- **Notes:** Good variety of lighting conditions on pool tables.

## 2. Snooker/Billiards Balls Tracking (Kaggle)
- **Dataset Name:** Snooker balls tracking on video
- **Source:** Kaggle
- **URL:** [Kaggle Dataset](https://www.kaggle.com/) (Search "Snooker balls tracking")
- **License:** Unknown / Open
- **Number of Images:** Extracted from video frames
- **Annotation Format:** YOLO
- **Classes:** Snooker balls (Red, Yellow, Green, Brown, Blue, Pink, Black, Cue)
- **Notes:** Helpful for tracking motion, though classes differ slightly from 8-ball pool.

## 3. Pool Ball V4 (Roboflow Universe)
- **Dataset Name:** Pool Ball V4
- **Source:** Roboflow Universe
- **URL:** [Roboflow Universe](https://universe.roboflow.com/pocketvisions-workspace/pool-ball-v4)
- **License:** CC BY 4.0
- **Number of Images:** ~2000+
- **Annotation Format:** YOLOv8
- **Classes:** Ball, Cue ball, pockets (varies by version)
- **Notes:** Specifically annotated for YOLO models. Excellent for object detection.

## 4. Pool Balls Detection (by Mark) (Roboflow Universe)
- **Dataset Name:** Pool Balls Detection Dataset
- **Source:** Roboflow Universe
- **URL:** [Roboflow Universe](https://universe.roboflow.com/mark-dj0yk/pool-balls-detection-srlqi)
- **License:** CC BY 4.0
- **Number of Images:** ~1000+
- **Annotation Format:** YOLOv8
- **Classes:** 1-15 balls, Cue ball
- **Notes:** High-quality bounding boxes and classes mapped to individual pool ball numbers.

## 5. 8-ball-pool Dataset (Roboflow Universe)
- **Dataset Name:** 8-ball-pool
- **Source:** Roboflow Universe
- **URL:** [Roboflow Universe](https://universe.roboflow.com/project-d6t5z/8-ball-pool)
- **License:** CC BY 4.0
- **Number of Images:** 500+
- **Annotation Format:** YOLOv8
- **Classes:** Solids, Stripes, 8-ball, Cue ball
- **Notes:** Useful if class simplification (solids vs stripes) is preferred over exact numbering.

## Downloading Instructions
1. Download the datasets from the links provided.
2. Ensure the annotations are in YOLO format (normalized `[class x_center y_center width height]`).
3. Place the downloaded and extracted folders into the `datasets/raw/` directory. Do not alter the original files.
