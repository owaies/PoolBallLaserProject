# Dataset Improvement Report: False Positive Reduction

This report documents the collection and validation of negative (background) and hard-negative (confusable circular objects) images added to the dataset of the **AI-Based Pool Ball Identification and Laser Positioning System**.

Adding these background examples is a critical step in training a robust YOLO model that generalizes well and minimizes false detections.

---

## 1. Why Background/Negative Images Matter

In object detection, training a model only on positive examples (images containing the target objects) often leads to a high rate of false positives. Without negative examples, the model learns features that correlate with the objects but are actually part of the background, or it fails to distinguish target objects from similar non-target objects.

Ultralytics YOLO natively supports background images. By introducing images without any annotations (0-byte `.txt` label files):
1. **Felt and Table Context:** The model learns that empty pool table cloth, pockets, cue sticks, table edges, and bar room interiors are not pool balls.
2. **Circular Distractors (Hard Negatives):** By presenting tennis balls, golf balls, cricket balls, fruits, and coins, the model learns that circular shape and roundness alone are not sufficient to classify an object as a pool ball.
3. **Loss Function Adjustment:** Background images are fed into the training pipeline to adjust the objectness loss, penalizing the model for detecting background objects as region proposals.

---

## 2. Dataset Composition

A total of **1,840 images** were collected and verified:

| Category | Description / Subcategories | Sourced / Generated | Count | YOLO Annotation |
|---|---|---|---|---|
| **Negatives** (Backgrounds) | Empty pool tables, felt cloth, table cushions, pockets, floors, walls, scoreboards, room scenes, ceiling lights. | Unsplash, PIL Synthetic, Kaggle (Scene-15) | **737** | Empty 0-byte `.txt` file |
| **Hard Negatives** (Circular Distractors) | Tennis balls, golf balls, cricket balls, ping pong balls, baseballs, basketballs, soccer balls, apples, oranges, lemons, limes, coins. | Unsplash, PIL Synthetic, Kaggle (Sports-Balls, Fruits-360, Count-Coins) | **1,103** | Empty 0-byte `.txt` file |
| **Total** | | | **1,840** | |

---

## 3. Data Sources & Provenance

All external datasets were downloaded programmatically via the Kaggle API and processed locally:

| Dataset / Source | Category | Original License | Description of Use |
|---|---|---|---|
| **Kaggle (yiklunchow/scene15)** | Negatives | CC0: Public Domain | Indoor scenes (offices, kitchens, living rooms, etc.) representing general environments. |
| **Kaggle (samuelcortinhas/sports-balls...)** | Hard Negatives | CC0: Public Domain | Tennis, golf, cricket, baseball, and soccer balls to represent round distractors. |
| **Kaggle (barisyasli/fruit360)** | Hard Negatives | MIT License | Apples, oranges, lemons, and limes representing organic spherical distractors. |
| **Kaggle (balabaskar/count-coins...)** | Hard Negatives | CC0: Public Domain | Coins representing metallic circular distractors. |
| **Unsplash API (Free Commercial)** | Mixed | Unsplash License | High-resolution real photos of pool rooms, pool tables, and sports settings. |
| **PIL Synthetic Generator** | Mixed | Self-Generated (None) | Programmatic felt surfaces, cue rails, pockets, coins, fruits, and spheres with variable shadows. |

---

## 4. Validation & Quality Checks

The dataset was validated using the automated pipeline `scripts/validate_negative_dataset.py` with the following checks:
1. **Corruption Check:** Every image was opened and verified using Pillow (`PIL.Image.verify()`). 100% of images are uncorrupted.
2. **Annotation Verification:** Confirmed that every image has a matching `.txt` file in the corresponding `labels/` folder, and that all label files are exactly 0 bytes (representing background images in YOLO format).
3. **Duplicate Check:** Computed perceptual hashes (using `imagehash.phash`) to group duplicates. Exact duplicates within the source datasets are documented.
4. **Resolution Check:**
   - **Negatives:** Average resolution of **568x478**.
   - **Hard Negatives:** Average resolution of **249x210** (suited for local feature distractor learning).
   - **Format:** All images are in standard `JPEG` format.

**Current Validation Status:** `ALL CHECKS PASSED`
