import os
import cv2
import time
import logging
import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from ultralytics import YOLO
from backend.app.core.config import settings

logger = logging.getLogger("backend_logger")

class YoloService:
    _instance: Optional['YoloService'] = None
    model: Optional[YOLO] = None
    model_path: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(YoloService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def initialize(self) -> bool:
        """Load the model on startup based on selected game config."""
        from backend.app.core.game_manager import game_manager
        cfg = game_manager.get_current_game_config()
        path = cfg["model_path"]
        return self.load_model(path)

    def load_model(self, path: str) -> bool:
        """Load or reload the YOLO model from a specific path."""
        # Resolve path relative to BASE_DIR if it's not absolute
        model_file = Path(path)
        if not model_file.is_absolute():
            model_file = settings.BASE_DIR / path

        if not model_file.exists():
            logger.error(f"Model file not found at {model_file}")
            self.model = None
            self.model_path = None
            return False

        try:
            logger.info(f"Loading YOLO model from {model_file}...")
            self.model = YOLO(str(model_file))
            self.model_path = path
            logger.info("YOLO model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self.model = None
            self.model_path = None
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        import ultralytics
        from backend.app.core.game_manager import game_manager
        cfg = game_manager.get_current_game_config()

        model_path = Path(settings.BASE_DIR) / cfg["model_path"]
        model_size_mb = 0.0
        file_creation_time = None
        if model_path.exists():
            model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
            file_creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_path.stat().st_ctime))
            
        yolo_ver = getattr(ultralytics, '__version__', '8.x')

        if not self.model:
            return {
                "model_name": Path(cfg["model_path"]).name,
                "model_version": cfg.get("model_version", "unknown"),
                "training_date": file_creation_time or "unknown",
                "yolo_version": yolo_ver,
                "image_size": settings.IMAGE_SIZE,
                "number_of_classes": len(cfg.get("classes", {})),
                "device": "None",
                "model_size": f"{model_size_mb} MB",
                "status": "Model Missing",
                "load_timestamp": game_manager.load_timestamp
            }

        names = getattr(self.model, 'names', {})
        if not names:
            names = cfg["classes"]

        return {
            "model_name": Path(self.model_path).name if self.model_path else "Unknown",
            "model_version": cfg.get("model_version", "unknown"),
            "training_date": file_creation_time or "unknown",
            "yolo_version": yolo_ver,
            "image_size": settings.IMAGE_SIZE,
            "number_of_classes": len(names),
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "model_size": f"{model_size_mb} MB",
            "status": "Ready",
            "load_timestamp": game_manager.load_timestamp
        }

    def predict_image(self, img_bytes: bytes, filename: str, confidence: Optional[float] = None, iou: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str, float]:
        """
        Perform YOLO inference, apply post-processing validation, and generate both
        clean and debug annotated images.
        Returns:
            accepted_detections: only accepted/final detections.
            all_detections: list of all detections with verification stats and rejection reasons.
            annotated_url: clean annotated image.
            debug_annotated_url: developer debug annotated image.
            processing_time: execution time.
        """
        if not self.model:
            raise RuntimeError("YOLO model is not loaded.")

        start_time = time.time()
        
        # Read image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file.")

        h_img, w_img, _ = img.shape

        # Run prediction
        conf_val = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD
        iou_val = iou if iou is not None else settings.IOU_THRESHOLD
        # Dynamic multi-scale & TTA enhancement:
        # Scale to 1280px with test-time augmentation (TTA) for dense clusters & close-up balls
        inference_sz = 1280 if max(h_img, w_img) >= 400 else 640
        results = self.model.predict(
            source=img,
            conf=conf_val,
            iou=iou_val,
            imgsz=inference_sz,
            augment=True,
            agnostic_nms=True,
            verbose=False
        )
        processing_time = time.time() - start_time
        
        result = results[0]
        boxes = result.boxes
        
        raw_detections = []
        global_det_id = 1

        # Process raw predictions
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Center and sizes
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)

            det_id = f"DET_{global_det_id:04d}"
            raw_detections.append({
                "detection_id": det_id,
                "class_name": cls_name,
                "confidence": conf,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "center_x": cx,
                "center_y": cy,
                "width": w,
                "height": h
            })
            global_det_id += 1

        # Run Verification Pipeline
        accepted_detections, all_detections = self.verify_detections(img, raw_detections)

        # Generate Clean Annotated Image
        img_clean = img.copy()
        for d in accepted_detections:
            x1, y1, x2, y2 = int(d["xmin"]), int(d["ymin"]), int(d["xmax"]), int(d["ymax"])
            cx, cy = int(d["center_x"]), int(d["center_y"])
            cls_name = d["class_name"]
            conf = d["confidence"]
            det_id = d["detection_id"]
            color = self.get_class_color(cls_name)

            # Draw outer circle border
            radius = int((d["width"] + d["height"]) / 4)
            cv2.circle(img_clean, (cx, cy), radius, color, 2)
            # Center point
            cv2.circle(img_clean, (cx, cy), 3, (0, 0, 255), -1)
            # Label (Dynamic placement: above circle)
            lbl_y = y1 - 8 if y1 - 8 > 15 else y2 + 15
            label = f"{det_id}|{cls_name.split('_')[0]} ({conf*100:.0f}%)"
            cv2.putText(img_clean, label, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Generate Developer Debug Annotated Image
        img_debug = img.copy()
        for d in all_detections:
            x1, y1, x2, y2 = int(d["xmin"]), int(d["ymin"]), int(d["xmax"]), int(d["ymax"])
            cx, cy = int(d["center_x"]), int(d["center_y"])
            cls_name = d["class_name"]
            conf = d["confidence"]
            det_id = d["detection_id"]
            
            if d["is_accepted"]:
                # Green box & center dot
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img_debug, (cx, cy), 3, (0, 0, 255), -1)
                lbl_y = y1 - 8 if y1 - 8 > 15 else y2 + 15
                label = f"{det_id}|{cls_name} ({conf*100:.0f}%) [OK]"
                cv2.putText(img_debug, label, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # Red dashed-like box (smaller size)
                reason = d["rejection_reason"]
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # Red 'X' in center
                cv2.line(img_debug, (cx - 4, cy - 4), (cx + 4, cy + 4), (0, 0, 255), 1)
                cv2.line(img_debug, (cx + 4, cy - 4), (cx - 4, cy + 4), (0, 0, 255), 1)
                lbl_y = y2 + 12 if y2 + 12 < h_img else y1 - 5
                label = f"{det_id}|{cls_name} [REJECTED: {reason}]"
                cv2.putText(img_debug, label, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # Save annotated images in static uploads folder
        upload_dir = settings.BACKEND_DIR / "app" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save clean
        clean_filename = f"annotated_{filename}"
        cv2.imwrite(str(upload_dir / clean_filename), img_clean)
        
        # Save debug
        debug_filename = f"debug_{filename}"
        cv2.imwrite(str(upload_dir / debug_filename), img_debug)

        annotated_url = f"/api/static/uploads/{clean_filename}"
        debug_annotated_url = f"/api/static/uploads/{debug_filename}"

        return accepted_detections, all_detections, annotated_url, debug_annotated_url, processing_time

    def verify_detections(self, img: np.ndarray, raw_detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Verify raw YOLO detections using four robust, image-agnostic filters:
          A. Aspect Ratio   â€“ pool balls form square bounding boxes (~1:1)
          B. Size Check     â€“ absolute min/max + median-width outlier filter
          C. ROI Margin     â€“ ignores outer 8% of frame (logo/border region)
          D. Duplicate NMS  â€“ center-distance deduplication

        NOTE: Contour-circularity and felt-color checks were intentionally
        removed.  Real pool ball crops produce low circularity scores
        (0.02â€“0.16) because of specular highlights, numbers, and stripe
        patterns, while smooth-edge logos score high â€” exactly the wrong
        behaviour.  The felt-color check had a denominator bug (divided by
        HÃ—WÃ—C instead of HÃ—W) and rejects valid balls photographed away from
        green felt (product images, lighting rigs, etc.).
        """
        h_img, w_img, _ = img.shape

        # Median width for size-outlier detection
        widths = [d["width"] for d in raw_detections]
        median_width = float(np.median(widths)) if widths else 40.0
        # Only apply median-outlier filter if we have enough detections to make
        # it meaningful. With 1â€“3 detections the 'median' IS the detection,
        # so the filter would incorrectly discard valid balls.
        use_median_filter = len(raw_detections) >= 4

        # Adaptive ROI margin â€” high-res images have more pixels near edges
        # so a fixed 8% would exclude many valid near-edge balls.
        # Use 3% for images wider than 3000 px; 5% otherwise.
        margin_frac = 0.03 if w_img > 3000 else 0.05

        processed_detections = []

        for d in raw_detections:
            d_copy = d.copy()
            cx, cy = d_copy["center_x"], d_copy["center_y"]
            w, h = d_copy["width"], d_copy["height"]

            # â”€â”€ A. Aspect Ratio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            aspect_ratio = w / h if h != 0 else 0.0
            d_copy["aspect_ratio"] = aspect_ratio
            d_copy["circularity"] = None   # not computed (see docstring)

            if aspect_ratio < 0.65 or aspect_ratio > 1.55:
                d_copy["is_accepted"] = False
                d_copy["rejection_reason"] = f"Abnormal Aspect Ratio ({aspect_ratio:.2f})"
                processed_detections.append(d_copy)
                continue

            # â”€â”€ B. Size Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if w < 8.0 or h < 8.0:
                d_copy["is_accepted"] = False
                d_copy["rejection_reason"] = "Size Too Small"
                processed_detections.append(d_copy)
                continue

            # Max size: 50% of image dimension (was 35% â€” too strict for close-up photos)
            if w > 0.50 * w_img or h > 0.50 * h_img:
                d_copy["is_accepted"] = False
                d_copy["rejection_reason"] = "Size Too Large"
                processed_detections.append(d_copy)
                continue

            # Median-outlier filter (disabled for small detection counts)
            if use_median_filter and median_width > 0 and abs(w - median_width) / median_width > 0.55:
                d_copy["is_accepted"] = False
                d_copy["rejection_reason"] = f"Size Outlier (Median: {median_width:.1f}px)"
                processed_detections.append(d_copy)
                continue

            # â”€â”€ C. ROI Margin (adaptive) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Adaptive margin: 3% for high-res (>3000px wide), 5% otherwise.
            # This prevents valid corner balls from being excluded in
            # high-resolution snooker photographs.
            margin_x = margin_frac * w_img
            margin_y = margin_frac * h_img
            if cx < margin_x or cx > (w_img - margin_x) or cy < margin_y or cy > (h_img - margin_y):
                d_copy["is_accepted"] = False
                d_copy["rejection_reason"] = f"Outside Frame Margin ({margin_frac*100:.0f}%)"
                processed_detections.append(d_copy)
                continue

            d_copy["is_accepted"] = True
            d_copy["rejection_reason"] = None
            processed_detections.append(d_copy)

        # â”€â”€ D. Centre-Distance Duplicate Suppression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Sort highest-confidence accepted first so the best box wins.
        processed_detections.sort(
            key=lambda x: (x["is_accepted"], x["confidence"]), reverse=True
        )

        final_detections: List[Dict[str, Any]] = []
        for d in processed_detections:
            if not d["is_accepted"]:
                final_detections.append(d)
                continue

            cx, cy = d["center_x"], d["center_y"]
            is_dup = any(
                accepted["is_accepted"] and
                np.sqrt((cx - accepted["center_x"]) ** 2 + (cy - accepted["center_y"]) ** 2)
                < 0.75 * median_width
                for accepted in final_detections
            )
            if is_dup:
                d["is_accepted"] = False
                d["rejection_reason"] = "Duplicate BBox Suppressed"

            final_detections.append(d)

        # Restore original order by detection_id
        final_detections.sort(key=lambda x: x["detection_id"])
        accepted = [d for d in final_detections if d["is_accepted"]]
        return accepted, final_detections


    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Return BGR color representing the class."""
        colors = {
            "cue_ball": (240, 240, 240), # White
            "1_ball": (0, 220, 255),     # Yellow
            "2_ball": (255, 0, 0),       # Blue
            "3_ball": (0, 0, 255),       # Red
            "4_ball": (128, 0, 128),     # Purple
            "5_ball": (0, 140, 255),     # Orange
            "6_ball": (0, 128, 0),       # Green
            "7_ball": (0, 50, 128),      # Brown
            "8_ball": (30, 30, 30),      # Black
            "9_ball": (0, 220, 255),     # Yellow stripe
            "10_ball": (255, 0, 0),      # Blue stripe
            "11_ball": (0, 0, 255),      # Red stripe
            "12_ball": (128, 0, 128),    # Purple stripe
            "13_ball": (0, 140, 255),    # Orange stripe
            "14_ball": (0, 128, 0),      # Green stripe
            "15_ball": (0, 50, 128),     # Brown stripe
            # Snooker
            "red_ball": (0, 0, 255),       # Red
            "yellow_ball": (0, 220, 255),  # Yellow
            "green_ball": (0, 128, 0),     # Green
            "brown_ball": (30, 80, 150),   # Brown
            "blue_ball": (255, 0, 0),      # Blue
            "pink_ball": (180, 100, 255),  # Pink
            "black_ball": (30, 30, 30)     # Black
        }
        return colors.get(class_name, (0, 255, 0))

    def predict_folder(self, folder_path: str) -> Dict[str, Any]:
        """Perform YOLO inference and filter duplicates/false positives on all folder images."""
        if not self.model:
            raise RuntimeError("YOLO model is not loaded.")

        target_dir = Path(folder_path)
        if not target_dir.exists():
            raise FileNotFoundError(f"Target folder not found at {folder_path}")

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [p for p in target_dir.glob("*.*") if p.suffix.lower() in valid_exts]
        
        if not image_files:
            return {"total_images": 0, "total_detections": 0, "csv_path": "", "json_path": ""}

        all_detections = []
        global_det_id = 1

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = self.model.predict(
                source=img,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.IOU_THRESHOLD,
                agnostic_nms=True,
                verbose=False
            )
            result = results[0]
            boxes = result.boxes
            
            raw_detections = []
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                cx = x1 + (w / 2)
                cy = y1 + (h / 2)

                raw_detections.append({
                    "detection_id": f"DET_{global_det_id + idx:04d}",
                    "class_name": cls_name,
                    "confidence": conf,
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "center_x": cx,
                    "center_y": cy,
                    "width": w,
                    "height": h
                })

            # Run Verification (filtering duplicates and false positive logos)
            accepted, _ = self.verify_detections(img, raw_detections)

            for d in accepted:
                all_detections.append({
                    "Image Name": img_path.name,
                    "Detection ID": f"DET_{global_det_id:04d}",
                    "Class": d["class_name"],
                    "Confidence": d["confidence"],
                    "Center X": d["center_x"],
                    "Center Y": d["center_y"],
                    "Xmin": d["xmin"],
                    "Ymin": d["ymin"],
                    "Xmax": d["xmax"],
                    "Ymax": d["ymax"],
                    "Width": d["width"],
                    "Height": d["height"]
                })
                global_det_id += 1

        # Save to CSV
        results_dir = settings.BASE_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "detections.csv"
        
        if all_detections:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_detections[0].keys())
                writer.writeheader()
                writer.writerows(all_detections)

        # Save to JSON
        json_path = results_dir / "detections.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, indent=4)

        return {
            "total_images": len(image_files),
            "total_detections": len(all_detections),
            "csv_path": str(csv_path.relative_to(settings.BASE_DIR)),
            "json_path": str(json_path.relative_to(settings.BASE_DIR))
        }

yolo_service = YoloService()
