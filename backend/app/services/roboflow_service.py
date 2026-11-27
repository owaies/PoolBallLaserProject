import os
import io
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
except ImportError:
    InferenceHTTPClient = None
    InferenceConfiguration = None

# Load local backend settings without embedding credentials in source code.
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

api_key = os.getenv("ROBOFLOW_API_KEY")
api_url = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
client = None

if api_key and InferenceHTTPClient is not None:
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key
    ).configure(InferenceConfiguration(
        api_key_transport="header"
    ))

BALL_COLORS = {
    "white": (255, 255, 255),
    "cue-ball": (255, 255, 255),
    "red": (0, 0, 230),
    "red-ball": (0, 0, 230),
    "yellow": (0, 215, 255),
    "yellow-ball": (0, 215, 255),
    "green": (0, 180, 0),
    "green-ball": (0, 180, 0),
    "brown": (19, 69, 139),
    "brown-ball": (19, 69, 139),
    "blue": (235, 120, 0),
    "blue-ball": (235, 120, 0),
    "pink": (180, 105, 255),
    "pink-ball": (180, 105, 255),
    "black": (30, 30, 30),
    "black-ball": (30, 30, 30),
}

HEX_COLORS = {
    "white": "#FFFFFF",
    "cue-ball": "#FFFFFF",
    "red": "#FF0000",
    "red-ball": "#FF0000",
    "yellow": "#FFD700",
    "yellow-ball": "#FFD700",
    "green": "#008000",
    "green-ball": "#008000",
    "brown": "#8B4513",
    "brown-ball": "#8B4513",
    "blue": "#0000FF",
    "blue-ball": "#0000FF",
    "pink": "#FFC0CB",
    "pink-ball": "#FFC0CB",
    "black": "#000000",
    "black-ball": "#000000",
}

CLASS_IDS = {
    "cue-ball": 0, "white": 0,
    "red-ball": 1, "red": 1,
    "yellow-ball": 2, "yellow": 2,
    "green-ball": 3, "green": 3,
    "brown-ball": 4, "brown": 4,
    "blue-ball": 5, "blue": 5,
    "pink-ball": 6, "pink": 6,
    "black-ball": 7, "black": 7,
}


class RoboflowService:
    def __init__(self):
        self.client = client
        self.model_id = os.getenv("ROBOFLOW_MODEL_ID", "snooker-ball-detection-rnhxo/3")
        self.upload_dir = Path(__file__).resolve().parent.parent.parent / "backend" / "app" / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def infer_and_format(self, image_bytes: bytes, filename: str, conf_threshold: float = 0.40):
        if self.client is None:
            raise RuntimeError(
                "Roboflow inference is unavailable. Configure ROBOFLOW_API_KEY "
                "and install the inference-sdk dependency."
            )

        t0 = time.time()

        # Save temp image for inference client
        temp_input = self.upload_dir / f"temp_{filename}"
        with open(temp_input, "wb") as f:
            f.write(image_bytes)

        try:
            res = self.client.infer(str(temp_input), model_id=self.model_id)
        finally:
            if temp_input.exists():
                try:
                    temp_input.unlink()
                except Exception:
                    pass

        # Load image via cv2 for annotation
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Uploaded image could not be decoded.")
        h_img, w_img = img_bgr.shape[:2]

        predictions = res.get("predictions", [])
        detections = []
        all_detections = []

        for idx, p in enumerate(predictions):
            c_name = str(p.get("class", "unknown")).lower()
            conf = float(p.get("confidence", 0.0))
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
            bw = float(p.get("width", 0.0))
            bh = float(p.get("height", 0.0))

            xmin = max(0.0, cx - bw / 2.0)
            ymin = max(0.0, cy - bh / 2.0)
            xmax = min(float(w_img), cx + bw / 2.0)
            ymax = min(float(h_img), cy + bh / 2.0)

            clean_name = c_name.replace("-ball", "")
            cid = CLASS_IDS.get(c_name, idx)
            hex_color = HEX_COLORS.get(c_name, "#FFFFFF")

            det_obj = {
                "id": f"{clean_name}_{idx}",
                "class_id": cid,
                "class_name": clean_name,
                "raw_class": c_name,
                "confidence": round(conf, 4),
                "x": round(cx, 2),
                "y": round(cy, 2),
                "norm_x": round(cx / w_img, 4) if w_img else 0,
                "norm_y": round(cy / h_img, 4) if h_img else 0,
                "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
                "color": hex_color
            }
            all_detections.append(det_obj)

            if conf >= conf_threshold:
                detections.append(det_obj)
                bgr_color = BALL_COLORS.get(c_name, (0, 255, 0))
                cv2.rectangle(img_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), bgr_color, 3)
                label = f"{clean_name} {int(conf * 100)}%"
                cv2.putText(img_bgr, label, (int(xmin), max(15, int(ymin) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2, cv2.LINE_AA)

        annotated_filename = f"annotated_rf_{filename}"
        annotated_path = self.upload_dir / annotated_filename
        if not cv2.imwrite(str(annotated_path), img_bgr):
            raise RuntimeError("Could not save annotated Roboflow image.")
        annotated_url = f"/api/static/uploads/{annotated_filename}"

        proc_time = time.time() - t0
        return detections, all_detections, annotated_url, proc_time, w_img, h_img


roboflow_service = RoboflowService()
