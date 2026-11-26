# Pool Ball Identification & Laser Positioning REST API Backend

This is the FastAPI backend for the AI-Based Pool Ball Identification and Laser Positioning System. It loads the YOLOv8 model, correct radial lens distortion, applies homography mapping, and exposes REST endpoints.

---

## Folder Structure

```
backend/
├── app/
│   ├── api/             # API Router endpoints (health, model, detect, mapping, stats, logs)
│   ├── core/            # Configuration setting (config.py)
│   ├── middleware/      # Middleware definitions
│   ├── models/          # Database models (if needed in future)
│   ├── schemas/         # Pydantic schemas (schemas.py)
│   ├── services/        # Service singletons (yolo_service, mapping_service)
│   ├── static/          # Static files served by FastAPI
│   ├── uploads/         # Destination folder for uploaded/annotated images
│   └── main.py          # FastAPI application entry point
├── logs/                # API log files (backend.log)
├── tests/               # Pytest unit tests (test_main.py)
├── .env                 # Local environment configuration file
├── .env.example         # Template for environment configuration
├── requirements.txt     # Python backend dependencies
└── README.md
```

---

## Installation & Setup

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment file template and configure it:
   ```bash
   copy .env.example .env
   ```

---

## Running the API

Start the development server using Uvicorn:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
* **Interactive API Documentation (Swagger UI):** [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/api/redoc](http://127.0.0.1:8000/api/redoc)

---

## Endpoint Details

### 1. Health Status
* **GET `/api/health`**
* **Response Example:**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 234.56,
    "gpu_available": true,
    "current_model": "models/best.pt"
  }
  ```

### 2. Bounding Box & Class Detection
* **POST `/api/detect/image`**
* **Accepts:** `multipart/form-data` with an image file under the `file` field.
* **Response Example:**
  ```json
  {
    "detections": [
      {
        "detection_id": "DET_0001",
        "class_name": "9_ball",
        "confidence": 0.895,
        "xmin": 120.5,
        "ymin": 230.2,
        "xmax": 160.0,
        "ymax": 270.0,
        "center_x": 140.25,
        "center_y": 250.1,
        "width": 39.5,
        "height": 39.8
      }
    ],
    "annotated_image_url": "/api/static/uploads/annotated_my_table.jpg",
    "processing_time": 0.085
  }
  ```

### 3. Coordinate Mapping
* **POST `/api/mapping`**
* **Accepts JSON:**
  ```json
  {
    "pixel_x": 400.0,
    "pixel_y": 300.0
  }
  ```
* **Response Example:**
  ```json
  {
    "world_x": 990.5,
    "world_y": 495.0
  }
  ```

---

## Running Unit Tests

To execute the Pytest unit testing suite:
```bash
cd backend
pytest tests/
```
