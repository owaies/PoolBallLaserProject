import time
import logging
from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.services.yolo_service import yolo_service
from backend.app.api import health, model, detect, mapping, statistics, logs

# 1. Setup Loggers
log_dir = settings.BACKEND_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "backend.log"

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("backend_logger")

file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
logger.addHandler(file_handler)

# Capture standard uvicorn logs into file too
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.addHandler(file_handler)

logger.info("Initializing backend server startup...")

# 2. Initialize App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 3. Mount Static Uploads
uploads_dir = settings.BACKEND_DIR / "app" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/static/uploads", StaticFiles(directory=uploads_dir), name="static_uploads")

# 4. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Startup and Shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    success = yolo_service.initialize()
    if not success:
        logger.warning("FastAPI started up but YOLO model loading was postponed or failed.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application...")

# 6. Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(f"Response: Status {response.status_code} | Duration {duration:.4f}s")
    return response

# 7. Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception encountered: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred on the server.",
            "details": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# 8. Include Routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(model.router, prefix="/api", tags=["Model"])
app.include_router(detect.router, prefix="/api", tags=["Detection"])
app.include_router(mapping.router, prefix="/api", tags=["Mapping"])
app.include_router(statistics.router, prefix="/api", tags=["Statistics"])
app.include_router(logs.router, prefix="/api", tags=["Logs"])

logger.info("Backend Application initialization finished.")
