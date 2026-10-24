from fastapi import APIRouter, HTTPException, status, Query
from backend.app.core.config import settings
from typing import List
from collections import deque

router = APIRouter()

@router.get("/logs", response_model=List[str])
def get_logs(lines: int = Query(100, ge=1, le=1000)):
    log_file = settings.BACKEND_DIR / "logs" / "backend.log"
    
    if not log_file.exists():
        return ["No log file found yet."]
        
    try:
        # Read the last N lines from the log file
        with open(log_file, "r", encoding="utf-8") as f:
            # Keep memory bounded even when the log file has grown for months.
            last_lines = deque(f, maxlen=lines)
            return [line.strip() for line in last_lines]
    except (OSError, UnicodeError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to read backend logs."
        ) from None
