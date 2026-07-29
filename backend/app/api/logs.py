from fastapi import APIRouter, HTTPException, status, Query
from backend.app.core.config import settings
from typing import List
import os

router = APIRouter()

@router.get("/logs", response_model=List[str])
def get_logs(lines: int = Query(100, ge=1, le=1000)):
    log_file = settings.BACKEND_DIR / "logs" / "backend.log"
    
    if not log_file.exists():
        return ["No log file found yet."]
        
    try:
        # Read the last N lines from the log file
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            # Strip newline characters for clean JSON array response
            return [line.strip() for line in last_lines]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading log file: {str(e)}"
        )
