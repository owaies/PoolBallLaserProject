from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Pool Ball Identification & Laser Positioning API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Model Configuration
    MODEL_PATH: str = "models/best.pt"
    CONFIDENCE_THRESHOLD: float = 0.60
    IOU_THRESHOLD: float = 0.45
    IMAGE_SIZE: int = 640
    DEVICE: str = "auto"
    
    # Upload Settings
    MAX_UPLOAD_SIZE: int = 10485760  # 10 MB

    # Logging
    LOG_LEVEL: str = "INFO"

    # Admin Authentication
    ADMIN_PASSWORD: Optional[str] = Field(default=None, min_length=12)
    ADMIN_JWT_SECRET: Optional[str] = Field(default=None, min_length=32)
    ADMIN_JWT_EXPIRE_HOURS: int = 24

    @field_validator("CONFIDENCE_THRESHOLD", "IOU_THRESHOLD")
    @classmethod
    def validate_probability_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Detection thresholds must be between 0 and 1.")
        return value

    @field_validator("IMAGE_SIZE", "MAX_UPLOAD_SIZE", "ADMIN_JWT_EXPIRE_HOURS")
    @classmethod
    def ensure_positive_integer_settings(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Image size, upload size, and JWT expiry must be positive integers.")
        return value

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("PORT must be between 1 and 65535.")
        return value
    
    # Path constants
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent

    @property
    def DATASETS_DIR(self) -> Path:
        return self.BASE_DIR / "datasets"

    @property
    def BACKUPS_DIR(self) -> Path:
        return self.BASE_DIR / "datasets" / ".backups"

    @property
    def ADMIN_LOGS_DIR(self) -> Path:
        return self.BASE_DIR / "logs" / "admin"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
