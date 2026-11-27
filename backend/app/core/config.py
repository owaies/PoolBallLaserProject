from pydantic import Field
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

    # Browser access (comma-separated exact origins)
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Admin Authentication
    ADMIN_PASSWORD: str | None = Field(default=None, repr=False)
    ADMIN_JWT_SECRET: str | None = Field(default=None, repr=False)
    ADMIN_JWT_EXPIRE_HOURS: int = 24

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
