import gc
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings

logger = logging.getLogger("backend_logger")

class GameManager:
    _instance: Optional['GameManager'] = None
    current_game_id: str = "pool"
    configs: Dict[str, Dict[str, Any]] = {}
    load_timestamp: float = 0.0

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GameManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def initialize(self):
        config_dir = settings.BASE_DIR / "configs" / "games"
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            
        self.configs = {}
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg and "game_id" in cfg:
                        self.configs[cfg["game_id"]] = cfg
                        logger.info(f"Loaded game config: {cfg['game_id']}")
            except Exception as e:
                logger.error(f"Failed to load config {yaml_file}: {e}")

        # Fallback configurations if empty
        if not self.configs:
            logger.warning("No game configs found, using static defaults.")
            self.configs["pool"] = {
                "game_id": "pool",
                "game_name": "Pool",
                "display_name": "Pool",
                "icon": "🎱",
                "description": "Standard 8-ball/9-ball pool game with 16 object balls.",
                "model_path": "models/pool/best.pt",
                "model_version": "v1.0.0",
                "default_confidence": 0.65,
                "default_iou": 0.45,
                "table": {"width_mm": 2240, "height_mm": 1120},
                "classes": {i: f"ball_{i}" for i in range(16)}
            }
        
        self.load_timestamp = time.time()

    def get_games_list(self) -> List[Dict[str, Any]]:
        games = []
        for gid, cfg in self.configs.items():
            model_path = Path(settings.BASE_DIR) / cfg["model_path"]
            model_installed = model_path.exists()
            games.append({
                "game_id": gid,
                "game_name": cfg["game_name"],
                "display_name": cfg["display_name"],
                "icon": cfg.get("icon", "🎱"),
                "description": cfg.get("description", ""),
                "model_installed": model_installed,
                "model_version": cfg.get("model_version", "v1.0.0"),
                "classes": cfg["classes"],
                "table": cfg.get("table", {}),
                "is_active": gid == self.current_game_id
            })
        return games

    def select_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        if game_id not in self.configs:
            return None
        
        # Switch and perform unloading to prevent memory leak
        self.unload_model()
        self.current_game_id = game_id
        self.load_timestamp = time.time()
        logger.info(f"Selected game changed to: {game_id}")
        return self.get_current_game_config()

    def get_current_game_config(self) -> Dict[str, Any]:
        cfg = self.configs.get(self.current_game_id)
        if not cfg:
            return next(iter(self.configs.values()))
        return cfg

    def is_model_installed(self, game_id: str) -> bool:
        cfg = self.configs.get(game_id)
        if not cfg:
            return False
        model_path = Path(settings.BASE_DIR) / cfg["model_path"]
        return model_path.exists()

    def unload_model(self):
        """Unload active YOLO model and trigger PyTorch/Python garbage collection."""
        from backend.app.services.yolo_service import yolo_service
        logger.info("Releasing YOLO model to prevent memory leaks...")
        yolo_service.model = None
        yolo_service.model_path = None
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared.")
        except ImportError:
            pass

game_manager = GameManager()
game_manager.initialize()
