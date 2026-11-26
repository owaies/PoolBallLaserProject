"""
Script to train, validate, and test the YOLO model for Pool Ball Detection.
Includes comprehensive logging, modular configuration, and model export.
"""
import os
import sys
import yaml
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO
import torch

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = BASE_DIR / "runs"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for d in [LOGS_DIR, MODELS_DIR, RESULTS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logger setup
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as we want."""
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Also log to stdout for the main logger
    if name == 'main_logger':
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stdout_handler)
        
    return logger

# Initialize loggers
main_logger = setup_logger('main_logger', LOGS_DIR / 'training.log')
val_logger = setup_logger('val_logger', LOGS_DIR / 'validation.log')
test_logger = setup_logger('test_logger', LOGS_DIR / 'testing.log')
error_logger = setup_logger('error_logger', LOGS_DIR / 'error.log', level=logging.ERROR)

def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        error_logger.error(f"Failed to load config {config_path}: {e}")
        sys.exit(1)

def determine_device(requested_device: str) -> str:
    """Determine whether to use GPU or CPU based on configuration and availability."""
    if requested_device.lower() == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = requested_device
    
    main_logger.info(f"Using device: {device.upper()}")
    return device

def train_model(train_cfg: dict, dataset_cfg_path: str):
    """Main function to train the YOLO model."""
    try:
        device = determine_device(train_cfg.get('device', 'auto'))
        model_name = train_cfg.get('model', 'yolov8n.pt')
        
        main_logger.info(f"Initializing YOLO model: {model_name}")
        model = YOLO(model_name)
        
        resume_flag = train_cfg.get('resume', False)
        
        main_logger.info("Starting training process...")
        # Train the model. Ultralytics automatically handles validation at the end of each epoch.
        results = model.train(
            data=dataset_cfg_path,
            epochs=train_cfg.get('epochs', 100),
            batch=train_cfg.get('batch_size', 16),
            imgsz=train_cfg.get('image_size', 640),
            lr0=train_cfg.get('learning_rate', 0.01),
            optimizer=train_cfg.get('optimizer', 'auto'),
            patience=train_cfg.get('patience', 50),
            weight_decay=train_cfg.get('weight_decay', 0.0005),
            workers=train_cfg.get('number_of_workers', 8),
            device=device,
            seed=train_cfg.get('random_seed', 42),
            resume=resume_flag,
            project=str(RUNS_DIR),
            name="train",
            exist_ok=True,
            **train_cfg.get('augmentation', {})
        )
        
        main_logger.info("Training completed successfully.")
        return model, results
        
    except Exception as e:
        error_logger.error(f"Error during training: {e}", exc_info=True)
        main_logger.error(f"Training aborted due to error. See error.log.")
        sys.exit(1)

def evaluate_on_test_set(model: YOLO, dataset_cfg_path: str, conf: float, iou: float):
    """Evaluate the trained model on the test dataset split."""
    test_logger.info("Starting evaluation on the test dataset...")
    try:
        metrics = model.val(
            data=dataset_cfg_path,
            split='test',
            conf=conf,
            iou=iou,
            project=str(RUNS_DIR),
            name="test",
            exist_ok=True
        )
        
        test_logger.info("Test Evaluation Metrics:")
        test_logger.info(f"mAP@50: {metrics.box.map50:.4f}")
        test_logger.info(f"mAP@50-95: {metrics.box.map:.4f}")
        
        # Log per-class metrics if available
        if hasattr(metrics.box, 'maps') and hasattr(metrics.box, 'ap_class_index'):
            test_logger.info("Per-class mAP@50-95:")
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = model.names[class_idx]
                test_logger.info(f"  {class_name}: {metrics.box.maps[i]:.4f}")
        
        test_logger.info("Testing completed successfully.")
        
    except Exception as e:
        error_logger.error(f"Error during testing evaluation: {e}", exc_info=True)
        test_logger.error("Testing aborted due to error.")

def export_model(model: YOLO):
    """Export the trained model to ONNX and TorchScript formats."""
    main_logger.info("Exporting models to ONNX and TorchScript formats...")
    try:
        # Export ONNX
        model.export(format='onnx')
        main_logger.info("Exported to ONNX successfully.")
    except Exception as e:
        error_logger.error(f"Failed to export ONNX: {e}")
        
    try:
        # Export TorchScript
        model.export(format='torchscript')
        main_logger.info("Exported to TorchScript successfully.")
    except Exception as e:
        error_logger.error(f"Failed to export TorchScript: {e}")

def organize_outputs():
    """Copy the best models and visualization plots to the designated directories."""
    main_logger.info("Organizing outputs...")
    
    # Paths in ultralytics run dir
    run_train_dir = RUNS_DIR / "train"
    weights_dir = run_train_dir / "weights"
    
    try:
        if (weights_dir / "best.pt").exists():
            shutil.copy2(weights_dir / "best.pt", MODELS_DIR / "best.pt")
            main_logger.info("Copied best.pt to models directory.")
            
        if (weights_dir / "last.pt").exists():
            shutil.copy2(weights_dir / "last.pt", MODELS_DIR / "last.pt")
            main_logger.info("Copied last.pt to models directory.")
            
        # Copy visualization curves and matrices
        plots_to_copy = [
            ("results.png", "training_statistics.png"),
            ("confusion_matrix.png", "confusion_matrix.png"),
            ("PR_curve.png", "pr_curve.png"),
            ("F1_curve.png", "f1_curve.png"),
            ("P_curve.png", "precision_curve.png"),
            ("R_curve.png", "recall_curve.png")
        ]
        
        for src, dst in plots_to_copy:
            src_path = run_train_dir / src
            if src_path.exists():
                shutil.copy2(src_path, RESULTS_DIR / dst)
                main_logger.info(f"Copied {src} to results/{dst}")
                
    except Exception as e:
        error_logger.error(f"Error organizing outputs: {e}", exc_info=True)

def main():
    os.chdir(BASE_DIR)
    
    # Paths
    training_cfg_path = CONFIGS_DIR / "training.yaml"
    dataset_cfg_path = CONFIGS_DIR / "dataset.yaml"
    
    if not training_cfg_path.exists():
        error_logger.error(f"Training config not found at {training_cfg_path}")
        sys.exit(1)
        
    if not dataset_cfg_path.exists():
        error_logger.error(f"Dataset config not found at {dataset_cfg_path}")
        sys.exit(1)
        
    main_logger.info("Loading configurations...")
    train_cfg = load_config(training_cfg_path)
    
    # 1. Train Model
    model, _ = train_model(train_cfg, str(dataset_cfg_path))
    
    # 2. Evaluate on Test Set
    conf_thresh = train_cfg.get('confidence_threshold', 0.25)
    iou_thresh = train_cfg.get('iou_threshold', 0.7)
    evaluate_on_test_set(model, str(dataset_cfg_path), conf_thresh, iou_thresh)
    
    # 3. Export Models
    export_model(model)
    
    # 4. Organize Results
    organize_outputs()
    
    main_logger.info("Phase 3 - Model Development pipeline executed successfully.")

if __name__ == "__main__":
    main()
