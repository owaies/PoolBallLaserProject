import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def train():
    data_yaml = r'D:/Final Year Project/AIPOOLBALLLASERPROJECT/datasets/snooker/data.yaml'
    output_model_dir = Path(r'D:/Final Year Project/AIPOOLBALLLASERPROJECT/models/snooker')
    output_model_dir.mkdir(parents=True, exist_ok=True)
    
    print('Initializing YOLOv8n base model...')
    model = YOLO('yolov8n.pt')
    
    print('Starting Snooker Ball Training on GPU...')
    results = model.train(
        data=data_yaml,
        epochs=35,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project='D:/Final Year Project/AIPOOLBALLLASERPROJECT/runs/snooker_train',
        name='yolov8_snooker_detector',
        exist_ok=True,
        patience=10,
        save=True,
        verbose=True
    )
    
    best_pt = Path(model.trainer.best)
    if best_pt.exists():
        dest = output_model_dir / 'best.pt'
        shutil.copy(best_pt, dest)
        print(f'SUCCESS: Best model saved to {dest}')
    else:
        print('Training completed, best.pt not found at expected location.')

if __name__ == '__main__':
    train()