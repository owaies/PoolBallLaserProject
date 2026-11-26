"""
Train YOLOv8 on your custom SnookerBalls dataset using local NVIDIA GPU.
"""
import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("Snooker Ball Custom YOLOv8 Training")
    print("=" * 60)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (CUDA not detected)")

    dataset_yaml = r"D:\Dataset\SnookerBalls.yolov8\data.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"Error: {dataset_yaml} not found!")
        return

    # Start from pretrained lightweight yolov8n or existing model
    base_model = "yolov8n.pt"
    print(f"Loading base architecture: {base_model}")
    model = YOLO(base_model)

    print("\nStarting training (50 epochs, imgsz=640, batch=8)...")
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        project="runs/snooker_custom",
        name="custom_table_model",
        save=True,
        exist_ok=True
    )
    
    best_weights = os.path.join("runs", "snooker_custom", "custom_table_model", "weights", "best.pt")
    print("\n" + "=" * 60)
    print(f"Training completed! Best weights saved to:")
    print(f"  {best_weights}")
    print("=" * 60)

if __name__ == "__main__":
    main()

