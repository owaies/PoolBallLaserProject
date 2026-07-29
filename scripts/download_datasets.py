"""
Download pool ball datasets from Roboflow Universe using the API.
"""
from roboflow import Roboflow
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"

API_KEY = "3Ns7DlPqnYlEkRqrNnl8"

# Dataset 1: Pool Ball V4 by Leonardo Wijaya (2076 images, 15 classes)
print("=" * 60)
print("Downloading: Pool Ball V4 (Leonardo Wijaya)")
print("=" * 60)
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("leonardo-wijaya-bdcih").project("pool-ball-v4")
version = project.version(4)
dataset = version.download("yolov8", location=str(RAW_DIR / "pool_ball_v4"))
print(f"Downloaded Pool Ball V4 to: {RAW_DIR / 'pool_ball_v4'}")

# Dataset 2: Pool Ball Detection (by Main)
print("=" * 60)
print("Downloading: Pool Ball Detection (Main)")
print("=" * 60)
try:
    project2 = rf.workspace("main-cpmku").project("pool-ball-detection-jhi6w")
    version2 = project2.version(1)
    dataset2 = version2.download("yolov8", location=str(RAW_DIR / "pool_ball_detection"))
    print(f"Downloaded Pool Ball Detection to: {RAW_DIR / 'pool_ball_detection'}")
except Exception as e:
    print(f"Could not download Pool Ball Detection: {e}")

# Dataset 3: Billiards dataset
print("=" * 60)
print("Downloading: Billiards Dataset")
print("=" * 60)
try:
    project3 = rf.workspace("yolo-q5vmp").project("billiards-y0wwp-et7re")
    version3 = project3.version(1)
    dataset3 = version3.download("yolov8", location=str(RAW_DIR / "billiards"))
    print(f"Downloaded Billiards to: {RAW_DIR / 'billiards'}")
except Exception as e:
    print(f"Could not download Billiards: {e}")

print("\n" + "=" * 60)
print("All available downloads completed!")
print("=" * 60)
