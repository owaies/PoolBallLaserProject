import os
import random
from pathlib import Path
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
out_dir = BASE_DIR / 'datasets' / 'raw' / 'synthetic_pool_balls'
(out_dir / 'images').mkdir(parents=True, exist_ok=True)
(out_dir / 'labels').mkdir(parents=True, exist_ok=True)

# Ball colors (approximate)
COLORS = {
    0: (255, 255, 255), # Cue
    1: (255, 255, 0),   # 1
    2: (0, 0, 255),     # 2
    3: (255, 0, 0),     # 3
    4: (128, 0, 128),   # 4
    5: (255, 165, 0),   # 5
    6: (0, 128, 0),     # 6
    7: (128, 0, 0),     # 7
    8: (0, 0, 0),       # 8
    9: (255, 255, 0),   # 9 stripe
    10: (0, 0, 255),    # 10 stripe
    11: (255, 0, 0),    # 11 stripe
    12: (128, 0, 128),  # 12 stripe
    13: (255, 165, 0),  # 13 stripe
    14: (0, 128, 0),    # 14 stripe
    15: (128, 0, 0)     # 15 stripe
}

def generate_dataset(num_images=50):
    img_size = 640
    ball_radius = 20
    
    for i in range(num_images):
        # Green background
        img = Image.new('RGB', (img_size, img_size), color=(34, 139, 34))
        draw = ImageDraw.Draw(img)
        
        num_balls = random.randint(1, 5)
        labels = []
        
        for _ in range(num_balls):
            class_id = random.randint(0, 15)
            x_center = random.randint(ball_radius, img_size - ball_radius)
            y_center = random.randint(ball_radius, img_size - ball_radius)
            
            # Draw ball
            color = COLORS[class_id]
            draw.ellipse(
                (x_center - ball_radius, y_center - ball_radius, 
                 x_center + ball_radius, y_center + ball_radius),
                fill=color, outline=(0, 0, 0)
            )
            
            # YOLO format: class x_center y_center width height (normalized)
            norm_x = x_center / img_size
            norm_y = y_center / img_size
            norm_w = (ball_radius * 2) / img_size
            norm_h = (ball_radius * 2) / img_size
            
            labels.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        img_name = f"synth_{i:04d}"
        img.save(out_dir / 'images' / f"{img_name}.jpg")
        with open(out_dir / 'labels' / f"{img_name}.txt", 'w') as f:
            f.write("\n".join(labels))

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    generate_dataset(50)
    print("Generated 50 synthetic images.")
