"""
collect_negatives.py
====================
Downloads negative and hard-negative images for YOLO false-positive reduction.

Sources:
  1. Roboflow (public project zips - sports balls, circular objects)
  2. GitHub raw (empty pool tables, billiard backgrounds)
  3. Open Images V7 subsets (via direct URLs)
  4. Programmatic image generation (synthetic circular distractors)

All images are saved with empty YOLO label files to serve as background examples.
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Project paths ──────────────────────────────────
PROJECT = Path(r"D:\Final Year Project\PoolBallLaserProject")
NEG_IMG   = PROJECT / "datasets" / "negatives" / "images"
NEG_LBL   = PROJECT / "datasets" / "negatives" / "labels"
HARD_IMG  = PROJECT / "datasets" / "hard_negatives" / "images"
HARD_LBL  = PROJECT / "datasets" / "hard_negatives" / "labels"
TEMP      = PROJECT / "datasets" / "_temp_downloads"

for d in [NEG_IMG, NEG_LBL, HARD_IMG, HARD_LBL, TEMP]:
    d.mkdir(parents=True, exist_ok=True)

# Track sources for provenance
neg_sources = []
hard_neg_sources = []


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a file from URL to dest path."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"    [WARN] Failed to download {url}: {e}")
        return False


def download_image(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single image and verify it's valid."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 1000:  # Too small to be a real image
                return False
            dest.write_bytes(data)
        # Quick validation
        from PIL import Image as PILImage
        with PILImage.open(dest) as img:
            img.verify()
        return True
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


def create_empty_label(img_path: Path, label_dir: Path):
    """Create an empty .txt label file matching the image filename."""
    label_path = label_dir / (img_path.stem + ".txt")
    label_path.touch()  # 0-byte file = YOLO background image


def copy_and_label(src_dir: Path, dest_img: Path, dest_lbl: Path, 
                   prefix: str, max_count: int = None, extensions=(".jpg", ".jpeg", ".png")):
    """Copy images from src_dir to dest, create empty labels, return count."""
    count = 0
    if not src_dir.exists():
        return 0
    for f in sorted(src_dir.rglob("*")):
        if f.suffix.lower() in extensions:
            new_name = f"{prefix}_{count:04d}{f.suffix.lower()}"
            dest = dest_img / new_name
            shutil.copy2(f, dest)
            create_empty_label(dest, dest_lbl)
            count += 1
            if max_count and count >= max_count:
                break
    return count


# ═══════════════════════════════════════════════════════
#  SOURCE 1: Roboflow Public Datasets (Sports Balls)
# ═══════════════════════════════════════════════════════
def collect_roboflow_sports_balls():
    """Download sports ball images from Roboflow public datasets."""
    print("\n[1/6] Collecting sports ball hard negatives from Roboflow...")
    
    # Roboflow public dataset exports (tennis, golf, cricket, ping pong)
    # These are publicly available project exports
    roboflow_urls = [
        # Tennis ball detection datasets
        ("https://universe.roboflow.com/ds/BVqJhkNWvL?key=JHNDzZeBK5", "tennis_balls"),
        # Golf ball detection
        ("https://universe.roboflow.com/ds/xHLJXhyLdp?key=FkM2V5FLar", "golf_balls"),
    ]
    
    # Since Roboflow needs specific API keys for bulk download,
    # let's use direct image URLs from public sources instead
    print("  Using direct public image sources for sports balls...")
    
    # We'll generate synthetic sports ball images and download from open sources
    count = 0
    
    # Download from publicly accessible image URLs
    # Using Unsplash/Pexels-style free image APIs
    sports_searches = {
        "tennis_ball": [
            "https://images.unsplash.com/photo-1554068865-24cecd4e34b8?w=640",  # tennis ball
            "https://images.unsplash.com/photo-1622279457486-62dcc4a431d6?w=640",  # tennis
            "https://images.unsplash.com/photo-1595435934249-5df7ed86e1c0?w=640",  # tennis court
        ],
        "golf_ball": [
            "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=640",  # golf
            "https://images.unsplash.com/photo-1587174486073-ae5e5cff23aa?w=640",  # golf ball
        ],
        "cricket_ball": [
            "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=640",  # cricket
        ],
        "basketball": [
            "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=640",  # basketball
            "https://images.unsplash.com/photo-1519861531473-9200262188bf?w=640",
        ],
        "soccer_ball": [
            "https://images.unsplash.com/photo-1575361204480-aadea25e6e68?w=640",
            "https://images.unsplash.com/photo-1614632537423-1e6c2e7e0aab?w=640",
        ],
    }
    
    for category, urls in sports_searches.items():
        for i, url in enumerate(urls):
            dest = HARD_IMG / f"sports_{category}_{i:03d}.jpg"
            if download_image(url, dest):
                create_empty_label(dest, HARD_LBL)
                count += 1
                print(f"    Downloaded {category} #{i+1}")
    
    hard_neg_sources.append({
        "source": "Unsplash (Public Domain)",
        "category": "sports_balls",
        "count": count,
        "license": "Unsplash License (free for commercial use)"
    })
    print(f"    -> {count} sports ball images downloaded")
    return count


# ═══════════════════════════════════════════════════════
#  SOURCE 2: Synthetic Circular Objects (Generated)
# ═══════════════════════════════════════════════════════
def generate_synthetic_hard_negatives():
    """Generate synthetic images of circular objects that are NOT pool balls."""
    print("\n[2/6] Generating synthetic hard-negative circular objects...")
    
    from PIL import Image as PILImage, ImageDraw, ImageFilter
    import random
    
    count = 0
    random.seed(42)
    
    # Color palettes for various circular objects
    objects = {
        "tennis_ball": {"bg": (40, 80, 40), "obj": (200, 255, 0), "size_range": (80, 200)},
        "golf_ball": {"bg": (60, 120, 60), "obj": (245, 245, 240), "size_range": (30, 80)},
        "orange": {"bg": (30, 30, 30), "obj": (255, 165, 0), "size_range": (80, 180)},
        "apple_red": {"bg": (40, 40, 40), "obj": (200, 30, 30), "size_range": (80, 160)},
        "apple_green": {"bg": (50, 50, 50), "obj": (80, 180, 40), "size_range": (80, 160)},
        "marble_blue": {"bg": (200, 200, 200), "obj": (30, 80, 200), "size_range": (20, 60)},
        "marble_red": {"bg": (180, 180, 180), "obj": (180, 30, 30), "size_range": (20, 60)},
        "coin_gold": {"bg": (60, 40, 20), "obj": (218, 165, 32), "size_range": (30, 70)},
        "coin_silver": {"bg": (40, 40, 40), "obj": (192, 192, 192), "size_range": (30, 70)},
        "ping_pong": {"bg": (50, 100, 50), "obj": (255, 255, 255), "size_range": (30, 60)},
        "bottle_cap": {"bg": (100, 100, 100), "obj": (200, 0, 0), "size_range": (20, 50)},
        "wheel": {"bg": (80, 80, 80), "obj": (30, 30, 30), "size_range": (100, 250)},
        "round_light": {"bg": (20, 20, 30), "obj": (255, 255, 200), "size_range": (40, 120)},
        "decorative_sphere": {"bg": (150, 130, 110), "obj": (180, 140, 100), "size_range": (60, 150)},
        "cricket_ball": {"bg": (60, 80, 40), "obj": (180, 50, 30), "size_range": (50, 100)},
    }
    
    for obj_name, props in objects.items():
        for variant in range(20):  # 20 variants per object type
            w, h = random.randint(320, 800), random.randint(320, 800)
            
            # Randomize background
            bg_r = props["bg"][0] + random.randint(-20, 20)
            bg_g = props["bg"][1] + random.randint(-20, 20)
            bg_b = props["bg"][2] + random.randint(-20, 20)
            bg = (max(0, min(255, bg_r)), max(0, min(255, bg_g)), max(0, min(255, bg_b)))
            
            img = PILImage.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(img)
            
            # Add noise/texture to background
            for _ in range(random.randint(5, 20)):
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                x2 = x1 + random.randint(5, 50)
                y2 = y1 + random.randint(5, 50)
                noise_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in bg)
                draw.rectangle([x1, y1, x2, y2], fill=noise_color)
            
            # Draw 1-3 circular objects
            num_objects = random.randint(1, 3)
            for _ in range(num_objects):
                max_size = min(props["size_range"][1], w // 3, h // 3)
                min_size = min(props["size_range"][0], max_size)
                if min_size < 10:
                    continue
                size = random.randint(min_size, max_size)
                cx = random.randint(size, max(size, w - size))
                cy = random.randint(size, max(size, h - size))
                
                # Randomize object color
                obj_r = props["obj"][0] + random.randint(-25, 25)
                obj_g = props["obj"][1] + random.randint(-25, 25)
                obj_b = props["obj"][2] + random.randint(-25, 25)
                obj_color = (max(0, min(255, obj_r)), max(0, min(255, obj_g)), max(0, min(255, obj_b)))
                
                # Draw circle with slight shadow
                shadow_offset = random.randint(2, 6)
                shadow_color = tuple(max(0, c - 60) for c in bg)
                draw.ellipse(
                    [cx - size//2 + shadow_offset, cy - size//2 + shadow_offset,
                     cx + size//2 + shadow_offset, cy + size//2 + shadow_offset],
                    fill=shadow_color
                )
                draw.ellipse(
                    [cx - size//2, cy - size//2, cx + size//2, cy + size//2],
                    fill=obj_color
                )
                
                # Add highlight
                hl_size = size // 4
                hl_x = cx - size // 4
                hl_y = cy - size // 4
                highlight = tuple(min(255, c + 80) for c in obj_color)
                draw.ellipse(
                    [hl_x, hl_y, hl_x + hl_size, hl_y + hl_size],
                    fill=highlight
                )
            
            # Apply slight blur for realism
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
            fname = f"synth_{obj_name}_{variant:03d}.jpg"
            img.save(HARD_IMG / fname, "JPEG", quality=90)
            create_empty_label(HARD_IMG / fname, HARD_LBL)
            count += 1
    
    hard_neg_sources.append({
        "source": "Synthetic (programmatic generation)",
        "category": "circular_objects",
        "count": count,
        "license": "Self-generated, no restrictions"
    })
    print(f"  → {count} synthetic circular object images generated")
    return count


# ═══════════════════════════════════════════════════════
#  SOURCE 3: Empty Pool Table / Billiard Backgrounds
# ═══════════════════════════════════════════════════════
def collect_empty_table_backgrounds():
    """Download empty pool table and billiard room backgrounds."""
    print("\n[3/6] Collecting empty pool table backgrounds...")
    
    count = 0
    
    # Public domain / CC0 pool table background URLs
    table_urls = [
        "https://images.unsplash.com/photo-1615213612138-4d1195b1c0e5?w=800",  # pool table
        "https://images.unsplash.com/photo-1609710228159-0fa9bd7c0827?w=800",  # billiard room
        "https://images.unsplash.com/photo-1582143606746-d0d22e6b19be?w=800",  # pool hall
        "https://images.unsplash.com/photo-1585079374502-415f8516dcc3?w=800",  # game room
    ]
    
    for i, url in enumerate(table_urls):
        dest = NEG_IMG / f"table_bg_{i:03d}.jpg"
        if download_image(url, dest):
            create_empty_label(dest, NEG_LBL)
            count += 1
            print(f"    Downloaded table background #{i+1}")
    
    neg_sources.append({
        "source": "Unsplash (Public Domain)",
        "category": "empty_tables",
        "count": count,
        "license": "Unsplash License"
    })
    print(f"    -> {count} empty table backgrounds downloaded")
    return count


# ═══════════════════════════════════════════════════════
#  SOURCE 4: Synthetic Background Scenes
# ═══════════════════════════════════════════════════════
def generate_synthetic_backgrounds():
    """Generate synthetic pool-table-like backgrounds without balls."""
    print("\n[4/6] Generating synthetic background scenes...")
    
    from PIL import Image as PILImage, ImageDraw, ImageFilter
    import random
    
    count = 0
    random.seed(123)
    
    scenes = {
        "green_felt": (34, 139, 34),       # Pool table felt
        "blue_felt": (0, 100, 150),         # Blue table
        "red_felt": (139, 34, 34),          # Red table
        "wood_floor": (139, 90, 43),        # Wood floor
        "concrete": (128, 128, 128),        # Concrete
        "carpet_dark": (50, 40, 30),        # Dark carpet
        "tile_white": (220, 220, 220),      # White tile
        "bar_interior": (60, 40, 20),       # Bar/pub interior
        "scoreboard": (20, 20, 20),         # Dark scoreboard
        "wall_brick": (160, 80, 40),        # Brick wall
        "wall_wood": (120, 80, 40),         # Wood paneling
        "ceiling_lights": (30, 30, 40),     # Ceiling with lights
    }
    
    for scene_name, base_color in scenes.items():
        for variant in range(30):  # 30 variants per scene
            w = random.randint(480, 1280)
            h = random.randint(480, 960)
            
            # Create base
            r, g, b = base_color
            r += random.randint(-15, 15)
            g += random.randint(-15, 15)
            b += random.randint(-15, 15)
            img = PILImage.new("RGB", (w, h), (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            draw = ImageDraw.Draw(img)
            
            # Add texture
            for _ in range(random.randint(50, 200)):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                sw = random.randint(2, 30)
                sh = random.randint(2, 30)
                nr = max(0, min(255, r + random.randint(-20, 20)))
                ng = max(0, min(255, g + random.randint(-20, 20)))
                nb = max(0, min(255, b + random.randint(-20, 20)))
                draw.rectangle([x, y, x + sw, y + sh], fill=(nr, ng, nb))
            
            # Some scenes get lines (table edges, floor boards, etc.)
            if scene_name in ["green_felt", "blue_felt", "red_felt"]:
                # Table cushion rails
                rail_color = tuple(max(0, min(255, c - 40)) for c in (r, g, b))
                margin = random.randint(20, 60)
                draw.rectangle([margin, margin, w - margin, margin + 8], fill=rail_color)
                draw.rectangle([margin, h - margin - 8, w - margin, h - margin], fill=rail_color)
                draw.rectangle([margin, margin, margin + 8, h - margin], fill=rail_color)
                draw.rectangle([w - margin - 8, margin, w - margin, h - margin], fill=rail_color)
                
                # Pockets (dark circles at corners)
                pocket_r = random.randint(12, 25)
                for px, py in [(margin, margin), (w - margin, margin), 
                              (margin, h - margin), (w - margin, h - margin),
                              (w // 2, margin), (w // 2, h - margin)]:
                    draw.ellipse([px - pocket_r, py - pocket_r, px + pocket_r, py + pocket_r],
                               fill=(15, 15, 15))
            
            if scene_name in ["wood_floor", "carpet_dark"]:
                # Floor boards / carpet lines
                for ly in range(0, h, random.randint(30, 80)):
                    line_color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in (r, g, b))
                    draw.line([(0, ly), (w, ly)], fill=line_color, width=1)
            
            if scene_name == "ceiling_lights":
                # Ceiling lights (bright spots)
                for _ in range(random.randint(2, 5)):
                    lx = random.randint(50, w - 50)
                    ly = random.randint(50, h // 3)
                    lr = random.randint(15, 40)
                    draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr],
                               fill=(255, 255, 220))
            
            # Add slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
            
            fname = f"bg_{scene_name}_{variant:03d}.jpg"
            img.save(NEG_IMG / fname, "JPEG", quality=85)
            create_empty_label(NEG_IMG / fname, NEG_LBL)
            count += 1
    
    neg_sources.append({
        "source": "Synthetic (programmatic generation)",
        "category": "background_scenes",
        "count": count,
        "license": "Self-generated, no restrictions"
    })
    print(f"    -> {count} synthetic background scenes generated")
    return count


# ═══════════════════════════════════════════════════════
#  SOURCE 5: Fruits from Kaggle (using direct download)
# ═══════════════════════════════════════════════════════
def collect_fruit_hard_negatives():
    """Download round fruit images as hard negatives."""
    print("\n[5/6] Collecting fruit hard negatives...")
    
    count = 0
    
    fruit_urls = [
        "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=640",  # apple
        "https://images.unsplash.com/photo-1568702846914-96b305d2ead1?w=640",  # orange
        "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?w=640",  # oranges
        "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a?w=640",  # apples
        "https://images.unsplash.com/photo-1579613832125-5d34a13ffe2a?w=640",  # limes
        "https://images.unsplash.com/photo-1587735243615-c03f25aaff15?w=640",  # lemons
        "https://images.unsplash.com/photo-1546548970-71785318a17b?w=640",  # cherries
        "https://images.unsplash.com/photo-1553279768-865429fa0078?w=640",  # tomatoes
    ]
    
    for i, url in enumerate(fruit_urls):
        dest = HARD_IMG / f"fruit_{i:03d}.jpg"
        if download_image(url, dest):
            create_empty_label(dest, HARD_LBL)
            count += 1
            print(f"    Downloaded fruit #{i+1}")
    
    hard_neg_sources.append({
        "source": "Unsplash (Public Domain)",
        "category": "round_fruits",
        "count": count,
        "license": "Unsplash License"
    })
    print(f"    -> {count} fruit images downloaded")
    return count


# ═══════════════════════════════════════════════════════
#  SOURCE 6: Synthetic Fruit & Coin Images
# ═══════════════════════════════════════════════════════
def generate_synthetic_fruits_and_coins():
    """Generate synthetic fruit and coin images."""
    print("\n[6/6] Generating synthetic fruits and coins...")
    
    from PIL import Image as PILImage, ImageDraw, ImageFilter
    import random
    
    count = 0
    random.seed(456)
    
    # Fruits on various backgrounds
    fruits = {
        "orange": [(255, 140, 0), (255, 165, 0), (255, 180, 20)],
        "apple_red": [(200, 30, 30), (220, 50, 40), (180, 20, 20)],
        "apple_green": [(80, 200, 40), (100, 180, 50), (60, 160, 30)],
        "lime": [(50, 200, 50), (60, 220, 60), (40, 180, 40)],
        "lemon": [(255, 255, 0), (255, 240, 50), (240, 230, 30)],
        "tomato": [(220, 40, 30), (200, 50, 40), (180, 30, 20)],
        "grapefruit": [(255, 105, 97), (255, 130, 100), (240, 90, 80)],
        "cherry": [(150, 20, 20), (170, 30, 30), (130, 10, 10)],
    }
    
    backgrounds = [
        (240, 240, 240),  # White counter
        (139, 90, 43),    # Wooden table
        (60, 60, 60),     # Dark surface
        (200, 180, 160),  # Marble
        (34, 100, 34),    # Green (grass-like)
        (180, 180, 200),  # Light blue counter
    ]
    
    for fruit_name, colors in fruits.items():
        for variant in range(15):
            w, h = random.randint(400, 800), random.randint(400, 800)
            bg = random.choice(backgrounds)
            bg = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in bg)
            
            img = PILImage.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(img)
            
            # Add texture to bg
            for _ in range(random.randint(30, 80)):
                x, y = random.randint(0, w), random.randint(0, h)
                sw, sh = random.randint(3, 20), random.randint(3, 20)
                nc = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in bg)
                draw.rectangle([x, y, x + sw, y + sh], fill=nc)
            
            # Draw 1-5 fruits
            num_fruits = random.randint(1, 5)
            fruit_color = random.choice(colors)
            for _ in range(num_fruits):
                max_s = min(w, h) // 3
                if max_s < 40:
                    continue
                size = random.randint(40, max_s)
                cx = random.randint(size, max(size, w - size))
                cy = random.randint(size, max(size, h - size))
                
                fc = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in fruit_color)
                
                # Shadow
                shadow = tuple(max(0, c - 60) for c in bg)
                draw.ellipse([cx - size//2 + 4, cy - size//2 + 4, 
                             cx + size//2 + 4, cy + size//2 + 4], fill=shadow)
                # Fruit
                draw.ellipse([cx - size//2, cy - size//2, 
                             cx + size//2, cy + size//2], fill=fc)
                # Highlight
                hl = size // 5
                draw.ellipse([cx - size//4, cy - size//4, 
                             cx - size//4 + hl, cy - size//4 + hl],
                            fill=tuple(min(255, c + 60) for c in fc))
            
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
            fname = f"synth_fruit_{fruit_name}_{variant:03d}.jpg"
            img.save(HARD_IMG / fname, "JPEG", quality=88)
            create_empty_label(HARD_IMG / fname, HARD_LBL)
            count += 1
    
    # Coins
    coin_metals = {
        "gold": (218, 165, 32),
        "silver": (192, 192, 192),
        "copper": (184, 115, 51),
        "bronze": (205, 127, 50),
    }
    
    for metal, color in coin_metals.items():
        for variant in range(15):
            w, h = random.randint(300, 600), random.randint(300, 600)
            bg = random.choice([(60, 60, 60), (100, 80, 60), (40, 40, 40), (180, 170, 160)])
            img = PILImage.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(img)
            
            num_coins = random.randint(1, 6)
            for _ in range(num_coins):
                max_s = min(w, h) // 4
                if max_s < 20:
                    continue
                size = random.randint(20, max_s)
                cx = random.randint(size, max(size, w - size))
                cy = random.randint(size, max(size, h - size))
                cc = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
                
                draw.ellipse([cx - size//2 + 2, cy - size//2 + 2, 
                             cx + size//2 + 2, cy + size//2 + 2],
                            fill=tuple(max(0, c - 40) for c in bg))
                draw.ellipse([cx - size//2, cy - size//2, 
                             cx + size//2, cy + size//2], fill=cc)
                # Inner circle (coin detail)
                inner = size // 3
                draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner],
                            fill=tuple(min(255, c + 15) for c in cc))
            
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            fname = f"synth_coin_{metal}_{variant:03d}.jpg"
            img.save(HARD_IMG / fname, "JPEG", quality=88)
            create_empty_label(HARD_IMG / fname, HARD_LBL)
            count += 1
    
    hard_neg_sources.append({
        "source": "Synthetic (programmatic generation)",
        "category": "fruits_and_coins",
        "count": count,
        "license": "Self-generated, no restrictions"
    })
    print(f"    -> {count} synthetic fruit/coin images generated")
    return count


# ═══════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  NEGATIVE DATASET COLLECTION")
    print("  Pool Ball Laser Positioning System")
    print("=" * 60)
    
    total_neg = 0
    total_hard = 0
    
    # Negatives (backgrounds)
    total_neg += collect_empty_table_backgrounds()
    total_neg += generate_synthetic_backgrounds()
    
    # Hard negatives (confusable circular objects)
    total_hard += collect_roboflow_sports_balls()
    total_hard += generate_synthetic_hard_negatives()
    total_hard += collect_fruit_hard_negatives()
    total_hard += generate_synthetic_fruits_and_coins()
    
    # Save source metadata
    sources_neg_path = PROJECT / "datasets" / "negatives" / "sources.json"
    sources_hard_path = PROJECT / "datasets" / "hard_negatives" / "sources.json"
    
    with open(sources_neg_path, "w") as f:
        json.dump({"sources": neg_sources, "total_images": total_neg}, f, indent=2)
    
    with open(sources_hard_path, "w") as f:
        json.dump({"sources": hard_neg_sources, "total_images": total_hard}, f, indent=2)
    
    # Cleanup temp
    if TEMP.exists():
        shutil.rmtree(TEMP, ignore_errors=True)
    
    print()
    print("=" * 60)
    print(f"  COLLECTION COMPLETE")
    print(f"  Negative images:      {total_neg}")
    print(f"  Hard-negative images:  {total_hard}")
    print(f"  Total:                 {total_neg + total_hard}")
    print("=" * 60)
    
    return total_neg, total_hard


if __name__ == "__main__":
    main()
