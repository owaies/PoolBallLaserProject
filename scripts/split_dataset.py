import os
import re
import csv
import zipfile
from pathlib import Path

def natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]

def main():
    source_dir = Path("D:/Final Year Project/PoolBallLaserProject/Snooker/snooker-images")
    output_dir = source_dir / "batches"
    
    # 1. Create output directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Scan recursively, ignoring files in the output directory
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    
    for p in source_dir.rglob("*"):
        # Exclude output dir
        if output_dir in p.parents or p == output_dir:
            continue
        if p.is_file() and p.suffix.lower() in extensions:
            images.append(p)
            
    # 3. Natural sorting
    images.sort(key=natural_sort_key)
    
    total_images = len(images)
    if total_images == 0:
        print("No images found to split.")
        return
        
    # 4. Split into exactly 10 batches
    batches_data = []
    start = 0
    for i in range(10):
        size = (total_images // 10) + (1 if i < (total_images % 10) else 0)
        batches_data.append(images[start:start+size])
        start += size
        
    manifest_rows = []
    batch_counts = []
    
    # 5. Create ZIP files
    for idx, batch in enumerate(batches_data):
        batch_num = idx + 1
        zip_name = f"snooker_batch_{batch_num:02d}.zip"
        zip_path = output_dir / zip_name
        
        # Write zip file (overwrite if exists)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zip_f:
            for img_path in batch:
                # Store with arcname as filename only to avoid directory structure inside ZIP
                zip_f.write(img_path, arcname=img_path.name)
                manifest_rows.append({
                    "Filename": img_path.name,
                    "Batch": zip_name
                })
        
        # Verify ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_f:
            bad_file = zip_f.testzip()
            if bad_file:
                print(f"[ERROR] Verification failed for {zip_name}: {bad_file} is corrupt")
            else:
                pass
                
        batch_counts.append((zip_name, len(batch)))
        
    # 6. Generate manifest CSV
    manifest_path = output_dir / "batch_manifest.csv"
    with open(manifest_path, 'w', newline='', encoding='utf-8') as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=["Filename", "Batch"])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # 7. Print summary
    print("====================================")
    print("Snooker Dataset Split Complete")
    print("====================================")
    print(f"Total Images: {total_images}")
    print("Total ZIP Files: 10")
    print("")
    for name, count in batch_counts:
        # Extract batch number from name for display
        batch_label = name.replace("snooker_batch_", "").replace(".zip", "")
        print(f"Batch {batch_label}: {count} images")
    print("")
    print("Output Folder:")
    print(str(output_dir))

if __name__ == "__main__":
    main()
