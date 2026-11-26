import os
import io
import zipfile
from pathlib import Path
from PIL import Image

def main():
    batches_dir = Path("D:/Final Year Project/PoolBallLaserProject/Snooker/snooker-images/batches")
    output_dir = Path("D:/Final Year Project/PoolBallLaserProject/Snooker/optimized-batches")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = sorted(list(batches_dir.glob("snooker_batch_*.zip")))
    if not zip_files:
        print("No zip batch files found in the batches directory.")
        return
        
    original_total_size = 0
    optimized_total_size = 0
    images_processed = 0
    failures = []
    
    print("Starting image optimization across batches...")
    
    for zip_path in zip_files:
        zip_name = zip_path.name
        original_size = zip_path.stat().st_size
        original_total_size += original_size
        
        out_zip_path = output_dir / zip_name
        print(f"Optimizing {zip_name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as in_zip:
                # Read all items
                with zipfile.ZipFile(out_zip_path, 'w', zipfile.ZIP_STORED) as out_zip:
                    for name in in_zip.namelist():
                        img_bytes = in_zip.read(name)
                        
                        try:
                            # Load image
                            img = Image.open(io.BytesIO(img_bytes))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                                
                            width, height = img.size
                            longest_side = max(width, height)
                            
                            # Resize if longest side exceeds 1920
                            if longest_side > 1920:
                                if width >= height:
                                    new_width = 1920
                                    new_height = int((height * 1920) / width)
                                else:
                                    new_height = 1920
                                    new_width = int((width * 1920) / height)
                                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                
                            # Save as JPEG (Quality 92, EXIF stripped)
                            out_io = io.BytesIO()
                            img.save(out_io, format="JPEG", quality=92)
                            optimized_bytes = out_io.getvalue()
                            
                            # Write to new zip preserving filename
                            out_zip.writestr(name, optimized_bytes)
                            images_processed += 1
                        except Exception as e:
                            print(f"Failed to process image {name} in {zip_name}: {e}")
                            # Fallback: copy original if PIL fails
                            out_zip.writestr(name, img_bytes)
                            failures.append(f"{zip_name}/{name}: {str(e)}")
            
            # Verify Zip
            with zipfile.ZipFile(out_zip_path, 'r') as ver_zip:
                bad_file = ver_zip.testzip()
                if bad_file:
                    failures.append(f"{zip_name} failed verification: {bad_file} is corrupt")
                    print(f"[WARNING] Verification failed for {zip_name}")
                    
            opt_size = out_zip_path.stat().st_size
            optimized_total_size += opt_size
            
        except Exception as e:
            print(f"Failed to process ZIP {zip_name}: {e}")
            failures.append(f"{zip_name}: {str(e)}")
            
    # Calculate reduction
    size_reduction = original_total_size - optimized_total_size
    percent_reduction = (size_reduction / original_total_size) * 100 if original_total_size > 0 else 0
    
    orig_mb = original_total_size / (1024 * 1024)
    opt_mb = optimized_total_size / (1024 * 1024)
    
    print("\n====================================")
    print("Image Optimization Complete")
    print("====================================")
    print(f"Original total size: {orig_mb:.2f} MB")
    print(f"Optimized total size: {opt_mb:.2f} MB")
    print(f"Percentage reduction: {percent_reduction:.1f}%")
    print(f"Number of images processed: {images_processed}")
    print(f"Any failures: {len(failures)}")
    if failures:
        print("Failures list:")
        for f in failures[:10]:
            print(f" - {f}")
        if len(failures) > 10:
            print(" ... and more")
            
if __name__ == "__main__":
    main()
