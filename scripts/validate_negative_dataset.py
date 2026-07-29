"""
validate_negative_dataset.py
============================
Validates the negative and hard-negative datasets:
  - Corrupted image detection (PIL verify)
  - Duplicate detection (perceptual hash)
  - Resolution statistics
  - Label file verification (must be empty)
  - Source/license reporting
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image as PILImage
import imagehash

# ── Paths ──────────────────────────────────────
PROJECT = Path(r"D:\Final Year Project\PoolBallLaserProject")
DATASETS = {
    "negatives": {
        "images": PROJECT / "datasets" / "negatives" / "images",
        "labels": PROJECT / "datasets" / "negatives" / "labels",
        "sources": PROJECT / "datasets" / "negatives" / "sources.json",
    },
    "hard_negatives": {
        "images": PROJECT / "datasets" / "hard_negatives" / "images",
        "labels": PROJECT / "datasets" / "hard_negatives" / "labels",
        "sources": PROJECT / "datasets" / "hard_negatives" / "sources.json",
    },
}


def validate_dataset(name: str, paths: dict):
    """Validate a single dataset category."""
    img_dir = paths["images"]
    lbl_dir = paths["labels"]
    sources_path = paths["sources"]
    
    print(f"\n{'='*60}")
    print(f"  VALIDATING: {name.upper()}")
    print(f"{'='*60}")
    
    results = {
        "name": name,
        "total_images": 0,
        "valid_images": 0,
        "corrupted": [],
        "duplicates": [],
        "missing_labels": [],
        "non_empty_labels": [],
        "resolutions": [],
        "formats": Counter(),
        "sizes_bytes": [],
    }
    
    if not img_dir.exists():
        print(f"  [ERROR] Image directory not found: {img_dir}")
        return results
    
    # Collect all images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])
    results["total_images"] = len(images)
    print(f"  Total files found: {len(images)}")
    
    # ── 1. Corruption Check ──────────────────
    print("\n  [1/4] Checking for corrupted images...")
    valid = []
    for img_path in images:
        try:
            with PILImage.open(img_path) as img:
                img.verify()
            # Re-open for size (verify() leaves file in bad state)
            with PILImage.open(img_path) as img:
                w, h = img.size
                results["resolutions"].append((w, h))
                results["formats"][img.format or "UNKNOWN"] += 1
                results["sizes_bytes"].append(img_path.stat().st_size)
            valid.append(img_path)
        except Exception as e:
            results["corrupted"].append(str(img_path.name))
            print(f"    [CORRUPT] {img_path.name}: {e}")
    
    results["valid_images"] = len(valid)
    if results["corrupted"]:
        print(f"    Found {len(results['corrupted'])} corrupted images")
    else:
        print(f"    All {len(valid)} images are valid")
    
    # ── 2. Duplicate Detection ───────────────
    print("\n  [2/4] Detecting duplicates (perceptual hash)...")
    hash_map = defaultdict(list)
    for img_path in valid:
        try:
            with PILImage.open(img_path) as img:
                h = str(imagehash.phash(img, hash_size=8))
                hash_map[h].append(img_path.name)
        except Exception:
            pass
    
    dup_groups = {h: names for h, names in hash_map.items() if len(names) > 1}
    if dup_groups:
        for h, names in dup_groups.items():
            results["duplicates"].append(names)
            print(f"    [DUP] {len(names)} images share hash {h}: {', '.join(names[:3])}...")
        print(f"    Found {len(dup_groups)} duplicate groups")
    else:
        print(f"    No duplicates found")
    
    # ── 3. Label Verification ────────────────
    print("\n  [3/4] Verifying label files (must be empty for negatives)...")
    for img_path in valid:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            results["missing_labels"].append(img_path.name)
        elif label_path.stat().st_size > 0:
            results["non_empty_labels"].append(img_path.name)
    
    if results["missing_labels"]:
        print(f"    [WARN] {len(results['missing_labels'])} images have no label file")
    if results["non_empty_labels"]:
        print(f"    [ERROR] {len(results['non_empty_labels'])} images have non-empty labels!")
    if not results["missing_labels"] and not results["non_empty_labels"]:
        print(f"    All {len(valid)} label files are valid (empty)")
    
    # ── 4. Resolution Statistics ─────────────
    print("\n  [4/4] Resolution statistics...")
    if results["resolutions"]:
        widths = [r[0] for r in results["resolutions"]]
        heights = [r[1] for r in results["resolutions"]]
        sizes = results["sizes_bytes"]
        
        print(f"    Width  — min: {min(widths)}, max: {max(widths)}, avg: {sum(widths)//len(widths)}")
        print(f"    Height — min: {min(heights)}, max: {max(heights)}, avg: {sum(heights)//len(heights)}")
        print(f"    Size   — min: {min(sizes)//1024}KB, max: {max(sizes)//1024}KB, avg: {sum(sizes)//len(sizes)//1024}KB")
        print(f"    Total disk usage: {sum(sizes)//(1024*1024)}MB")
        print(f"    Formats: {dict(results['formats'])}")
    
    # ── Source Info ───────────────────────────
    if sources_path.exists():
        print(f"\n  Source metadata:")
        with open(sources_path) as f:
            sources = json.load(f)
        for src in sources.get("sources", []):
            print(f"    [{src['category']}] {src['count']} images from {src['source']} ({src['license']})")
    
    return results


def main():
    print("=" * 60)
    print("  NEGATIVE DATASET VALIDATION REPORT")
    print("  Pool Ball Laser Positioning System")
    print("=" * 60)
    
    all_results = []
    
    for name, paths in DATASETS.items():
        result = validate_dataset(name, paths)
        all_results.append(result)
    
    # ── Final Summary ────────────────────────
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    
    total_images = sum(r["total_images"] for r in all_results)
    total_valid = sum(r["valid_images"] for r in all_results)
    total_corrupt = sum(len(r["corrupted"]) for r in all_results)
    total_dups = sum(len(r["duplicates"]) for r in all_results)
    total_missing = sum(len(r["missing_labels"]) for r in all_results)
    total_nonempty = sum(len(r["non_empty_labels"]) for r in all_results)
    
    print(f"  Total images:        {total_images}")
    print(f"  Valid images:        {total_valid}")
    print(f"  Corrupted:           {total_corrupt}")
    print(f"  Duplicate groups:    {total_dups}")
    print(f"  Missing labels:      {total_missing}")
    print(f"  Non-empty labels:    {total_nonempty}")
    
    for r in all_results:
        print(f"\n  {r['name']}:")
        print(f"    Images: {r['valid_images']}/{r['total_images']}")
        if r['resolutions']:
            print(f"    Avg resolution: {sum(w for w,h in r['resolutions'])//len(r['resolutions'])}x{sum(h for w,h in r['resolutions'])//len(r['resolutions'])}")
    
    # Status
    issues = total_corrupt + total_missing + total_nonempty
    if issues == 0:
        print(f"\n  STATUS: ALL CHECKS PASSED")
    else:
        print(f"\n  STATUS: {issues} ISSUES FOUND — review above")
    
    print("=" * 60)
    
    return total_images, total_valid, total_corrupt


if __name__ == "__main__":
    main()
