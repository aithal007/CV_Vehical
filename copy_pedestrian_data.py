"""
Copy pedestrian walking images from pred2/pred3/preded into dataset/train/None and dataset/val/None
"""
import os
import shutil
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")

SOURCES = ["pred2", "pred3", "preded"]

def copy_images(src_dir, dst_dir, prefix):
    """Copy only .jpg files (skip .txt annotation files)"""
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for f in os.listdir(src_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(src_dir, f)
            # Add prefix to avoid filename collisions across sources
            dst = os.path.join(dst_dir, f"{prefix}_{f}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
    return copied

total_train = 0
total_val = 0

for source in SOURCES:
    src_base = os.path.join(BASE, source)
    
    # Train split
    train_src = os.path.join(src_base, "train")
    if os.path.isdir(train_src):
        n = copy_images(train_src, os.path.join(DATASET, "train", "None"), source)
        total_train += n
        print(f"  {source}/train -> dataset/train/None: {n} images copied")
    
    # Validation split
    valid_src = os.path.join(src_base, "valid")
    if os.path.isdir(valid_src):
        n = copy_images(valid_src, os.path.join(DATASET, "val", "None"), source)
        total_val += n
        print(f"  {source}/valid -> dataset/val/None:   {n} images copied")

print(f"\nTotal copied: {total_train} train + {total_val} val = {total_train + total_val} images")

# Show final counts
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
for split in ['train', 'val']:
    split_path = os.path.join(DATASET, split)
    if not os.path.exists(split_path):
        continue
    print(f"\n=== {split} ===")
    total = 0
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
        count = sum(1 for f in os.listdir(cls_path) if os.path.splitext(f)[1].lower() in exts)
        total += count
        print(f"  {cls:<20}: {count:>5} images")
    print(f"  {'TOTAL':<20}: {total:>5} images")
