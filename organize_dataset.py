"""
Organize user-provided data into dataset/train and dataset/val
with 80/20 split for 5-class vehicle classification.

Class mapping:
  0: Bus    <- bus/bus
  1: Truck  <- truck/output, VTID2/Pickup
  2: Car    <- Car-Bike-Dataset/Car, VTID2/Hatchback, VTID2/Seden, VTID2/SUV
  3: Bike   <- Car-Bike-Dataset/Bike, VTID2/Other
  4: None   <- PennFudanPed/PNGImages, Vyronasdbmin, My Dataset/*, buildings (if extracted)
"""

import os
import sys
import shutil
import random
import zipfile
import glob

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TRAIN_RATIO = 0.8

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def get_images(folder):
    """Get all image files from a folder."""
    if not os.path.exists(folder):
        return []
    files = []
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if os.path.isfile(full) and os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(full)
    return sorted(files)


def get_images_recursive(folder):
    """Get all image files recursively."""
    if not os.path.exists(folder):
        return []
    files = []
    for root, dirs, fnames in os.walk(folder):
        for f in fnames:
            full = os.path.join(root, f)
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(full)
    return sorted(files)


def copy_images(image_list, class_name, prefix=""):
    """Copy images into train/val split for the given class."""
    random.shuffle(image_list)
    split_idx = int(len(image_list) * TRAIN_RATIO)
    train_imgs = image_list[:split_idx]
    val_imgs = image_list[split_idx:]

    train_cls = os.path.join(TRAIN_DIR, class_name)
    val_cls = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_cls, exist_ok=True)
    os.makedirs(val_cls, exist_ok=True)

    for i, src in enumerate(train_imgs):
        ext = os.path.splitext(src)[1].lower()
        dst_name = f"{prefix}{i:05d}{ext}" if prefix else os.path.basename(src)
        # Avoid duplicates by adding prefix
        dst = os.path.join(train_cls, f"{prefix}{dst_name}" if not prefix else dst_name)
        shutil.copy2(src, dst)

    for i, src in enumerate(val_imgs):
        ext = os.path.splitext(src)[1].lower()
        dst_name = f"{prefix}{i:05d}{ext}" if prefix else os.path.basename(src)
        dst = os.path.join(val_cls, f"{prefix}{dst_name}" if not prefix else dst_name)
        shutil.copy2(src, dst)

    return len(train_imgs), len(val_imgs)


def main():
    print("=" * 60)
    print("Dataset Organizer: 5-Class Vehicle Classification")
    print("=" * 60)
    print("\nClass mapping:")
    print("  0: Bus, 1: Truck, 2: Car, 3: Bike, 4: None\n")

    # Clean previous dataset
    if os.path.exists(DATASET_DIR):
        print("Removing old dataset folder...")
        shutil.rmtree(DATASET_DIR)

    # Extract buildings.zip if exists and not extracted
    buildings_zip = os.path.join(BASE, "buildings.zip")
    buildings_dir = os.path.join(BASE, "buildings")
    if os.path.exists(buildings_zip) and not os.path.exists(buildings_dir):
        print("Extracting buildings.zip...")
        with zipfile.ZipFile(buildings_zip, 'r') as zf:
            zf.extractall(BASE)
        print("  ✓ Extracted")

    # ===================================
    # Define source -> class mapping
    # ===================================
    VTID2 = os.path.join(BASE, "Vehicle Type Image Dataset (Version 2) VTID2")

    class_sources = {
        "Bus": [
            ("bus_main", os.path.join(BASE, "bus", "bus")),
        ],
        "Truck": [
            ("truck_out", os.path.join(BASE, "truck", "output")),
            ("vtid_pickup", os.path.join(VTID2, "Pickup")),
        ],
        "Car": [
            ("carbike_car", os.path.join(BASE, "Car-Bike-Dataset", "Car")),
            ("vtid_hatch", os.path.join(VTID2, "Hatchback")),
            ("vtid_seden", os.path.join(VTID2, "Seden")),
            ("vtid_suv", os.path.join(VTID2, "SUV")),
        ],
        "Bike": [
            ("carbike_bike", os.path.join(BASE, "Car-Bike-Dataset", "Bike")),
            ("vtid_other", os.path.join(VTID2, "Other")),
        ],
        "None": [
            ("penn_ped", os.path.join(BASE, "PennFudanPed", "PNGImages")),
            ("vyronas", os.path.join(BASE, "Vyronasdbmin")),
            ("road_plain_tr", os.path.join(BASE, "My Dataset", "train", "Plain")),
            ("road_pothole_tr", os.path.join(BASE, "My Dataset", "train", "Pothole")),
            ("road_plain_te", os.path.join(BASE, "My Dataset", "test", "Plain")),
            ("road_pothole_te", os.path.join(BASE, "My Dataset", "test", "Pothole")),
        ],
    }

    # Add buildings if extracted
    if os.path.exists(buildings_dir):
        buildings_imgs = get_images_recursive(buildings_dir)
        if buildings_imgs:
            class_sources["None"].append(("buildings", buildings_dir))

    # ===================================
    # Process each class
    # ===================================
    stats = {}

    for class_name in ["Bus", "Truck", "Car", "Bike", "None"]:
        print(f"\n{'='*40}")
        print(f"  Processing: {class_name}")
        print(f"{'='*40}")

        total_train = 0
        total_val = 0

        for prefix, src_dir in class_sources[class_name]:
            if not os.path.exists(src_dir):
                print(f"    ⚠ Not found: {src_dir}")
                continue

            images = get_images(src_dir)
            if not images:
                # Try recursive
                images = get_images_recursive(src_dir)

            if not images:
                print(f"    ⚠ No images in: {src_dir}")
                continue

            t, v = copy_images(images, class_name, prefix=f"{prefix}_")
            total_train += t
            total_val += v
            print(f"    ✓ {prefix}: {len(images)} imgs -> {t} train, {v} val")

        stats[class_name] = {"train": total_train, "val": total_val}
        print(f"    TOTAL {class_name}: {total_train} train, {total_val} val")

    # ===================================
    # Summary
    # ===================================
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\n  {'Class':<8} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("  " + "-" * 36)

    grand_train = 0
    grand_val = 0
    for cls in ["Bus", "Truck", "Car", "Bike", "None"]:
        t = stats[cls]["train"]
        v = stats[cls]["val"]
        grand_train += t
        grand_val += v
        print(f"  {cls:<8} {t:>8} {v:>8} {t+v:>8}")

    print("  " + "-" * 36)
    print(f"  {'TOTAL':<8} {grand_train:>8} {grand_val:>8} {grand_train+grand_val:>8}")

    print(f"\n  Dataset: {DATASET_DIR}")
    print(f"  Train:   {TRAIN_DIR}")
    print(f"  Val:     {VAL_DIR}")

    print("\n" + "=" * 60)
    print("✓ Dataset organized!")
    print("=" * 60)
    print("\nTo train:")
    print(f"  python train.py --data_dir dataset/train --val_dir dataset/val --epochs 50 --pretrained")


if __name__ == "__main__":
    main()
