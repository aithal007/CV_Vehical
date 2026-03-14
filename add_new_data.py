"""
Script to add new data to the existing structured dataset.
Sources:
  - truck2/          -> Truck class (all 466 images)
  - bus/bus/         -> Bus class (all 1000 images)
  - dataset/Train Data/Fit Bus/   -> Bus class (1306 images)
  - dataset/Train Data/Unfit Bus/ -> Bus class (963 images)
  - dataset/Test Data/Fit Bus/    -> Bus class (299 images)
  - dataset/Test Data/Unfit Bus/  -> Bus class (299 images)
  - car2/            -> Car class (random 3000 subset to avoid extreme imbalance)

All new images are split 80/20 into train/val.
"""

import os
import shutil
import random

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.jfif'}


def get_image_files(folder):
    """Get all image files from a folder."""
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS]


def long_path(p):
    """Add Windows long path prefix if needed."""
    p = os.path.abspath(p)
    if len(p) > 200 and not p.startswith('\\\\?\\'):
        return '\\\\?\\' + p
    return p


def safe_copy(src, dst):
    """Copy file with long path support on Windows."""
    shutil.copy2(long_path(src), long_path(dst))


def short_name(img, prefix, max_len=80):
    """Shorten filename if too long, preserving extension."""
    name = f"{prefix}{img}" if prefix else img
    base, ext = os.path.splitext(name)
    if len(name) > max_len:
        # Truncate and add hash to avoid collisions
        import hashlib
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        base = base[:max_len - len(ext) - 9] + '_' + h
        name = base + ext
    return name


def add_images_to_dataset(source_folder, class_name, images=None, prefix=''):
    """Copy images from source to dataset train/val with 80/20 split."""
    train_dest = os.path.join(TRAIN_DIR, class_name)
    val_dest = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)

    if images is None:
        images = get_image_files(source_folder)

    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    copied_train = 0
    copied_val = 0

    for img in train_imgs:
        src = os.path.join(source_folder, img)
        dest_name = short_name(img, prefix)
        dst = os.path.join(train_dest, dest_name)
        if os.path.exists(dst):
            name, ext = os.path.splitext(dest_name)
            dst = os.path.join(train_dest, f"{name}_dup{ext}")
        try:
            safe_copy(src, dst)
            copied_train += 1
        except Exception as e:
            print(f"    WARN: skip {img}: {e}")

    for img in val_imgs:
        src = os.path.join(source_folder, img)
        dest_name = short_name(img, prefix)
        dst = os.path.join(val_dest, dest_name)
        if os.path.exists(dst):
            name, ext = os.path.splitext(dest_name)
            dst = os.path.join(val_dest, f"{name}_dup{ext}")
        try:
            safe_copy(src, dst)
            copied_val += 1
        except Exception as e:
            print(f"    WARN: skip {img}: {e}")

    return copied_train, copied_val


def main():
    print("=" * 60)
    print("ADDING NEW DATA TO DATASET")
    print("=" * 60)

    # Print current counts
    print("\n--- BEFORE: Current dataset counts ---")
    total_before = 0
    for split in ['train', 'val']:
        for cls in ['Bus', 'Truck', 'Car', 'Bike', 'None']:
            p = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(p):
                c = len(get_image_files(p))
                total_before += c
                print(f"  {split}/{cls}: {c}")
    print(f"  TOTAL: {total_before}")

    total_added_train = 0
    total_added_val = 0

    # 1. Add truck2 -> Truck
    print("\n[1/4] Adding truck2/ -> Truck...")
    truck2_dir = os.path.join(BASE, 'truck2')
    t, v = add_images_to_dataset(truck2_dir, 'Truck', prefix='truck2_')
    print(f"  Added {t} train + {v} val = {t+v} truck images")
    total_added_train += t
    total_added_val += v

    # 2. Add bus/bus -> Bus
    print("\n[2/4] Adding bus/bus/ -> Bus...")
    bus_dir = os.path.join(BASE, 'bus', 'bus')
    t, v = add_images_to_dataset(bus_dir, 'Bus', prefix='busbus_')
    print(f"  Added {t} train + {v} val = {t+v} bus images")
    total_added_train += t
    total_added_val += v

    # 3. Add dataset/Train Data + Test Data (bus) -> Bus
    print("\n[3/4] Adding dataset bus folders -> Bus...")
    bus_sources = [
        (os.path.join(DATASET_DIR, 'Train Data', 'Fit Bus'), 'fitbus_train_'),
        (os.path.join(DATASET_DIR, 'Train Data', 'Unfit Bus'), 'unfitbus_train_'),
        (os.path.join(DATASET_DIR, 'Test Data', 'Fit Bus'), 'fitbus_test_'),
        (os.path.join(DATASET_DIR, 'Test Data', 'Unfit Bus'), 'unfitbus_test_'),
    ]
    bus_total_t, bus_total_v = 0, 0
    for src_dir, pfx in bus_sources:
        if os.path.exists(src_dir):
            t, v = add_images_to_dataset(src_dir, 'Bus', prefix=pfx)
            print(f"  {os.path.basename(os.path.dirname(src_dir))}/{os.path.basename(src_dir)}: {t} train + {v} val")
            bus_total_t += t
            bus_total_v += v
    print(f"  Total bus from dataset folders: {bus_total_t} train + {bus_total_v} val = {bus_total_t + bus_total_v}")
    total_added_train += bus_total_t
    total_added_val += bus_total_v

    # 4. Add car2 subset -> Car (random 3000 to avoid extreme imbalance)
    print("\n[4/4] Adding car2/ subset -> Car...")
    car2_dir = os.path.join(BASE, 'car2')
    all_car2 = get_image_files(car2_dir)
    print(f"  car2 has {len(all_car2)} total images")
    CAR_SAMPLE_SIZE = 3000
    if len(all_car2) > CAR_SAMPLE_SIZE:
        random.shuffle(all_car2)
        sampled = all_car2[:CAR_SAMPLE_SIZE]
        print(f"  Sampling {CAR_SAMPLE_SIZE} images to maintain class balance")
    else:
        sampled = all_car2
    t, v = add_images_to_dataset(car2_dir, 'Car', images=sampled, prefix='car2_')
    print(f"  Added {t} train + {v} val = {t+v} car images")
    total_added_train += t
    total_added_val += v

    # Print final counts
    print("\n" + "=" * 60)
    print("--- AFTER: Updated dataset counts ---")
    total_after = 0
    for split in ['train', 'val']:
        for cls in ['Bus', 'Truck', 'Car', 'Bike', 'None']:
            p = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(p):
                c = len(get_image_files(p))
                total_after += c
                print(f"  {split}/{cls}: {c}")
    print(f"  TOTAL: {total_after}")
    print(f"\n  New images added: {total_added_train} train + {total_added_val} val = {total_added_train + total_added_val}")
    print("=" * 60)
    print("DONE! Dataset restructured successfully.")


if __name__ == '__main__':
    main()
