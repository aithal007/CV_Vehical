import os
import random
import shutil
import argparse


SEED = 42
TRAIN_RATIO = 0.8
CLASS_ORDER = ["Bus", "Truck", "Car", "Bike", "None"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".jfif"}


def npath(path):
    return os.path.normcase(os.path.abspath(path))


def iter_images_recursive(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(root, name))
    return out


def collect_from_sources(base, source_dirs):
    found = []
    for rel in source_dirs:
        src = os.path.join(base, rel)
        if os.path.isdir(src):
            found.extend(iter_images_recursive(src))
        else:
            print(f"  Warning: source not found: {src}")
    # De-duplicate while keeping stable order.
    dedup = []
    seen = set()
    for p in found:
        np = npath(p)
        if np not in seen:
            seen.add(np)
            dedup.append(p)
    return dedup


def split_copy(images, class_name, train_dir, val_dir, rng):
    images = list(images)
    rng.shuffle(images)
    split = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split]
    val_imgs = images[split:]

    dst_train = os.path.join(train_dir, class_name)
    dst_val = os.path.join(val_dir, class_name)
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_val, exist_ok=True)

    for i, src in enumerate(train_imgs):
        ext = os.path.splitext(src)[1].lower()
        name = f"{class_name.lower()}_{i:07d}{ext}"
        fast_copy(src, os.path.join(dst_train, name))

    for i, src in enumerate(val_imgs):
        ext = os.path.splitext(src)[1].lower()
        name = f"{class_name.lower()}_{i:07d}{ext}"
        fast_copy(src, os.path.join(dst_val, name))

    return len(train_imgs), len(val_imgs)


def fast_copy(src, dst):
    """Use hard links for speed when possible, otherwise copy file data."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Organize complete data dataset")
    parser.add_argument("--only_none", action="store_true", help="Only add/update None class in existing complete data")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base, "complete data")
    out_train = os.path.join(out_root, "train")
    out_val = os.path.join(out_root, "val")

    # User-requested class mapping.
    source_map = {
        "Bus": [
            os.path.join("bus"),
            os.path.join("dataset", "train", "Bus"),
            os.path.join("dataset", "val", "Bus"),
            os.path.join("dataset", "Train Data", "Fit Bus"),
            os.path.join("dataset", "Test Data", "Fit Bus"),
        ],
        "Truck": [
            os.path.join("truck"),
            os.path.join("truck2"),
            os.path.join("Vehicle Type Image Dataset (Version 2) VTID2", "Pickup"),
        ],
        "Car": [
            os.path.join("Car-Bike-Dataset", "Car"),
            os.path.join("car2"),
            os.path.join("Vehicle Type Image Dataset (Version 2) VTID2", "Hatchback"),
            os.path.join("Vehicle Type Image Dataset (Version 2) VTID2", "Seden"),
            os.path.join("Vehicle Type Image Dataset (Version 2) VTID2", "SUV"),
        ],
        "Bike": [
            os.path.join("images.cv_o35cpmt8no96e06lkqt81x"),
            os.path.join("Car-Bike-Dataset", "Bike"),
            os.path.join("Vehicle Type Image Dataset (Version 2) VTID2", "Other"),
        ],
        "None": [
            os.path.join("Vyronasdbmin"),
            os.path.join("pred2"),
            os.path.join("pred3"),
            os.path.join("preded"),
            os.path.join("nature"),
            os.path.join("PennFudanPed"),
        ],
    }

    print("=" * 64)
    print("Creating complete data with user-defined mapping")
    print("=" * 64)

    if os.path.exists(out_root) and not args.only_none:
        print(f"Removing existing output: {out_root}")
        shutil.rmtree(out_root)

    explicit_paths = {"Bus": [], "Truck": [], "Car": [], "Bike": [], "None": []}
    classes_to_collect = ["None"] if args.only_none else ["Bus", "Truck", "Car", "Bike", "None"]

    for cls in classes_to_collect:
        cls_imgs = collect_from_sources(base, source_map[cls])
        explicit_paths[cls] = cls_imgs
        print(f"Collected {len(cls_imgs)} explicit images for {cls}")

    rng = random.Random(SEED)
    stats = {}

    if not args.only_none:
        for cls in ["Bus", "Truck", "Car", "Bike"]:
            t, v = split_copy(explicit_paths[cls], cls, out_train, out_val, rng)
            stats[cls] = (t, v)
    else:
        for cls in ["Bus", "Truck", "Car", "Bike"]:
            train_cls = os.path.join(out_train, cls)
            val_cls = os.path.join(out_val, cls)
            t = len([f for f in os.listdir(train_cls)]) if os.path.isdir(train_cls) else 0
            v = len([f for f in os.listdir(val_cls)]) if os.path.isdir(val_cls) else 0
            stats[cls] = (t, v)

    if args.only_none:
        none_images = explicit_paths["None"]
        print(f"Using {len(none_images)} images for None")
    else:
        non_none_set = set()
        for cls in ["Bus", "Truck", "Car", "Bike"]:
            for p in explicit_paths[cls]:
                non_none_set.add(npath(p))
        none_images = [p for p in explicit_paths["None"] if npath(p) not in non_none_set]
        print(f"Using {len(none_images)} images for None after overlap removal")

    # Refresh None folders when updating only this class.
    if args.only_none:
        shutil.rmtree(os.path.join(out_train, "None"), ignore_errors=True)
        shutil.rmtree(os.path.join(out_val, "None"), ignore_errors=True)

    t, v = split_copy(none_images, "None", out_train, out_val, rng)
    stats["None"] = (t, v)

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    total_train = 0
    total_val = 0
    for cls in CLASS_ORDER:
        tr, va = stats[cls]
        total_train += tr
        total_val += va
        print(f"{cls:<8} train={tr:>7}  val={va:>7}  total={tr + va:>7}")

    print("-" * 64)
    print(f"TOTAL    train={total_train:>7}  val={total_val:>7}  total={total_train + total_val:>7}")
    print(f"\nOutput folder: {out_root}")
    print(f"Train folder:  {out_train}")
    print(f"Val folder:    {out_val}")


if __name__ == "__main__":
    main()