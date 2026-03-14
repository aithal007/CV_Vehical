import os
import random
import shutil
import argparse


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".jfif"}
CLASSES = ["Bus", "Truck", "Car", "Bike", "None"]


def iter_images(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMG_EXTS:
            out.append(path)
    return out


def fast_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Build balanced dataset with up to 3k images per class")
    parser.add_argument("--source", default="complete data", help="Source root containing train/val/class folders")
    parser.add_argument("--output", default="dataset_3k", help="Output root for balanced train/val")
    parser.add_argument("--per_class", type=int, default=3000, help="Target total images per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.join(base, args.source)
    out_root = os.path.join(base, args.output)
    out_train = os.path.join(out_root, "train")
    out_val = os.path.join(out_root, "val")

    if os.path.exists(out_root):
        shutil.rmtree(out_root)

    rng = random.Random(args.seed)
    print("=" * 68)
    print(f"Building balanced dataset from: {src_root}")
    print(f"Output: {out_root}")
    print(f"Target per class: {args.per_class}")
    print("=" * 68)

    totals = {"train": 0, "val": 0}
    for cls in CLASSES:
        cls_train = iter_images(os.path.join(src_root, "train", cls))
        cls_val = iter_images(os.path.join(src_root, "val", cls))
        pool = cls_train + cls_val

        if not pool:
            print(f"{cls:<8} no images found, skipping")
            continue

        rng.shuffle(pool)
        take = min(len(pool), args.per_class)
        selected = pool[:take]

        split = int(take * args.train_ratio)
        tr = selected[:split]
        va = selected[split:]

        for i, src in enumerate(tr):
            ext = os.path.splitext(src)[1].lower()
            fast_copy(src, os.path.join(out_train, cls, f"{cls.lower()}_{i:05d}{ext}"))

        for i, src in enumerate(va):
            ext = os.path.splitext(src)[1].lower()
            fast_copy(src, os.path.join(out_val, cls, f"{cls.lower()}_{i:05d}{ext}"))

        totals["train"] += len(tr)
        totals["val"] += len(va)
        print(f"{cls:<8} available={len(pool):>6} selected={take:>6} train={len(tr):>6} val={len(va):>6}")

    print("-" * 68)
    print(f"TOTAL            train={totals['train']:>6} val={totals['val']:>6} all={totals['train'] + totals['val']:>6}")
    print(f"\nUse for training:")
    print(f"python train.py --data_dir \"{os.path.join(args.output, 'train')}\" --val_dir \"{os.path.join(args.output, 'val')}\" --epochs 80 --img_size 160 --pretrained --quantize")


if __name__ == "__main__":
    main()