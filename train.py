"""
Vehicle Classification — Simple Training Script
================================================
- No warmup, no mixup, no fancy LR schedules
- Just: pretrained MobileNetV2 → train → prune(55%) + FP16 → < 5 MB
- Uses CosineAnnealingLR (set-and-forget, stable)

Command:
    python train.py --data_dir new_added --epochs 50 --pretrained --quantize
"""

import os
import copy
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter, ImageFile
import numpy as np
from tqdm import tqdm
import json

from vehicle_classifier import LightweightVehicleCNN, CLASS_IDX, save_compressed

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASS_TO_IDX = {"Bus": 0, "Truck": 1, "Car": 2, "Bike": 3, "None": 4}
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif"}


# ══════════════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.05, num_classes: int = 5):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma

        if self.label_smoothing > 0:
            smooth = torch.full_like(logits, self.label_smoothing / (self.num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = -(smooth * log_probs).sum(dim=1)
        else:
            loss = F.nll_loss(log_probs, targets, reduction="none")

        return (focal_w * loss).mean()


# ══════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = float(decay)
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1.0 - self.decay) * p.data

    def apply(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════
class VehicleDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, bad_image_retries: int = 3):
        self.samples = list(samples)
        self.transform = transform
        self.targets = [s[1] for s in self.samples]
        self.bad_image_retries = max(0, int(bad_image_retries))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        last_label = 0
        for _ in range(self.bad_image_retries + 1):
            path, label = self.samples[i]
            last_label = label
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")
                    img.load()
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception:
                i = random.randrange(len(self.samples))

        return torch.zeros(3, 160, 160), int(last_label)


# ══════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════
class GaussianBlur:
    def __init__(self, p: float = 0.2):
        self.p = float(p)

    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return img


def get_train_transforms(img_size: int = 160):
    return transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        GaussianBlur(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value="random"),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 160):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════
#  SAMPLER
# ══════════════════════════════════════════════════════════════
def make_sampler(dataset: VehicleDataset) -> WeightedRandomSampler:
    targets = list(dataset.targets)
    counts = Counter(targets)
    n, nc = len(targets), len(counts)
    w = {c: n / (nc * cnt) for c, cnt in counts.items()}

    print("\n  Class distribution + weights:")
    for i in sorted(counts):
        print(f"    {CLASS_IDX[i]:>5s}: {counts[i]:5d} imgs → {w[i]:.3f}x")

    return WeightedRandomSampler([w[t] for t in targets], num_samples=n, replacement=True)


# ══════════════════════════════════════════════════════════════
#  TRAIN / VAL
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, ema: EMA):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Train Ep {epoch:3d}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)

        preds = out.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        ema.update(model)

        loss_sum += float(loss.item())
        pbar.set_postfix(loss=f"{loss_sum/(pbar.n+1):.4f}", acc=f"{100*correct/total:.1f}%")

    return loss_sum / max(1, len(loader)), 100.0 * correct / max(1, total)


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cc = [0] * 5
    ct = [0] * 5
    conf = torch.zeros(5, 5, dtype=torch.long)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Validating ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            for t, p in zip(labels.tolist(), preds.tolist()):
                ct[t] += 1
                if t == p:
                    cc[t] += 1
                conf[t][p] += 1

    acc = 100.0 * correct / max(1, total)

    print(f"\n  Val Accuracy: {acc:.2f}%")
    print("  Per-class:")
    for i in range(5):
        if ct[i]:
            print(f"    {CLASS_IDX[i]:>5s}: {100*cc[i]/ct[i]:.1f}%  ({cc[i]}/{ct[i]})")

    print("\n  Confusion Matrix (row=actual, col=predicted):")
    print("        " + "  ".join(f"{CLASS_IDX[i]:>5s}" for i in range(5)))
    for i in range(5):
        print(f"  {CLASS_IDX[i]:>5s}  " + "  ".join(f"{conf[i][j]:5d}" for j in range(5)))

    return acc


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Prune(55%) + FP16 to stay under 5 MB",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save_path", default="student_model.pth")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bad_image_retries", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = device.type == "cuda"

    print(f"\n{'='*55}")
    print("  Vehicle Classifier — Simple Training")
    print(f"  Device     : {device}")
    print(f"  Pretrained : {args.pretrained}")
    print(f"{'='*55}")

    # ── load all samples ────────────────────────────────────
    print("\nScanning dataset...")
    all_samples: List[Tuple[str, int]] = []
    for cls, idx in CLASS_TO_IDX.items():
        cls_dir = os.path.join(args.data_dir, cls)
        found: List[Tuple[str, int]] = []
        if os.path.isdir(cls_dir):
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in VALID_EXT:
                        found.append((os.path.join(root, f), idx))
        all_samples += found
        print(f"  {cls:>5s}: {len(found):5d}  {'✅' if found else '❌ NOT FOUND'}")
    print(f"  Total: {len(all_samples)}")

    # ── stratified split ────────────────────────────────────
    if args.val_dir:
        train_samples = all_samples
        val_samples: List[Tuple[str, int]] = []
        for cls, idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(args.val_dir, cls)
            if os.path.isdir(cls_dir):
                for root, _, files in os.walk(cls_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in VALID_EXT:
                            val_samples.append((os.path.join(root, f), idx))
    else:
        by_class = defaultdict(list)
        for s in all_samples:
            by_class[s[1]].append(s)
        train_samples, val_samples = [], []
        for _, slist in by_class.items():
            random.shuffle(slist)
            n_val = max(1, int(len(slist) * args.val_split))
            val_samples += slist[:n_val]
            train_samples += slist[n_val:]

    print(f"\n  Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    train_tf = get_train_transforms(args.img_size)
    val_tf = get_val_transforms(args.img_size)

    train_ds = VehicleDataset(train_samples, transform=train_tf, bad_image_retries=args.bad_image_retries)
    val_ds = VehicleDataset(val_samples, transform=val_tf, bad_image_retries=args.bad_image_retries)

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin,
    )

    # ── model ───────────────────────────────────────────────
    print("\nCreating model...")
    model = LightweightVehicleCNN(
        num_classes=5,
        pretrained=args.pretrained,
    ).to(device)
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=1.5, label_smoothing=0.05)
    ema = EMA(model, decay=0.998)

    # Differential LR: backbone lower than head
    backbone_p = [p for n, p in model.named_parameters() if "backbone.classifier" not in n]
    head_p = [p for n, p in model.named_parameters() if "backbone.classifier" in n]
    optimizer = optim.AdamW(
        [
            {"params": backbone_p, "lr": args.lr * 0.1},
            {"params": head_p, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )

    print(f"  Optimiser  : AdamW | backbone_lr={args.lr*0.1:.5f} | head_lr={args.lr:.5f}")
    print(f"  Scheduler  : CosineAnnealingLR T_max={args.epochs}")

    # ── training loop ───────────────────────────────────────
    best_acc = 0.0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"\n{'='*55}")
    print(f"  Epochs={args.epochs} | Batch={args.batch_size} | LR={args.lr}")
    print(f"{'='*55}\n")

    for ep in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device, ep, ema)
        scheduler.step()

        ema.apply(model)
        va = validate(model, val_loader, device)
        ema.restore(model)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        history["lr"].append(float(optimizer.param_groups[1]["lr"]))

        current_lr = optimizer.param_groups[1]["lr"]
        print(
            f"  Ep {ep:3d}/{args.epochs} | loss={tl:.4f} | "
            f"train={ta:.1f}% | val={va:.2f}% | lr={current_lr:.2e}"
        )

        if va > best_acc:
            best_acc = va
            no_improve = 0
            ema.apply(model)
            try:
                if args.quantize:
                    mb = save_compressed(
                        model,
                        args.save_path,
                    )
                else:
                    payload = {
                        "arch": "mobilenet_v3_small",
                        "quantized": False,
                        "model_state_dict": copy.deepcopy(model).cpu().state_dict(),
                    }
                    torch.save(payload, args.save_path)
                    mb = os.path.getsize(args.save_path) / 1024 ** 2
            finally:
                ema.restore(model)

            tag = "✅ OK" if mb < 5 else "❌ OVER 5MB"
            print(f"  ★ New best {va:.2f}% | {mb:.2f} MB [{tag}] → {args.save_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping — no improvement for {args.patience} epochs.")
                break

    print(f"\n{'='*55}")
    print(f"  Done! Best val accuracy : {best_acc:.2f}%")
    if os.path.exists(args.save_path):
        mb = os.path.getsize(args.save_path) / 1024 ** 2
        print(f"  Model size             : {mb:.2f} MB {'✅' if mb < 5 else '❌ OVER 5MB'}")
    print(f"{'='*55}")

    hist_path = args.save_path.replace(".pth", "_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  History → {hist_path}")


if __name__ == "__main__":
    main()
