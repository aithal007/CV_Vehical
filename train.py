"""
Advanced Training Script for Vehicle Classification CNN
========================================================
Key Improvements over v1:
  1. WeightedRandomSampler  -> auto-balances classes (Bus gets 4.5x more sampling)
  2. Focal Loss              -> focuses on hard/misclassified examples
  3. Label Smoothing (0.1)   -> prevents overconfident predictions
  4. Mixup Augmentation      -> creates virtual training examples between classes
  5. OneCycleLR Scheduler    -> super-convergence for faster, better training
  6. EMA (Exponential Moving Average) -> smoother final model
  7. Gradient Clipping       -> stabilizes training
  8. Stronger Augmentation   -> GaussianBlur, AffineTransforms, heavier ColorJitter
  9. Class-aware evaluation  -> confusion matrix at end

Usage:
    python train.py --data_dir dataset/train --val_dir dataset/val --epochs 60 --pretrained --quantize
"""

import os
import copy
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import json
from collections import Counter

# Import our model
from vehicle_classifier import LightweightVehicleCNN, CLASS_IDX


# -----------------------------------------------
#  Focal Loss  -  down-weights easy examples
# -----------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) focuses on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, num_classes=5):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = focal_weight * loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()


# -----------------------------------------------
#  Mixup  -  virtual training examples
# -----------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion_fn, pred, y_a, y_b, lam):
    return lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)


# -----------------------------------------------
#  EMA  -  exponential moving average of weights
# -----------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])


# -----------------------------------------------
#  Dataset
# -----------------------------------------------
class VehicleDataset(Dataset):
    CLASS_TO_IDX = {"Bus": 0, "Truck": 1, "Car": 2, "Bike": 3, "None": 4}
    VALID_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.jfif'}

    def __init__(self, data_dir, transform=None, extra_class_dirs=None):
        self.data_dir = data_dir
        self.transform = transform
        self.extra_class_dirs = extra_class_dirs or {}
        self.samples = []
        self.targets = []
        self._load()

    def _collect_images_recursive(self, folder):
        files = []
        for root, _, filenames in os.walk(folder):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in self.VALID_EXT:
                    files.append(os.path.join(root, fname))
        return files

    def _load(self):
        seen = set()

        for class_name, class_idx in self.CLASS_TO_IDX.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"  Warning: {class_dir} not found")
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in self.VALID_EXT:
                    path = os.path.join(class_dir, fname)
                    norm = os.path.normcase(os.path.abspath(path))
                    if norm in seen:
                        continue
                    seen.add(norm)
                    self.samples.append((path, class_idx))
                    self.targets.append(class_idx)

        # Optionally inject additional folders into existing classes (e.g., pedestrian data -> None)
        for class_name, extra_dirs in self.extra_class_dirs.items():
            if class_name not in self.CLASS_TO_IDX:
                print(f"  Warning: Unknown class '{class_name}' in extra_class_dirs, skipping")
                continue

            class_idx = self.CLASS_TO_IDX[class_name]
            for extra_dir in extra_dirs:
                if not os.path.isdir(extra_dir):
                    print(f"  Warning: extra dir not found: {extra_dir}")
                    continue

                added = 0
                for path in self._collect_images_recursive(extra_dir):
                    norm = os.path.normcase(os.path.abspath(path))
                    if norm in seen:
                        continue
                    seen.add(norm)
                    self.samples.append((path, class_idx))
                    self.targets.append(class_idx)
                    added += 1

                print(f"  Added {added} extra images to class '{class_name}' from: {extra_dir}")

        print(f"  Loaded {len(self.samples)} images from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -----------------------------------------------
#  Augmentation Transforms
# -----------------------------------------------
class GaussianBlur:
    def __init__(self, radius_range=(0.1, 2.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


def get_train_transforms(img_size=160):
    """Heavy augmentation for robustness to illumination, occlusion, viewpoint."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.3),

        # Photometric augmentation
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        transforms.RandomGrayscale(p=0.15),
        GaussianBlur(p=0.3),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomEqualize(p=0.2),

        transforms.ToTensor(),

        # Occlusion simulation
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random'),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=160):
    """Clean validation transforms."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# -----------------------------------------------
#  Weighted Sampler for Class Balance
# -----------------------------------------------
def make_weighted_sampler(dataset):
    """Oversamples minority classes so each class has equal probability."""
    targets = dataset.targets
    class_counts = Counter(targets)
    total = len(targets)
    num_classes = len(class_counts)

    print("\n  Class distribution:")
    for idx in sorted(class_counts.keys()):
        name = CLASS_IDX[idx]
        count = class_counts[idx]
        print(f"    {name:>5s}: {count:5d} images ({100.0 * count / total:5.1f}%)")

    class_weights = {cls: total / (num_classes * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]

    print("\n  Effective sampling weights:")
    for idx in sorted(class_weights.keys()):
        print(f"    {CLASS_IDX[idx]:>5s}: {class_weights[idx]:.3f}x")

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(targets), replacement=True)


# -----------------------------------------------
#  Training + Validation
# -----------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch, use_mixup=True, mixup_alpha=0.2, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Train Ep {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        do_mixup = use_mixup and epoch > 3 and random.random() < 0.5
        if do_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(y_a).sum().item() + (1 - lam) * preds.eq(y_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}
    confusion = torch.zeros(5, 5, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            for t, p in zip(labels, preds):
                ti, pi = t.item(), p.item()
                class_total[ti] += 1
                if ti == pi:
                    class_correct[ti] += 1
                confusion[ti][pi] += 1

    overall_acc = 100. * correct / total

    print(f"\n  Val Accuracy: {overall_acc:.2f}%")
    print("  Per-class:")
    for idx in range(5):
        name = CLASS_IDX[idx]
        if class_total[idx] > 0:
            acc = 100. * class_correct[idx] / class_total[idx]
            print(f"    {name:>5s}: {acc:6.2f}%  ({class_correct[idx]}/{class_total[idx]})")

    print("\n  Confusion Matrix (row=actual, col=predicted):")
    print("        " + "  ".join(f"{CLASS_IDX[i]:>5s}" for i in range(5)))
    for i in range(5):
        print(f"  {CLASS_IDX[i]:>5s}  " + "  ".join(f"{confusion[i][j]:5d}" for j in range(5)))

    return overall_acc


# -----------------------------------------------
#  Quantize + Save
# -----------------------------------------------
def quantize_and_save(model, path):
    m = copy.deepcopy(model).cpu().eval()
    q = torch.quantization.quantize_dynamic(m, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    torch.save(q.state_dict(), path)
    return os.path.getsize(path) / (1024 * 1024)


def save_float(model, path):
    torch.save(model.cpu().state_dict(), path)
    return os.path.getsize(path) / (1024 * 1024)


# -----------------------------------------------
#  Main
# -----------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Vehicle Classifier Training v2')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=160)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, default='student_model.pth')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_pred_pedestrian', action='store_true',
                        help='Add pred2/pred3/preded (train/valid) as extra None-class images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"  Vehicle Classifier Training v2")
    print(f"  Device: {device}")
    print(f"{'='*55}")

    # Transforms
    train_tf = get_train_transforms(args.img_size)
    val_tf = get_val_transforms(args.img_size)

    # Datasets
    print("\nLoading datasets...")
    train_extra_dirs = {}
    val_extra_dirs = {}

    if args.use_pred_pedestrian:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_none_dirs = [
            os.path.join(base_dir, 'pred2', 'train'),
            os.path.join(base_dir, 'pred3', 'train'),
            os.path.join(base_dir, 'preded', 'train'),
        ]
        val_none_dirs = [
            os.path.join(base_dir, 'pred2', 'valid'),
            os.path.join(base_dir, 'pred3', 'valid'),
            os.path.join(base_dir, 'preded', 'valid'),
        ]
        train_extra_dirs['None'] = train_none_dirs
        val_extra_dirs['None'] = val_none_dirs

    train_ds = VehicleDataset(args.data_dir, transform=train_tf, extra_class_dirs=train_extra_dirs)
    if args.val_dir:
        val_ds = VehicleDataset(args.val_dir, transform=val_tf, extra_class_dirs=val_extra_dirs)
    else:
        from torch.utils.data import random_split
        n_val = int(0.2 * len(train_ds))
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])

    # Weighted sampler
    print("\nBuilding weighted sampler for class balance...")
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Model
    print("\nCreating model...")
    model = LightweightVehicleCNN(num_classes=5, pretrained=args.pretrained)
    model = model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    # Simple CE with label smoothing — class balancing is handled by WeightedRandomSampler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"  Loss: CrossEntropyLoss (label_smoothing={args.label_smoothing})")

    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    best_val_acc = 0.0
    patience = 15
    patience_ctr = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n{'='*55}")
    print(f"  Epochs: {args.epochs} | Mixup: {'ON' if not args.no_mixup else 'OFF'}")
    print(f"  Loss: CE + label_smoothing={args.label_smoothing}")
    print(f"  LR: {args.lr} | Scheduler: CosineAnnealing | Grad clip={args.clip_grad}")
    print(f"{'='*55}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, use_mixup=not args.no_mixup, mixup_alpha=args.mixup_alpha,
            clip_grad=args.clip_grad
        )

        # Validate with actual model weights
        val_acc = validate(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Ep {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.2f}% | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            if args.quantize:
                size_mb = quantize_and_save(model, args.save_path)
            else:
                size_mb = save_float(model, args.save_path)
                model.to(device)
            ok = "OK" if size_mb < 5 else "OVER 5MB!"
            print(f"  >>> New best! {val_acc:.2f}% | {size_mb:.2f} MB [{ok}]")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping (no improvement for {patience} epochs)")
                break

    print(f"\n{'='*55}")
    print(f"  Done! Best accuracy: {best_val_acc:.2f}%")
    if os.path.exists(args.save_path):
        sz = os.path.getsize(args.save_path) / (1024 * 1024)
        print(f"  Model: {sz:.2f} MB {'(under 5MB)' if sz < 5 else '(OVER 5MB!)'}")
    print(f"{'='*55}")

    with open(args.save_path.replace('.pth', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
