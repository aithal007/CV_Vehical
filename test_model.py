"""
Test the trained vehicle classifier on validation data.
Reports overall accuracy, per-class accuracy, precision/recall/F1,
confusion matrix, and misclassification analysis.
"""

import os
import sys
import time
import warnings

# Suppress noisy deprecation warnings so exit code stays 0
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from PIL import Image
from collections import defaultdict
from vehicle_classifier import VehicleClassifier, CLASS_IDX

BASE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(BASE, "dataset", "val")
MODEL_PATH = os.path.join(BASE, "student_model.pth")

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = ["Bus", "Truck", "Car", "Bike", "None"]
NUM_CLASSES = len(CLASS_NAMES)


def compute_metrics(confusion):
    """Compute precision, recall, F1 from confusion dict."""
    metrics = {}
    for cls in CLASS_NAMES:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in CLASS_NAMES if other != cls)
        fn = sum(confusion[cls][other] for other in CLASS_NAMES if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1,
                        'tp': tp, 'fp': fp, 'fn': fn}
    return metrics


def main():
    print("=" * 65)
    print("  Vehicle Classifier - Validation Evaluation")
    print("=" * 65)

    # ---- Check model ----
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    status = "PASS" if size_mb < 5 else "FAIL (>5 MB)"
    print(f"\n  Model : {os.path.basename(MODEL_PATH)}")
    print(f"  Size  : {size_mb:.2f} MB  [{status}]")

    # ---- Check val dir ----
    if not os.path.isdir(VAL_DIR):
        print(f"\n  ERROR: Validation directory not found at {VAL_DIR}")
        sys.exit(1)

    # ---- Load classifier ----
    print("\n  Loading classifier...")
    t0 = time.time()
    classifier = VehicleClassifier(model_path=MODEL_PATH)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")

    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ---- Evaluate ----
    class_to_idx = {name: idx for idx, name in CLASS_IDX.items()}
    total = 0
    correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    errors = []
    inference_times = []

    print(f"\n  Evaluating on: {VAL_DIR}")
    print("-" * 65)

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(VAL_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  WARNING: {class_name}/ directory not found, skipping")
            continue

        images = sorted([f for f in os.listdir(class_dir)
                         if os.path.splitext(f)[1].lower() in IMG_EXTS])

        true_idx = class_to_idx[class_name]
        class_total[class_name] = len(images)

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            try:
                t1 = time.time()
                pred_idx = classifier.predict(img_path)
                inference_times.append(time.time() - t1)

                pred_name = CLASS_IDX[pred_idx]
                total += 1
                confusion[class_name][pred_name] += 1

                if pred_idx == true_idx:
                    correct += 1
                    class_correct[class_name] += 1
            except Exception as e:
                errors.append((img_name, str(e)))

        acc = 100.0 * class_correct[class_name] / max(class_total[class_name], 1)
        print(f"  {class_name:>5s} : {class_correct[class_name]:>4}/{class_total[class_name]:<4}  = {acc:6.2f}%")

    # ---- Overall ----
    overall_acc = 100.0 * correct / max(total, 1)
    avg_time_ms = 1000 * sum(inference_times) / max(len(inference_times), 1)

    print("-" * 65)
    print(f"  OVERALL ACCURACY : {correct}/{total}  = {overall_acc:.2f}%")
    print(f"  Avg inference    : {avg_time_ms:.1f} ms/image")
    print(f"  Total images     : {total}")
    if errors:
        print(f"  Errors           : {len(errors)}")

    # ---- Precision / Recall / F1 ----
    metrics = compute_metrics(confusion)
    print(f"\n{'=' * 65}")
    print(f"  Per-Class Metrics")
    print(f"{'=' * 65}")
    print(f"  {'Class':>6s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'Support':>8s}")
    print(f"  {'-'*6:>6s}  {'-'*7:>7s}  {'-'*7:>7s}  {'-'*7:>7s}  {'-'*8:>8s}")

    macro_p, macro_r, macro_f1 = 0, 0, 0
    for cls in CLASS_NAMES:
        m = metrics[cls]
        support = class_total[cls]
        print(f"  {cls:>6s}  {100*m['precision']:6.2f}%  {100*m['recall']:6.2f}%  {100*m['f1']:6.2f}%  {support:>8d}")
        macro_p += m['precision']
        macro_r += m['recall']
        macro_f1 += m['f1']

    macro_p /= NUM_CLASSES
    macro_r /= NUM_CLASSES
    macro_f1 /= NUM_CLASSES
    print(f"  {'-'*6:>6s}  {'-'*7:>7s}  {'-'*7:>7s}  {'-'*7:>7s}  {'-'*8:>8s}")
    print(f"  {'Macro':>6s}  {100*macro_p:6.2f}%  {100*macro_r:6.2f}%  {100*macro_f1:6.2f}%  {total:>8d}")

    # ---- Confusion Matrix ----
    print(f"\n{'=' * 65}")
    print(f"  Confusion Matrix (rows = True, cols = Predicted)")
    print(f"{'=' * 65}")
    header = f"  {'':>6s}" + "".join(f"{c:>7s}" for c in CLASS_NAMES)
    print(header)
    print("  " + "-" * (6 + 7 * NUM_CLASSES))

    for true_name in CLASS_NAMES:
        row = f"  {true_name:>6s}"
        for pred_name in CLASS_NAMES:
            count = confusion[true_name][pred_name]
            row += f"{count:>7d}"
        print(row)

    # ---- Misclassification Analysis ----
    misclass = []
    for true_name in CLASS_NAMES:
        for pred_name in CLASS_NAMES:
            if true_name != pred_name and confusion[true_name][pred_name] > 0:
                misclass.append((confusion[true_name][pred_name], true_name, pred_name))
    misclass.sort(reverse=True)

    if misclass:
        print(f"\n{'=' * 65}")
        print(f"  Top Misclassifications")
        print(f"{'=' * 65}")
        for count, true_name, pred_name in misclass[:10]:
            pct = 100.0 * count / max(class_total[true_name], 1)
            print(f"  {true_name:>5s} -> {pred_name:<5s} : {count:>4d}  ({pct:.1f}% of {true_name})")

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    constraint = "PASS" if size_mb < 5 else "FAIL"
    acc_status = "PASS" if overall_acc >= 80 else "FAIL"
    print(f"  SUMMARY")
    print(f"    Model size  : {size_mb:.2f} MB  [{constraint}]")
    print(f"    Accuracy    : {overall_acc:.2f}%    [{acc_status}]")
    print(f"    Macro F1    : {100*macro_f1:.2f}%")
    print(f"    Latency     : {avg_time_ms:.1f} ms/image")
    print(f"{'=' * 65}")

    if errors:
        print(f"\n  Failed images:")
        for name, err in errors[:5]:
            print(f"    {name}: {err}")


if __name__ == "__main__":
    main()
