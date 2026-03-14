"""
Predict vehicle class for one or more images.

Usage:
    python predict.py <image_path>                 # single image
    python predict.py <img1> <img2> <img3> ...     # multiple images
    python predict.py <folder_path>                # all images in a folder
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
from PIL import Image
from vehicle_classifier import VehicleClassifier, LightweightVehicleCNN, CLASS_IDX

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "student_model.pth")
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def predict_with_confidence(classifier, image_path):
    """Predict class and return top-k confidences."""
    image = Image.open(image_path).convert("RGB")
    tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)

    with torch.no_grad():
        logits = classifier.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = probs.argmax().item()
    pred_name = CLASS_IDX[pred_idx]
    confidence = probs[pred_idx].item()

    # All class probabilities
    all_probs = {CLASS_IDX[i]: probs[i].item() for i in range(len(CLASS_IDX))}

    return pred_idx, pred_name, confidence, all_probs


def collect_images(paths):
    """Collect image paths from args (files or folders)."""
    images = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    images.append(os.path.join(p, fname))
        elif os.path.isfile(p):
            images.append(p)
        else:
            print(f"  WARNING: Not found: {p}")
    return images


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [image2] [folder] ...")
        print("\nExamples:")
        print("  python predict.py photo.jpg")
        print("  python predict.py img1.jpg img2.png img3.jpeg")
        print("  python predict.py dataset/val/Bus/")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Load model
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\nModel: {os.path.basename(MODEL_PATH)} ({size_mb:.2f} MB)")
    print("Loading classifier...")
    classifier = VehicleClassifier(model_path=MODEL_PATH)
    print("Ready.\n")

    # Collect all images
    images = collect_images(sys.argv[1:])
    if not images:
        print("No valid images found.")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"  {'Image':<35s}  {'Prediction':<10s}  {'Conf':>6s}  {'Details'}")
    print(f"{'=' * 70}")

    for img_path in images:
        try:
            t0 = time.time()
            pred_idx, pred_name, conf, all_probs = predict_with_confidence(classifier, img_path)
            elapsed = (time.time() - t0) * 1000

            # Top-2 for detail
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_probs[:2]
            detail = f"{top2[0][0]}={100*top2[0][1]:.1f}%"
            if len(top2) > 1 and top2[1][1] > 0.05:
                detail += f", {top2[1][0]}={100*top2[1][1]:.1f}%"

            fname = os.path.basename(img_path)
            if len(fname) > 33:
                fname = fname[:30] + "..."

            print(f"  {fname:<35s}  {pred_name:<10s}  {100*conf:5.1f}%  {detail}")

        except Exception as e:
            fname = os.path.basename(img_path)
            print(f"  {fname:<35s}  ERROR: {e}")

    print(f"{'=' * 70}")
    print(f"  {len(images)} image(s) processed.")

    # If single image, show full breakdown
    if len(images) == 1:
        _, _, _, all_probs = predict_with_confidence(classifier, images[0])
        print(f"\n  Full class probabilities:")
        for cls_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob * 40)
            print(f"    {cls_name:>5s}: {100*prob:6.2f}%  {bar}")


if __name__ == "__main__":
    main()
