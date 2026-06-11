# 🚗 Vehicle Classification CNN

> A lightweight CNN-based vehicle image classifier constrained to **< 5 MB**, built with PyTorch and MobileNetV3-Small. Classifies street-level images into five categories.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Model Compression](#model-compression)
  - [Evaluation](#evaluation)
- [Dataset Preparation](#dataset-preparation)
- [Data Augmentation](#data-augmentation)
- [Training Results](#training-results)
- [References](#references)
- [License](#license)

---

## Overview

This project implements a memory-constrained (< 5 MB) image classification model for street vehicle recognition. It uses **MobileNetV3-Small** as the backbone with **INT8 dynamic quantization** to meet the size constraint while maintaining high accuracy.

Key highlights:

- **~96.9% validation accuracy** after 10 epochs of training
- **~3.95 MB** quantized model size (well under the 5 MB limit)
- Robust to illumination changes, occlusions, and geometric variations via heavy data augmentation
- Simple single-file inference API via `VehicleClassifier`

---

## Classes

| Index | Class   | Description                        |
|-------|---------|------------------------------------|
| 0     | Bus     | Buses and large passenger vehicles |
| 1     | Truck   | Trucks and freight vehicles        |
| 2     | Car     | Sedans, SUVs, hatchbacks, etc.     |
| 3     | Bike    | Motorcycles and bicycles           |
| 4     | None    | Non-vehicle / background images    |

---

## Model Architecture

| Property            | Value                 |
|---------------------|-----------------------|
| Backbone            | MobileNetV3-Small     |
| Parameters          | ~2.5 M                |
| FP32 Size           | ~9.7 MB               |
| Quantized (INT8)    | ~3.95 MB ✅           |
| Input Size          | 224 × 224 × 3         |
| Pretrained Weights  | ImageNet              |

MobileNetV3-Small uses **depthwise separable convolutions** and **squeeze-and-excitation blocks** for an excellent accuracy-to-size trade-off. An alternative SqueezeNet1.1 backbone (~1.3 MB quantized) is also supported for ultra-strict constraints.

---

## Project Structure

```
CV_Vehical/
│
├── vehicle_classifier.py          # Inference API — VehicleClassifier class
├── train.py                       # Full training pipeline with augmentation & quantization
├── predict.py                     # CLI tool for single-image or batch prediction
├── test_model.py                  # Comprehensive model evaluation & robustness testing
├── compress_model.py              # Post-training compression (quantization, pruning, export)
│
├── student_model.pth              # Trained & quantized model weights (~3.95 MB)
├── student_model_v1_backup.pth    # Backup of earlier model version (~3.81 MB)
├── student_model_history.json     # Training loss & accuracy history (10 epochs)
│
├── add_new_data.py                # Download & integrate new images from web sources
├── build_balanced_3k_dataset.py   # Build a balanced 3 000-image subset
├── check_dataset.py               # Quick dataset class-distribution checker
├── copy_pedestrian_data.py        # Import pedestrian images into the "None" class
├── organize_complete_data.py      # Merge & deduplicate images from multiple sources
├── organize_dataset.py            # Full dataset pipeline — download, split, & balance
│
├── README.dataset.txt             # Pedestrian dataset attribution (Roboflow, CC BY 4.0)
├── .gitignore                     # Ignores datasets, outputs, caches, IDE files
├── .gitattributes                 # Git LFS tracking for .pth files
└── README.md                      # This file
```

### File Roles at a Glance

| File | Purpose |
|------|---------|
| `vehicle_classifier.py` | **Core inference module.** Defines `VehicleClassifier` with `predict(image_path) → int` and `predict_with_confidence(image_path) → (int, float)`. Handles model loading, preprocessing, and INT8 quantization. |
| `train.py` | **Training script.** Supports MobileNetV3 & SqueezeNet, progressive resizing (160→224), Albumentations augmentation, class-weighted sampling, cosine LR scheduling, mixed precision, and post-training quantization. |
| `predict.py` | **Prediction CLI.** Run predictions on a single image or an entire directory, with optional top-k confidence display and annotated output saving. |
| `test_model.py` | **Evaluation suite.** Generates a classification report, confusion matrix, per-class metrics, and robustness tests (brightness, contrast, noise, blur, occlusion). |
| `compress_model.py` | **Compression toolkit.** Applies dynamic/static quantization, structured & unstructured pruning, ONNX export, and reports size & accuracy impact. |
| `add_new_data.py` | Downloads additional training images from Unsplash and web searches, validates them, and organises by class. |
| `build_balanced_3k_dataset.py` | Samples a balanced 600-per-class subset from the full dataset. |
| `organize_dataset.py` | End-to-end dataset builder — downloads from multiple sources, deduplicates, balances, and creates train/val splits. |
| `organize_complete_data.py` | Merges images from heterogeneous source directories with hash-based deduplication. |
| `copy_pedestrian_data.py` | Copies pedestrian images (from PennFudanPed or Roboflow) into the `None` class for negative samples. |
| `check_dataset.py` | Prints per-class image counts for a dataset directory. |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch ≥ 1.9
- torchvision

```bash
# Core dependencies
pip install torch torchvision pillow numpy tqdm

# Optional — advanced augmentations (recommended)
pip install albumentations
```

---

## Usage

### Training

1. **Prepare the dataset** (see [Dataset Preparation](#dataset-preparation) below).

2. **Run training:**

```bash
# MobileNetV3-Small with pretrained weights + quantization (recommended)
python train.py --data_dir ./data --epochs 50 --pretrained --quantize

# With Albumentations augmentation
python train.py --data_dir ./data --epochs 50 --pretrained --quantize --use_albumentations

# SqueezeNet backbone (smaller model)
python train.py --data_dir ./data --epochs 50 --model squeezenet --pretrained --quantize
```

The trained model is saved to `student_model.pth`.

### Inference

```python
from vehicle_classifier import VehicleClassifier

classifier = VehicleClassifier(model_path="student_model.pth")

# Basic prediction — returns class index (0–4)
class_idx = classifier.predict("test_image.jpg")
print(f"Predicted class: {class_idx}")

# Prediction with confidence score
class_idx, confidence = classifier.predict_with_confidence("test_image.jpg")
print(f"Class: {class_idx}, Confidence: {confidence:.2%}")
```

**CLI prediction:**

```bash
# Single image
python predict.py --image test_image.jpg --model student_model.pth

# Entire directory
python predict.py --dir ./test_images/ --model student_model.pth --save_output
```

### Model Compression

```bash
# Analyse and apply compression techniques
python compress_model.py
```

This will:
- Apply dynamic & static INT8 quantization
- Apply structured and unstructured pruning
- Export to ONNX format
- Report size and accuracy trade-offs for each method

### Evaluation

```bash
python test_model.py --model student_model.pth --data_dir ./data
```

Generates:
- Per-class precision, recall, and F1 scores
- Confusion matrix
- Robustness test results (brightness, contrast, noise, blur, occlusion variations)

---

## Dataset Preparation

Organise images into class folders:

```
data/
├── Bus/
│   ├── img001.jpg
│   └── ...
├── Truck/
│   └── ...
├── Car/
│   └── ...
├── Bike/
│   └── ...
└── None/
    └── ...
```

**Automated tools included:**

| Script | What it does |
|--------|-------------|
| `organize_dataset.py` | Downloads from multiple web sources, deduplicates, balances to 600/class, creates train/val split |
| `build_balanced_3k_dataset.py` | Samples a balanced 3 000-image subset from an existing dataset |
| `add_new_data.py` | Augments the dataset with images from Unsplash and web searches |
| `copy_pedestrian_data.py` | Adds pedestrian images as negative ("None" class) samples |

---

## Data Augmentation

The training pipeline applies aggressive augmentation for robustness:

### Illumination
- ColorJitter (brightness, contrast, saturation, hue)
- Random gamma correction
- Synthetic shadows and fog

### Occlusion
- Random erasing / cutout
- Coarse dropout
- Grid-based masking

### Geometric
- Random resized crop
- Horizontal flip
- Rotation (±15°)
- Perspective distortion

---

## Training Results

Results from the included `student_model_history.json` (10 epochs):

| Metric | Final Value |
|--------|-------------|
| Training Accuracy | 90.47% |
| Validation Accuracy | **96.91%** |
| Training Loss | 0.613 |
| Model Size (quantized) | ~3.95 MB |

---

## References

- **MobileNetV3** — Howard et al., *"Searching for MobileNetV3"* (2019)
- **SqueezeNet** — Iandola et al., *"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"* (2016)
- **Random Erasing** — Zhong et al., *"Random Erasing Data Augmentation"* (2020)
- **Pedestrian Dataset** — [Roboflow Pedestrian Detection](https://universe.roboflow.com/training-data-kgqsn/pedestrian-detection-v6aln) (CC BY 4.0)

---

## License

This project is provided for educational and research purposes. The pedestrian dataset used for the "None" class is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
