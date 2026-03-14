# Vehicle Classification CNN - Memory Constrained (<5MB)

## Overview
This project implements a robust CNN-based image classifier for street vehicle images under strict memory constraints. The model classifies images into 5 categories:

| Index | Class |
|-------|-------|
| 0 | Bus |
| 1 | Truck |
| 2 | Car |
| 3 | Bike |
| 4 | None |

## Architecture Choice

### Primary: MobileNetV3-Small
- **Parameters**: ~2.5M
- **FP32 Size**: ~9.7 MB
- **Quantized (INT8)**: ~2.5 MB ✓
- Uses depthwise separable convolutions for efficiency
- Pretrained on ImageNet for better feature extraction

### Alternative: SqueezeNet1.1
- **Parameters**: ~1.25M
- **FP32 Size**: ~5 MB
- **Quantized**: ~1.3 MB ✓
- Extremely lightweight, good for very strict constraints

## Data Augmentation Strategy

### For Illumination Robustness:
- **ColorJitter**: Random brightness, contrast, saturation, hue changes
- **RandomGamma**: Simulates exposure variations
- **RandomShadow**: Adds synthetic shadows
- **RandomFog**: Simulates foggy conditions

### For Occlusion Robustness:
- **RandomErasing/Cutout**: Randomly masks rectangular regions
- **CoarseDropout**: Random hole-based occlusion
- **GridDropout**: Grid-based masking

### Geometric Augmentations:
- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation
- RandomPerspective

## Project Structure

```
Assignment_Submission/
├── vehicle_classifier.py    # Main inference class (required)
├── student_model.pth        # Trained model weights (required, <5MB)
├── train.py                 # Training script
├── compress_model.py        # Model compression utilities
├── report.pdf               # Assignment report (required)
└── README.md                # This file (required)
```

## Installation

```bash
pip install torch torchvision pillow numpy tqdm

# Optional: For advanced augmentation
pip install albumentations
```

## Usage

### Training

1. **Organize your dataset** in the following structure:
```
data/
├── Bus/
│   ├── img1.jpg
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

2. **Run training**:
```bash
# Basic training with pretrained MobileNetV3
python train.py --data_dir ./data --epochs 50 --pretrained --quantize

# With Albumentations (recommended for better augmentation)
python train.py --data_dir ./data --epochs 50 --pretrained --quantize --use_albumentations

# Using SqueezeNet (smaller model)
python train.py --data_dir ./data --epochs 50 --model squeezenet --pretrained --quantize
```

### Model Compression

```bash
# Analyze compression options
python compress_model.py
```

### Inference

```python
from vehicle_classifier import VehicleClassifier

# Load classifier
classifier = VehicleClassifier(model_path="student_model.pth")

# Predict
class_idx = classifier.predict("test_image.jpg")
print(f"Predicted class: {class_idx}")  # 0=Bus, 1=Truck, 2=Car, 3=Bike, 4=None
```

## Model Compression Techniques

### 1. Dynamic Quantization (Primary)
- Converts FP32 weights to INT8
- ~4x size reduction
- No accuracy loss for inference
- Applied automatically with `--quantize` flag

### 2. Pruning (Optional)
- Removes low-magnitude weights
- Can achieve additional 20-30% reduction
- May require fine-tuning after pruning

### 3. Architecture Choice
- MobileNetV3-Small uses depthwise separable convolutions
- ~10x fewer parameters than standard convolutions

## Verification

Check model size:
```python
import os
size_mb = os.path.getsize("student_model.pth") / (1024 * 1024)
print(f"Model size: {size_mb:.2f} MB")
assert size_mb < 5, "Model exceeds 5MB limit!"
```

## Training Tips

1. **Use pretrained weights**: Transfer learning from ImageNet significantly improves accuracy
2. **Heavy augmentation**: Essential for handling test variations
3. **Class balancing**: Ensure "None" class is well-represented
4. **Early stopping**: Monitor validation loss to avoid overfitting
5. **Learning rate scheduling**: Use cosine annealing for smooth convergence

## Expected Performance

With proper training:
- Training accuracy: 95%+
- Validation accuracy: 85-90%
- Model size: 2-4 MB (quantized)

## References

- MobileNetV3: Howard et al., "Searching for MobileNetV3"
- SqueezeNet: Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
- Random Erasing: Zhong et al., "Random Erasing Data Augmentation"
