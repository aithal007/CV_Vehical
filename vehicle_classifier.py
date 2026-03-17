import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# -----------------------------
# Class Index Mapping
# -----------------------------
CLASS_IDX = {
    0: "Bus",
    1: "Truck",
    2: "Car",
    3: "Bike",
    4: "None"
}

# -----------------------------
# MobileNetV3-Small Classifier (memory-friendly)
# -----------------------------
class LightweightVehicleCNN(nn.Module):
    """MobileNetV3-Small backbone + custom classifier head.

    Designed to stay comfortably under 5 MiB when saved using dynamic INT8
    quantization of Linear layers.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Torchvision MobileNetV3-Small classifier begins with Linear.
        first = self.backbone.classifier[0]
        if not isinstance(first, nn.Linear):
            raise TypeError(f"Unexpected MobileNetV3 classifier type: {type(first)}")
        in_features = int(first.in_features)
        head_dim = 512
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, head_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.35),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def save_compressed(model: nn.Module, path: str) -> float:
    """Save a <5 MiB checkpoint via dynamic INT8 quantization.

    We quantize Linear layers only (safe + fast on CPU) and then save the
    quantized state_dict. This is typically ~3–4 MiB for MobileNetV3-Small.
    """
    m = model.cpu().eval()
    q = torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    torch.save(q.state_dict(), path)
    return os.path.getsize(path) / (1024 * 1024)


# -----------------------------
# Inference Class
# DONT CHANGE THE INTERFACE OF THE CLASS
# -----------------------------
class VehicleClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        self.model = None

        ckpt = None
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        # Support both raw state_dict checkpoints and metadata-rich dict checkpoints.
        state_dict = None
        if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "state_dict" in ckpt):
            state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        elif isinstance(ckpt, dict):
            state_dict = ckpt

        self.model = LightweightVehicleCNN(num_classes=5, pretrained=False)

        if state_dict is not None:
            is_quantized = any('_packed_params' in k for k in state_dict.keys())
            if is_quantized:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Must match training val transforms exactly: 160x160, ImageNet norm
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path: str) -> int:
        """
        Predict class: 0=Bus, 1=Truck, 2=Car, 3=Bike, 4=None
        """
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Test inference
    classifier = VehicleClassifier(model_path="student_model.pth")
    
    # Check model size
    model_path = "student_model.pth"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        if size_mb < 5:
            print("Model size is within 5 MB limit")
        else:
            print("Model size exceeds 5 MB limit!")
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test prediction (if test image exists)
    test_img = "test_image.jpg"
    if os.path.exists(test_img):
        idx = classifier.predict(test_img)
        print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")