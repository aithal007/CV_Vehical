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
# MobileNetV3-Small Classifier (<5 MB quantized)
# -----------------------------
class LightweightVehicleCNN(nn.Module):
    """
    MobileNetV3-Small backbone + custom classifier head.
    ~1.1M params, ~3.8 MB with INT8 quantization.
    """
    def __init__(self, num_classes=5, pretrained=False):
        super(LightweightVehicleCNN, self).__init__()
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)

        in_features = self.backbone.classifier[0].in_features
        head_dim = 512
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, head_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.35),
            nn.Linear(head_dim, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# Inference Class
# DONT CHANGE THE INTERFACE OF THE CLASS
# -----------------------------
class VehicleClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        self.model = LightweightVehicleCNN(num_classes=5, pretrained=False)

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

            # Detect quantized model (has _packed_params keys)
            is_quantized = any('_packed_params' in k for k in state_dict.keys())
            if is_quantized:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
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
        image = Image.open(image_path).convert("RGB")
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
            print("✓ Model size is within 5 MB limit")
        else:
            print("✗ Model size exceeds 5 MB limit!")
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test prediction (if test image exists)
    test_img = "test_image.jpg"
    if os.path.exists(test_img):
        idx = classifier.predict(test_img)
        print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")