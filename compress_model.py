"""
Model Compression Utilities
============================
Tools for compressing the trained model to meet the <5MB constraint.
Includes:
- Dynamic quantization
- Static quantization
- Pruning
- Model size analysis
"""

import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import copy

from vehicle_classifier import LightweightVehicleCNN, SqueezeNetClassifier, CLASS_IDX


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_size(model, dtype=torch.float32):
    """Estimate model size in MB based on parameter count and dtype."""
    total_params, _ = count_parameters(model)
    
    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.float16:
        bytes_per_param = 2
    elif dtype == torch.int8 or dtype == torch.qint8:
        bytes_per_param = 1
    else:
        bytes_per_param = 4
    
    size_bytes = total_params * bytes_per_param
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def apply_dynamic_quantization(model):
    """
    Apply dynamic quantization to linear and conv layers.
    This is the simplest form of quantization - no calibration needed.
    """
    model_cpu = model.cpu()
    model_cpu.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model


def apply_static_quantization(model, calibration_loader, device='cpu'):
    """
    Apply static quantization with calibration.
    This provides better accuracy than dynamic quantization but requires a calibration dataset.
    """
    model = model.cpu()
    model.eval()
    
    # Fuse modules (required for static quantization)
    # Note: This depends on the specific model architecture
    # For MobileNetV3, we'd need to access the internal fuse_modules
    
    # Set up quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data
    print("Calibrating model...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            model_prepared(images.cpu())
            break  # Just use one batch for calibration
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model_prepared)
    
    return quantized_model


def apply_pruning(model, amount=0.3, method='l1_unstructured'):
    """
    Apply pruning to reduce model size.
    
    Args:
        model: PyTorch model
        amount: Fraction of connections to prune (0.0 to 1.0)
        method: Pruning method ('l1_unstructured', 'random_unstructured', 'ln_structured')
    
    Returns:
        Pruned model
    """
    model = copy.deepcopy(model)
    
    # Get all conv and linear layers
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))
    
    # Apply global pruning
    if method == 'l1_unstructured':
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
    elif method == 'random_unstructured':
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=amount
        )
    
    # Remove pruning reparameterization to make it permanent
    for module, _ in modules_to_prune:
        if hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
    
    return model


def analyze_sparsity(model):
    """Analyze the sparsity of model weights."""
    total_zeros = 0
    total_params = 0
    
    print("\nLayer-wise sparsity analysis:")
    print("-" * 50)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            zeros = (param == 0).sum().item()
            total = param.numel()
            sparsity = 100 * zeros / total
            
            total_zeros += zeros
            total_params += total
            
            print(f"{name}: {sparsity:.2f}% sparse ({zeros}/{total})")
    
    print("-" * 50)
    print(f"Overall sparsity: {100 * total_zeros / total_params:.2f}%")
    
    return total_zeros / total_params


def save_compressed_model(model, save_path, quantize=True, prune_amount=0.0):
    """
    Save model with compression options.
    
    Args:
        model: PyTorch model
        save_path: Output file path
        quantize: Whether to apply dynamic quantization
        prune_amount: Fraction of weights to prune (0.0 = no pruning)
    """
    model = copy.deepcopy(model)
    model.eval()
    
    # Apply pruning if requested
    if prune_amount > 0:
        print(f"Applying pruning with amount={prune_amount}")
        model = apply_pruning(model, amount=prune_amount)
        analyze_sparsity(model)
    
    # Apply quantization if requested
    if quantize:
        print("Applying dynamic quantization...")
        model = apply_dynamic_quantization(model)
    
    # Save
    torch.save(model.state_dict(), save_path)
    
    # Report size
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nModel saved to: {save_path}")
    print(f"Final model size: {size_mb:.2f} MB")
    
    if size_mb < 5:
        print("✓ Model meets the <5 MB requirement!")
    else:
        print("✗ Model exceeds 5 MB limit. Consider more aggressive compression.")
    
    return size_mb


def compare_model_sizes(model):
    """Compare model sizes with different compression options."""
    print("\n" + "="*60)
    print("Model Size Comparison")
    print("="*60)
    
    # Save temporary files to measure actual sizes
    import tempfile
    
    results = {}
    temp_files = []
    
    # FP32 (no compression)
    try:
        f = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_files.append(f.name)
        f.close()
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        results['FP32 (no compression)'] = size
    except Exception as e:
        print(f"Could not save FP32 model: {e}")
    
    # Dynamic quantization
    try:
        quantized = apply_dynamic_quantization(model)
        f = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_files.append(f.name)
        f.close()
        torch.save(quantized.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        results['Dynamic Quantization (INT8)'] = size
    except Exception as e:
        print(f"Could not apply dynamic quantization: {e}")
    
    # Pruning (30%)
    try:
        pruned = apply_pruning(model, amount=0.3)
        f = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_files.append(f.name)
        f.close()
        torch.save(pruned.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        results['Pruning (30%)'] = size
    except Exception as e:
        print(f"Could not apply pruning: {e}")
    
    # Pruning + Quantization
    try:
        pruned = apply_pruning(model, amount=0.3)
        pruned_quantized = apply_dynamic_quantization(pruned)
        f = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_files.append(f.name)
        f.close()
        torch.save(pruned_quantized.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        results['Pruning (30%) + Quantization'] = size
    except Exception as e:
        print(f"Could not apply pruning + quantization: {e}")
    
    # Cleanup temp files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    # Print results
    print(f"\n{'Method':<40} {'Size (MB)':<10} {'Status'}")
    print("-" * 60)
    for method, size in results.items():
        status = "✓" if size < 5 else "✗"
        print(f"{method:<40} {size:<10.2f} {status}")
    
    return results


def main():
    """Test compression utilities."""
    print("="*60)
    print("Vehicle Classifier - Model Compression Analysis")
    print("="*60)
    
    # Create model
    print("\n1. Creating MobileNetV3-Small model...")
    model = LightweightVehicleCNN(num_classes=5, pretrained=False)
    
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Estimate sizes
    print("\n2. Estimated model sizes:")
    print(f"   FP32: {estimate_model_size(model, torch.float32):.2f} MB")
    print(f"   FP16: {estimate_model_size(model, torch.float16):.2f} MB")
    print(f"   INT8: {estimate_model_size(model, torch.qint8):.2f} MB")
    
    # Compare actual sizes
    compare_model_sizes(model)
    
    # Also check SqueezeNet
    print("\n" + "="*60)
    print("Checking SqueezeNet as alternative...")
    print("="*60)
    
    squeezenet = SqueezeNetClassifier(num_classes=5, pretrained=False)
    total_params, _ = count_parameters(squeezenet)
    print(f"SqueezeNet parameters: {total_params:,}")
    compare_model_sizes(squeezenet)


if __name__ == "__main__":
    main()
