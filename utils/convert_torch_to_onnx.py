"""
PyTorch to ONNX Model Converter

This script converts a PyTorch model to ONNX format with automatic detection
of whether the input is a state dictionary or a complete model.
"""

import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from typing import Any

from torchvision import models
from torch import nn


class PreTrainedClassifier(nn.Module):
    """ResNet or EfficientNet backbone → custom FC head for N classes."""

    def __init__(self, 
                 num_classes: int = 3,
                 dropout: float = 0.5, 
                 pretrained: bool = True,
                 model_backbone: str = "resnet18",
                 ) -> None:
        super().__init__()
        self.model_backbone_map = {
            'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
            'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
            'efficientnetb1': models.EfficientNet_B1_Weights.IMAGENET1K_V2, 
            'efficientnetb4': models.EfficientNet_B4_Weights.IMAGENET1K_V1, 
        }
        self.dropout = dropout
        if model_backbone in self.model_backbone_map and pretrained:
            weights = self.model_backbone_map[model_backbone] 
        elif model_backbone in self.model_backbone_map and not pretrained:
            weights = None
        else:
            raise ValueError(f"Unsupported model backbone: {model_backbone}")
            
        self.backbone = models.resnet18(weights=weights)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, num_classes)
        )
        self.classifier = self.backbone.fc

    def forward(self, x):
        return self.backbone(x)


def is_state_dict(obj: Any) -> bool:
    """
    Determine if the object is a PyTorch state dictionary.
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if the object appears to be a state dictionary
    """
    if not isinstance(obj, dict):
        return False
    
    # Check if it has typical state_dict structure (keys are strings, values are tensors)
    return all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items())


def load_model(model_path: str, 
               num_classes: int = 3, 
               dropout: float = 0.5,
               model_backbone: str = "resnet18",
               device: str = "cpu") -> nn.Module:
    """
    Load a PyTorch model from a file, detecting whether it's a state dict or complete model.
    
    Args:
        model_path: Path to the model file
        num_classes: Number of output classes (used if creating a new model)
        dropout: Dropout rate (used if creating a new model)
        model_backbone: Model backbone architecture (used if creating a new model)
        device: Device to load the model on
        
    Returns:
        nn.Module: Loaded PyTorch model
    """
    device = torch.device(device)
    
    try:
        # First try to load the file
        loaded_obj = torch.load(model_path, map_location=device)
        
        # Check if it's a state dict
        if is_state_dict(loaded_obj):
            print(f"Detected state dictionary at {model_path}")
            # Create a new model and load the state dict
            model = PreTrainedClassifier(
                num_classes=num_classes,
                dropout=dropout,
                pretrained=False,  # We're loading weights, so no need for pretrained
                model_backbone=model_backbone
            )
            model.load_state_dict(loaded_obj)
        else:
            # Assume it's a complete model
            print(f"Detected complete model at {model_path}")
            model = loaded_obj
            
        model.to(device)
        model.eval()  # Set to evaluation mode
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


def convert_to_onnx(model: nn.Module, 
                   onnx_path: str, 
                   input_shape: tuple = (1, 3, 224, 224),
                   device: str = "cpu",
                   opset_version: int = 13,
                   dynamic_batch: bool = True,
                   verify: bool = True):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        onnx_path: Path to save the ONNX model
        input_shape: Shape of the input tensor
        device: Device to use for conversion
        opset_version: ONNX opset version
        dynamic_batch: Whether to use dynamic batch size
        verify: Whether to verify the ONNX model after conversion
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Set up dynamic axes if requested
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    
    # Export to ONNX
    print(f"Converting model to ONNX format...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True, 
        opset_version=opset_version,
        do_constant_folding=True, 
        input_names=['input'],
        output_names=['output'], 
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX model saved to {onnx_path}")
    
    # Verify the model if requested
    if verify:
        verify_onnx_model(onnx_path, dummy_input, model)


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor, torch_model: nn.Module):
    """
    Verify the ONNX model produces the same output as PyTorch.
    
    Args:
        onnx_path: Path to the ONNX model
        dummy_input: Input tensor used for verification
        torch_model: Original PyTorch model
    """
    print("Verifying ONNX model...")
    
    # Check that the model is well-formed
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Compare ONNX Runtime and PyTorch results
    torch_model.eval()
    with torch.no_grad():
        torch_output = torch_model(dummy_input).cpu().numpy()
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Run the model on the backend
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare the results
    try:
        np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-03, atol=1e-05)
        print("✅ ONNX model verified successfully!")
    except AssertionError as e:
        print(f"❌ ONNX model verification failed: {e}")
        print(f"PyTorch output shape: {torch_output.shape}, ONNX output shape: {ort_outputs[0].shape}")
        print(f"PyTorch output (first few values): {torch_output.flatten()[:5]}")
        print(f"ONNX output (first few values): {ort_outputs[0].flatten()[:5]}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    
    # Input/Output paths
    parser.add_argument('--torch-model-path', type=str, required=True,
                        help='Path to the PyTorch model or state dict')
    parser.add_argument('--onnx-model-path', type=str, required=True,
                        help='Path to save the ONNX model')
    
    # Model configuration
    parser.add_argument('--model-backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnetb1', 'efficientnetb4'],
                        help='Model backbone architecture (used if loading state dict)')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of output classes (used if loading state dict)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (used if loading state dict)')
    
    # Input shape
    parser.add_argument('--input-height', type=int, default=224,
                        help='Input image height')
    parser.add_argument('--input-width', type=int, default=224,
                        help='Input image width')
    parser.add_argument('--input-channels', type=int, default=3,
                        help='Input image channels')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for the input shape')
    
    # ONNX export parameters
    parser.add_argument('--opset-version', type=int, default=13,
                        help='ONNX opset version')
    parser.add_argument('--no-dynamic-batch', action='store_true',
                        help='Disable dynamic batch size in ONNX model')
    
    # Device and verification
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for model loading and conversion')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip ONNX model verification')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load the PyTorch model
    model = load_model(
        model_path=args.torch_model_path,
        num_classes=args.num_classes,
        dropout=args.dropout,
        model_backbone=args.model_backbone,
        device=args.device
    )
    
    # Convert to ONNX
    convert_to_onnx(
        model=model,
        onnx_path=args.onnx_model_path,
        input_shape=(args.batch_size, args.input_channels, args.input_height, args.input_width),
        device=args.device,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
        verify=not args.no_verify
    )
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
