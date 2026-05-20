"""
Model module for the ResNet image classification project.

This module provides functions to create and configure ResNet models
for image classification, with support for different variants and
custom number of classes.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, List
import logging


class ResNetClassifier(nn.Module):
    """
    A ResNet-based classifier for image classification.
    
    This class wraps torchvision ResNet models and provides
    a flexible interface for different ResNet variants.
    """
    
    def __init__(
        self,
        resnet_variant: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the ResNet classifier.
        
        Args:
            resnet_variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(ResNetClassifier, self).__init__()
        
        self.resnet_variant = resnet_variant
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.logger = logging.getLogger(__name__)
        
        # Load the appropriate ResNet model
        self.model = self._load_resnet_variant(resnet_variant, pretrained)
        
        # Get the number of input features for the final layer
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        self.logger.info(f"Initialized {resnet_variant} with {num_classes} classes")
        self.logger.info(f"Pretrained: {pretrained}, Frozen backbone: {freeze_backbone}")
    
    def _load_resnet_variant(self, variant: str, pretrained: bool) -> nn.Module:
        """
        Load the specified ResNet variant.
        
        Args:
            variant: ResNet variant name
            pretrained: Whether to use pretrained weights
        
        Returns:
            ResNet model
        """
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if variant not in resnet_models:
            raise ValueError(
                f"Invalid ResNet variant: {variant}. "
                f"Must be one of: {list(resnet_models.keys())}"
            )
        
        weights = 'DEFAULT' if pretrained else None
        model = resnet_models[variant](weights=weights)
        
        return model
    
    def _freeze_backbone(self):
        """Freeze the backbone layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final layer
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        self.logger.info("Backbone layers frozen, only final layer trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.logger.info("All layers unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'resnet_variant': self.resnet_variant,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
        return info


def create_model(
    resnet_variant: str = 'resnet50',
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> ResNetClassifier:
    """
    Create a ResNet classifier model.
    
    Args:
        resnet_variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone layers
    
    Returns:
        ResNetClassifier instance
    """
    model = ResNetClassifier(
        resnet_variant=resnet_variant,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    return model


def get_available_resnet_variants() -> List[str]:
    """
    Get list of available ResNet variants.
    
    Returns:
        List of available ResNet variant names
    """
    return ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


if __name__ == "__main__":
    # Test the model
    import torch
    
    # Create a model
    model = create_model(
        resnet_variant='resnet50',
        num_classes=10,
        pretrained=False
    )
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    
    # Test with different variants
    print("\nAvailable ResNet variants:")
    for variant in get_available_resnet_variants():
        print(f"  - {variant}")
