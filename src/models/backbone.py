"""
Backbone model loading and configuration.
Handles transfer learning setup and model factory functions.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .architecture import SteeringCNN


def get_resnet_backbone(
    model_name: str = "resnet18",
    pretrained: bool = True,
    remove_fc: bool = True
) -> nn.Module:
    """
    Load a ResNet backbone model for transfer learning.
    
    Args:
        model_name (str): Name of ResNet model ("resnet18", "resnet34", etc.).
        pretrained (bool): Whether to load ImageNet pre-trained weights.
        remove_fc (bool): Whether to remove the fully connected layer.
    
    Returns:
        nn.Module: ResNet backbone model without the classification head.
    
    Example:
        >>> backbone = get_resnet_backbone("resnet18", pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = backbone(x)  # Shape: (1, 512, 7, 7) for resnet18
    """
    model = models.__dict__[model_name](pretrained=pretrained)
    
    if remove_fc:
        backbone = nn.Sequential(*list(model.children())[:-2])
    else:
        backbone = nn.Sequential(*list(model.children())[:-1])
    
    return backbone


def create_model(
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    input_channels: int = 512,
    hidden_size: int = 256,
    output_size: int = 1,
    dropout_rate: float = 0.5,
    device: torch.device = None
) -> nn.Module:
    """
    Factory function to create a Steering CNN model.
    
    Args:
        backbone_name: Name of the backbone architecture.
        pretrained: Use pre-trained weights.
        input_channels: Number of input channels from backbone.
        hidden_size: Number of hidden units.
        output_size: Output dimension.
        dropout_rate: Dropout probability.
        device: Device to place the model on (CPU or CUDA).
    
    Returns:
        nn.Module: Configured SteeringCNN model.
    
    Example:
        >>> model = create_model(device=torch.device("cuda"))
        >>> model.eval()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone = get_resnet_backbone(backbone_name, pretrained)
    model = SteeringCNN(
        backbone=backbone,
        input_shape=input_channels,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    )
    
    return model.to(device)
