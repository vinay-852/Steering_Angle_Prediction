"""
Neural network architectures for Steering Angle Prediction.
Contains model components like SPP and main CNN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling layer.
    
    Enables variable input sizes by pooling features at multiple scales,
    making the network invariant to object scale while preserving spatial hierarchy.
    
    Args:
        pool_list (List[int]): List of pooling levels (output spatial sizes).
                              e.g., [1, 2, 4] creates 1x1, 2x2, and 4x4 pools.
    
    Example:
        >>> spp = SpatialPyramidPooling([1, 2, 4])
        >>> x = torch.randn(32, 512, 14, 14)
        >>> output = spp(x)  # Shape: (32, 512 * 21)
    """
    
    def __init__(self, pool_list: List[int]):
        """Initialize the SPP layer with pool sizes."""
        super(SpatialPyramidPooling, self).__init__()
        self.pool_list = pool_list
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SPP layer.
        
        Args:
            x (torch.Tensor): Input feature maps of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Flattened pooled features of shape (batch_size, channels * sum(pool_sizes)).
        """
        batch_size, num_channels, _, _ = x.size()
        pooled_outputs = []
        
        for pool_size in self.pool_list:
            pooled = F.adaptive_max_pool2d(x, output_size=(pool_size, pool_size))
            pooled = pooled.view(batch_size, num_channels, -1)
            pooled_outputs.append(pooled)
        
        output = torch.cat(pooled_outputs, dim=2)
        output = output.view(batch_size, -1)
        
        return output


class SteeringCNN(nn.Module):
    """
    CNN architecture for steering angle prediction.
    
    Combines ResNet backbone, Spatial Pyramid Pooling, and fully connected layers
    to predict steering angles from input images.
    
    Architecture:
        1. Backbone (ResNet18): Feature extraction
        2. SPP Layer: Multi-scale spatial pooling
        3. Fully Connected Layers: Steering angle regression
    
    Args:
        backbone (nn.Module): Pre-trained backbone model (e.g., ResNet18).
        input_shape (int): Number of input channels from backbone.
        hidden_size (int): Number of hidden units in first FC layer.
        output_size (int): Output dimension (1 for single steering angle).
    
    Example:
        >>> backbone = get_resnet_backbone()
        >>> model = SteeringCNN(backbone, 512, 256, 1)
        >>> x = torch.randn(32, 3, 224, 224)
        >>> output = model(x)  # Shape: (32, 1)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        input_shape: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the Steering CNN model.
        
        Args:
            backbone: Pre-trained CNN backbone for feature extraction.
            input_shape: Number of channels in SPP output.
            hidden_size: Number of neurons in first dense layer.
            output_size: Number of output neurons (1 for steering angle).
            dropout_rate: Dropout probability for regularization.
        """
        super(SteeringCNN, self).__init__()
        self.backbone = backbone
        self.spp = SpatialPyramidPooling(pool_list=[1, 2, 4])
        
        spp_output_size = input_shape * (1 + 4 + 16)
        
        self.fc1 = nn.Linear(spp_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming normal distribution."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224).
        
        Returns:
            torch.Tensor: Predicted steering angles of shape (batch_size, 1).
        """
        x = self.backbone(x)
        x = self.spp(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
