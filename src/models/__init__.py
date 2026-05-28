"""
Models module for Steering Angle Prediction.
Contains neural network architectures and backbone configurations.
"""

from .architecture import SpatialPyramidPooling, SteeringCNN
from .backbone import get_resnet_backbone, create_model

__all__ = [
    "SpatialPyramidPooling",
    "SteeringCNN",
    "get_resnet_backbone",
    "create_model",
]
