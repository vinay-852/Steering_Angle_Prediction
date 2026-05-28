"""
Training module for Steering Angle Prediction.
Handles model training, validation, and checkpoint management.
"""

from .trainer import Trainer
from .setup import create_optimizer, create_scheduler

__all__ = [
    "Trainer",
    "create_optimizer",
    "create_scheduler",
]
