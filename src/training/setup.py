"""
Optimizer and scheduler setup.
Handles optimizer initialization and learning rate scheduling configuration.
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config.config import (
    LEARNING_RATE, LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA, WEIGHT_DECAY
)


def create_optimizer(
    model: nn.Module,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY
) -> optim.Optimizer:
    """
    Create Adam optimizer for model training.
    
    Args:
        model: Model parameters to optimize.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
    
    Returns:
        optim.Optimizer: Configured optimizer.
    """
    return optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    step_size: int = LR_SCHEDULER_STEP_SIZE,
    gamma: float = LR_SCHEDULER_GAMMA
) -> StepLR:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule.
        step_size: Period of learning rate decay.
        gamma: Multiplicative factor of learning rate decay.
    
    Returns:
        StepLR: Configured learning rate scheduler.
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
