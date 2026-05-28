"""
Helper utilities for model management and configuration.
Contains utility functions for model operations and reproducibility.
"""

from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model.
    
    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary string of model architecture and parameters.
    
    Args:
        model (nn.Module): PyTorch model.
    
    Returns:
        str: Model summary string.
    """
    total_params = count_parameters(model)
    
    summary = f"""
    Model Summary:
    {'=' * 50}
    Total Parameters: {total_params:,}
    Trainable Parameters: {total_params:,}
    {'=' * 50}
    """
    
    return summary


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config_to_file(config_dict: dict, filepath: str) -> None:
    """
    Save configuration dictionary to a text file.
    
    Args:
        config_dict (dict): Configuration parameters.
        filepath (str): Path to save configuration file.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("Configuration:\n")
        f.write("=" * 50 + "\n")
        for key, value in sorted(config_dict.items()):
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n")


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Load model weights from checkpoint.
    
    Args:
        model (nn.Module): Model to load weights into.
        checkpoint_path (str): Path to checkpoint file.
        device (torch.device): Device to load model on.
    
    Returns:
        nn.Module: Model with loaded weights.
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    return model
