"""
Data loaders and batch creation.
Handles DataLoader setup for training and evaluation.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from config.config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
from .dataset import SteeringDataset
from .transforms import get_transforms


def create_dataloaders(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir (str): Path to dataset directory.
        batch_size (int): Number of samples per batch.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        shuffle_train (bool): Whether to shuffle training data.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    
    Raises:
        ValueError: If train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)
    
    dataset = SteeringDataset(root_dir=data_dir, transform=train_transform)
    
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
