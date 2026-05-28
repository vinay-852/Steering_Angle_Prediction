"""
Dataset class and data loading utilities.
Handles image preprocessing and steering angle normalization.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config.config import ANGLE_MAX_VALUE, ANGLE_CONVERSION_FACTOR


class SteeringDataset(Dataset):
    """
    Custom PyTorch Dataset for steering angle prediction.
    
    Loads images and corresponding steering angles from a directory structure.
    Expects a data.txt file with format: image_filename angle_value (space-separated).
    
    Attributes:
        root_dir (Path): Root directory containing images and data.txt.
        transform (transforms.Compose): Image transformation pipeline.
        data (List[Tuple]): List of (image_path, angle) tuples.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Path to directory containing images and data.txt.
            transform (transforms.Compose, optional): Image transformations to apply.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data = self._load_data_file()
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and normalized angle.
        """
        img_name, angle = self.data[idx]
        img_path = self.root_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        
        normalized_angle = self._normalize_angle(angle)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(normalized_angle, dtype=torch.float32)
    
    def _load_data_file(self) -> List[Tuple[str, float]]:
        """
        Load image-angle pairs from data.txt file.
        
        Expected format: image_filename angle_value (space-separated, one per line)
        
        Returns:
            List[Tuple[str, float]]: List of (image_filename, angle) pairs.
        
        Raises:
            FileNotFoundError: If data.txt is not found in root_dir.
        """
        data_file = self.root_dir / 'data.txt'
        
        if not data_file.exists():
            raise FileNotFoundError(f"data.txt not found in {self.root_dir}")
        
        data = []
        with open(data_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        angle = float(parts[1])
                        data.append((img_name, angle))
        
        return data
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize steering angle to a standardized range.
        
        Args:
            angle (float): Raw steering angle value.
        
        Returns:
            float: Normalized angle in radians.
        """
        normalized = ((angle + 450) % ANGLE_MAX_VALUE) / ANGLE_MAX_VALUE
        return normalized * ANGLE_CONVERSION_FACTOR
