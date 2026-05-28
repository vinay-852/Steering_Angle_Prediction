"""
Data transformations and augmentations.
Handles image preprocessing and augmentation pipelines.
"""

import torchvision.transforms as transforms
from config.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get image transformation pipeline.
    
    Args:
        is_training (bool): If True, applies augmentation transforms for training.
                           If False, applies only normalization for evaluation.
    
    Returns:
        transforms.Compose: Composition of image transformations.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
