"""
Device management utilities.
Handles GPU/CPU device configuration and detection.
"""

import torch


def get_device(
    device_type: str = "cuda",
    verbose: bool = True
) -> torch.device:
    """
    Get PyTorch device for computation.
    
    Args:
        device_type (str): Preferred device ("cuda" or "cpu").
        verbose (bool): Whether to print device information.
    
    Returns:
        torch.device: Device object (CUDA if available, else CPU).
    """
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"CUDA is available. Using: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("CUDA not available. Using CPU.")
    
    return device
