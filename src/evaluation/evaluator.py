"""
Model evaluator and evaluation pipeline.
Handles model evaluation and metrics computation.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .metrics import compute_mse, compute_mae, compute_rmse, compute_r2


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for computing model performance metrics.
    
    Computes:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - R² Score
        - Test Loss
    
    Attributes:
        model (nn.Module): Model to evaluate.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device = None
    ):
        """
        Initialize the Evaluator.
        
        Args:
            model: Neural network model to evaluate.
            criterion: Loss function for computing loss.
            device: Device to evaluate on. Defaults to GPU if available.
        """
        self.model = model
        self.criterion = criterion
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader (DataLoader): Test data loader.
        
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        
        y_true = []
        y_pred = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, angles in tqdm(
                test_loader,
                desc="Evaluating",
                leave=False
            ):
                images = images.to(self.device)
                angles = angles.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, angles)
                
                y_true.extend(angles.cpu().numpy().flatten())
                y_pred.extend(outputs.cpu().numpy().flatten())
                total_loss += loss.item()
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'test_loss': total_loss / len(test_loader),
            'mse': compute_mse(y_true, y_pred),
            'mae': compute_mae(y_true, y_pred),
            'r2': compute_r2(y_true, y_pred),
            'rmse': compute_rmse(y_true, y_pred)
        }
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary of computed metrics.
        """
        logger.info("=" * 50)
        logger.info("Evaluation Metrics")
        logger.info("=" * 50)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.6f}")
        logger.info("=" * 50)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module = None,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Convenient function to evaluate a model.
    
    Args:
        model: Model to evaluate.
        test_loader: Test data loader.
        criterion: Loss function (defaults to MSE if None).
        device: Device to evaluate on.
    
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    evaluator = Evaluator(model, criterion, device)
    metrics = evaluator.evaluate(test_loader)
    evaluator.log_metrics(metrics)
    
    return metrics
