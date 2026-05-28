"""
Evaluation metrics computation.
Computes performance metrics for model evaluation.
"""

import numpy as np


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        float: MSE value.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        float: MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        float: RMSE value.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² Score (Coefficient of Determination).
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
    
    Returns:
        float: R² score in range [0, 1].
    """
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 0.0
    
    r2 = 1 - (ss_residual / ss_total)
    return float(np.clip(r2, 0, 1))
