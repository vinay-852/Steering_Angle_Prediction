"""
Training pipeline and trainer class.
Handles model training, validation, and checkpoint management.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config.config import (
    GRADIENT_CLIP_VALUE, EARLY_STOPPING_PATIENCE, MODEL_CHECKPOINT_PATH,
    MODEL_BEST_PATH
)


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for managing model training and validation loops.
    
    Handles:
        - Training loop with gradient updates
        - Validation loop with loss tracking
        - Learning rate scheduling
        - Model checkpointing
        - Early stopping
    
    Attributes:
        model (nn.Module): Neural network model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (optim.Optimizer): Optimization algorithm.
        scheduler (StepLR): Learning rate scheduler.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on (CPU or CUDA).
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = None,
        scheduler: Optional[StepLR] = None
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: Neural network model to train.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            optimizer: Optimizer for model parameters.
            criterion: Loss function.
            device: Device to train on. Defaults to GPU if available.
            scheduler: Learning rate scheduler (optional).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self) -> float:
        """
        Execute a single training epoch.
        
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for images, angles in tqdm(
            self.train_loader,
            desc="Training",
            leave=False
        ):
            images = images.to(self.device)
            angles = angles.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, angles)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                GRADIENT_CLIP_VALUE
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """
        Execute a validation loop.
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, angles in tqdm(
                self.val_loader,
                desc="Validating",
                leave=False
            ):
                images = images.to(self.device)
                angles = angles.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, angles)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Execute the complete training loop.
        
        Args:
            num_epochs (int): Number of epochs to train.
        
        Returns:
            Dict[str, List[float]]: Dictionary containing training and validation losses.
        """
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            
            if self.scheduler:
                self.scheduler.step()
            
            self._check_early_stopping(val_loss, epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(MODEL_BEST_PATH, is_best=True)
            
            if self._should_stop():
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1}"
                )
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def _check_early_stopping(self, val_loss: float, epoch: int) -> None:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss (float): Current validation loss.
            epoch (int): Current epoch number.
        """
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def _should_stop(self) -> bool:
        """
        Check if training should stop due to early stopping.
        
        Returns:
            bool: True if early stopping condition is met.
        """
        return self.patience_counter >= EARLY_STOPPING_PATIENCE
    
    def save_checkpoint(
        self,
        filepath: str,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath (str): Path to save the checkpoint.
            is_best (bool): Whether this is the best model checkpoint.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        
        if is_best:
            logger.info("New best model checkpoint saved!")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file.
        
        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
