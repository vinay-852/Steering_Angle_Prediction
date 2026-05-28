"""
Main training script for Steering Angle Prediction model.
Orchestrates the complete training pipeline.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from config.config import (
    DATA_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    INPUT_CHANNELS, HIDDEN_SIZE, OUTPUT_SIZE, DEVICE_TYPE,
    MODEL_CHECKPOINT_PATH, LOG_LEVEL
)
from src.data import create_dataloaders
from src.models import create_model
from src.training import Trainer, create_optimizer, create_scheduler
from src.utils import (
    setup_logging, get_device, seed_everything, get_model_summary,
    count_parameters
)


logger = logging.getLogger('steering_angle')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Steering Angle Prediction Model"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(DATA_DIR),
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=DEVICE_TYPE,
        help='Device to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=str(MODEL_CHECKPOINT_PATH),
        help='Path to save model checkpoints'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    setup_logging(log_level=LOG_LEVEL)
    
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration: {vars(args)}")
    
    seed_everything(args.seed)
    
    device = get_device(args.device, verbose=True)
    
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        logger.info(
            f"Data loaded: "
            f"{len(train_loader.dataset)} train, "
            f"{len(val_loader.dataset)} val, "
            f"{len(test_loader.dataset)} test samples"
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    logger.info("Creating model...")
    model = create_model(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        device=device
    )
    
    total_params = count_parameters(model)
    logger.info(f"Model created with {total_params:,} parameters")
    logger.info(get_model_summary(model))
    
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, learning_rate=args.learning_rate)
    scheduler = create_scheduler(optimizer)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )
    
    logger.info("Starting training...")
    history = trainer.train(num_epochs=args.epochs)
    
    trainer.save_checkpoint(args.checkpoint_path)
    logger.info(f"Model saved to {args.checkpoint_path}")
    
    logger.info("Training completed successfully!")
    
    return model, trainer, history


if __name__ == "__main__":
    main()
