"""
Evaluation script for Steering Angle Prediction model.
Loads a trained model and evaluates it on test data.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from config.config import (
    DATA_DIR, TEST_DATA_DIR, MODEL_CHECKPOINT_PATH,
    INPUT_CHANNELS, HIDDEN_SIZE, OUTPUT_SIZE, DEVICE_TYPE, LOG_LEVEL
)
from src.data import create_dataloaders, SteeringDataset, get_transforms
from src.models import create_model
from src.evaluation import Evaluator
from src.utils import (
    setup_logging, get_device, load_model, get_model_summary
)
from torch.utils.data import DataLoader


logger = logging.getLogger('steering_angle')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Steering Angle Prediction Model"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(TEST_DATA_DIR),
        help='Path to test dataset directory'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=str(MODEL_CHECKPOINT_PATH),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=DEVICE_TYPE,
        help='Device to evaluate on'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    setup_logging(log_level=LOG_LEVEL)
    
    logger.info("Starting evaluation pipeline...")
    logger.info(f"Configuration: {vars(args)}")
    
    device = get_device(args.device, verbose=True)
    
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info("Creating model...")
    model = create_model(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        device=device
    )
    
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    model = load_model(model, str(checkpoint_path), device)
    logger.info("Model loaded successfully")
    logger.info(get_model_summary(model))
    
    logger.info("Loading test data...")
    try:
        transform = get_transforms(is_training=False)
        dataset = SteeringDataset(
            root_dir=args.data_dir,
            transform=transform
        )
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        logger.info(f"Test data loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise
    
    criterion = nn.MSELoss()
    
    logger.info("Starting evaluation...")
    evaluator = Evaluator(model, criterion, device)
    metrics = evaluator.evaluate(test_loader)
    evaluator.log_metrics(metrics)
    
    logger.info("Evaluation completed successfully!")
    
    return metrics


if __name__ == "__main__":
    main()
