"""
Configuration module for Steering Angle Prediction Model.
Centralized configuration management following standard ML practices.
"""

from pathlib import Path
from typing import Tuple

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "driving-dataset" / "driving_dataset"
TEST_DATA_DIR = PROJECT_ROOT / "Test_Images"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model save/load paths
MODEL_CHECKPOINT_PATH = MODELS_DIR / "steering_model.pth"
MODEL_BEST_PATH = MODELS_DIR / "steering_model_best.pth"

# ============================================================================
# Model Architecture Configuration
# ============================================================================
# ResNet18 backbone configuration
BACKBONE_NAME: str = "resnet18"
PRETRAINED: bool = True

# Spatial Pyramid Pooling configuration
SPP_POOL_LIST: list = [1, 2, 4]

# Model layer dimensions
INPUT_CHANNELS: int = 512  # ResNet18 output channels
HIDDEN_SIZE: int = 256
OUTPUT_SIZE: int = 1
DROPOUT_RATE: float = 0.5

# ============================================================================
# Data Configuration
# ============================================================================
# Image preprocessing
IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ============================================================================
# Training Configuration
# ============================================================================
# Training hyperparameters
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 70
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 1e-5

# Learning rate scheduler
LR_SCHEDULER_STEP_SIZE: int = 10
LR_SCHEDULER_GAMMA: float = 0.1

# Data split ratios
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1

# Training flags
USE_MIXED_PRECISION: bool = False
GRADIENT_CLIP_VALUE: float = 1.0
EARLY_STOPPING_PATIENCE: int = 15

# ============================================================================
# Angle Normalization Configuration
# ============================================================================
# Angle normalization parameters
ANGLE_MAX_VALUE: float = 900.0  # Maximum angle value for normalization
ANGLE_CONVERSION_FACTOR: float = 3.141592653589793  # pi value for radians conversion

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE_TYPE: str = "cuda"  # or "cpu"
NUM_WORKERS: int = 4
PIN_MEMORY: bool = True

# ============================================================================
# Loss Function Configuration
# ============================================================================
LOSS_FUNCTION: str = "mse"  # mean squared error
OPTIMIZER_TYPE: str = "adam"  # adam optimizer

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SAVE_LOGS_TO_FILE: bool = True

# ============================================================================
# Evaluation Configuration
# ============================================================================
METRICS_TO_TRACK: list = ["MSE", "MAE", "R2"]
SAVE_PREDICTIONS: bool = True
