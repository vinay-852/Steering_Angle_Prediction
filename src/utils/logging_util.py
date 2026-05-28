"""
Logging configuration and utilities.
Handles application logging setup.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: str = "steering_angle.log"
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_dir (str): Directory to save log files.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (str): Name of log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    log_file_path = log_dir_path / log_file
    
    logger = logging.getLogger('steering_angle')
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
