"""
Logger utility module for the ResNet image classification project.

This module provides a centralized logging configuration that supports both
console and file logging with proper formatting and timestamp management.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import sys


class Logger:
    """
    A centralized logger class that configures and manages logging for the project.
    
    This class sets up both console and file handlers with consistent formatting,
    ensuring logs are properly timestamped and categorized by severity level.
    """
    
    def __init__(
        self,
        name: str = "ResNetClassifier",
        log_dir: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the logger with console and file handlers.
        
        Args:
            name: Name of the logger instance
            log_dir: Directory to save log files (optional)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_dir is provided)
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir_path / "training.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """
        Return the configured logger instance.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        return self.logger
    
    def log_system_info(self):
        """
        Log system information including Python version and available devices.
        """
        import torch
        import platform
        
        self.logger.info("=" * 50)
        self.logger.info("System Information")
        self.logger.info("=" * 50)
        self.logger.info(f"Python Version: {platform.python_version()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"Torchvision Version: {torchvision.__version__ if 'torchvision' in sys.modules else 'N/A'}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                self.logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        else:
            self.logger.info("Running on CPU")
        self.logger.info("=" * 50)


def setup_logger(
    name: str = "ResNetClassifier",
    log_dir: Optional[str] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Convenience function to set up and return a configured logger.
    
    Args:
        name: Name of the logger instance
        log_dir: Directory to save log files (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger_instance = Logger(name=name, log_dir=log_dir, log_level=log_level)
    return logger_instance.get_logger()


if __name__ == "__main__":
    # Test the logger
    logger = setup_logger(log_dir="outputs/logs", log_level="DEBUG")
    logger.info("Logger initialized successfully")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.log_system_info()
