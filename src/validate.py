"""
Validation script for the ResNet image classification project.

This script loads a trained model checkpoint and evaluates it on the validation dataset,
generating metrics and confusion matrices.
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.metrics import MetricsCalculator
from utils.plots import PlotGenerator
from datasets.dataset import create_dataloaders
from models.model import create_model


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config.yaml file
    
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate(config_path: str, checkpoint_path: Optional[str] = None):
    """
    Main validation function.
    
    Args:
        config_path: Path to the config.yaml file
        checkpoint_path: Path to the checkpoint file (optional, uses best_model.pth if not provided)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    log_dir = config['output_directories']['logs']
    logger = setup_logger(
        name="ResNetValidator",
        log_dir=log_dir,
        log_level=config['logging']['log_level']
    )
    
    logger.info("=" * 50)
    logger.info("Starting Validation")
    logger.info("=" * 50)
    
    # Determine device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    
    logger.info(f"Using device: {device}")
    
    # Create output directories
    for dir_name in config['output_directories'].values():
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    logger.info("Creating validation dataloader...")
    _, val_loader, _, class_to_idx = create_dataloaders(
        train_csv=config['paths']['train_csv'],
        val_csv=config['paths']['val_csv'],
        test_csv=config['paths']['test_csv'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        image_size=config['dataset']['image_size'],
        normalization_mean=config['normalization']['mean'],
        normalization_std=config['normalization']['std'],
        pin_memory=True if device == 'cuda' else False
    )
    
    if val_loader is None:
        logger.error("Validation dataloader could not be created. Check val_csv path.")
        return
    
    num_classes = len(class_to_idx)
    class_names = list(class_to_idx.keys())
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        resnet_variant=config['model']['resnet_variant'],
        num_classes=num_classes,
        pretrained=False
    )
    
    # Move model to device
    model = model.to(device)
    
    # Handle multi-GPU
    if device == 'cuda' and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config['output_directories']['checkpoints']) / 'best_model.pth'
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully")
    
    # Get epoch from checkpoint
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Checkpoint from epoch: {epoch}")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    
    # Validation loop
    logger.info("Starting validation...")
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            # Track loss
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_probabilities = np.concatenate(all_probabilities)
    
    # Calculate average loss
    avg_loss = total_loss / total_samples
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = metrics_calculator.compute_all_metrics(
        predictions=all_predictions,
        targets=all_targets,
        probabilities=all_probabilities
    )
    
    # Log metrics
    logger.info("=" * 50)
    logger.info("Validation Results")
    logger.info("=" * 50)
    logger.info(f"Loss: {avg_loss:.4f}")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Get per-class metrics
    per_class_metrics = metrics_calculator.get_per_class_metrics(
        predictions=all_predictions,
        targets=all_targets
    )
    
    logger.info("\nPer-class Metrics:")
    for class_name, class_metrics in per_class_metrics.items():
        logger.info(
            f"{class_name}: Precision={class_metrics['precision']:.4f}, "
            f"Recall={class_metrics['recall']:.4f}, "
            f"F1={class_metrics['f1-score']:.4f}, "
            f"Support={class_metrics['support']}"
        )
    
    # Compute confusion matrix
    logger.info("Computing confusion matrix...")
    cm = metrics_calculator.compute_confusion_matrix(
        predictions=all_predictions,
        targets=all_targets,
        normalize=None
    )
    
    # Generate plots
    logger.info("Generating plots...")
    plot_generator = PlotGenerator(output_dir=config['output_directories']['metrics'])
    
    plot_generator.plot_confusion_matrix(
        confusion_matrix=cm,
        class_names=class_names,
        normalize=False,
        save_name='validation_confusion_matrix.png'
    )
    
    plot_generator.plot_confusion_matrix(
        confusion_matrix=cm,
        class_names=class_names,
        normalize=True,
        save_name='validation_confusion_matrix_normalized.png'
    )
    
    plot_generator.plot_per_class_metrics(
        per_class_metrics=per_class_metrics,
        save_name='validation_per_class_metrics.png'
    )
    
    # Save metrics to JSON
    metrics_output = {
        'epoch': epoch,
        'loss': avg_loss,
        **metrics,
        'per_class_metrics': per_class_metrics
    }
    
    metrics_path = Path(config['output_directories']['metrics']) / 'validation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        Path(config['output_directories']['metrics']) / 'validation_metrics.csv',
        index=False
    )
    
    logger.info("=" * 50)
    logger.info("Validation completed successfully")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate ResNet image classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (optional)'
    )
    
    args = parser.parse_args()
    
    validate(config_path=args.config, checkpoint_path=args.checkpoint)
