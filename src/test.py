"""
Test script for the ResNet image classification project.

This script loads a trained model checkpoint and evaluates it on the test dataset,
generating comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrices, and prediction CSV.
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
from utils.output_manager import create_timestamped_output_dirs, get_latest_output_dir
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


def test(config_path: str, checkpoint_path: Optional[str] = None):
    """
    Main test function.
    
    Args:
        config_path: Path to the config.yaml file
        checkpoint_path: Path to the checkpoint file (optional, uses best_model.pth if not provided)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create timestamped output directories
    output_dirs = create_timestamped_output_dirs(base_output_dir=config['output_base_dir'])
    
    # Setup logger
    log_dir = output_dirs['logs']
    logger = setup_logger(
        name="ResNetTester",
        log_dir=log_dir,
        log_level=config['logging']['log_level']
    )
    
    logger.info("=" * 50)
    logger.info("Starting Testing")
    logger.info("=" * 50)
    logger.info(f"Created timestamped output directories in: {Path(config['output_base_dir']) / Path(output_dirs['logs']).parent.name}")
    
    # Determine device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    
    logger.info(f"Using device: {device}")
    
    # Output directories already created by create_timestamped_output_dirs
    
    # Create dataloaders
    logger.info("Creating test dataloader...")
    _, _, test_loader, class_to_idx = create_dataloaders(
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
    
    if test_loader is None:
        logger.error("Test dataloader could not be created. Check test_csv path.")
        return
    
    num_classes = len(class_to_idx)
    class_names = list(class_to_idx.keys())
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
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
        # Try to get the latest output directory and use its checkpoint
        latest_dir = get_latest_output_dir(base_output_dir=config['output_base_dir'])
        if latest_dir:
            checkpoint_path = Path(latest_dir) / 'checkpoints' / 'best_model.pth'
            logger.info(f"Using checkpoint from latest run: {checkpoint_path}")
        else:
            checkpoint_path = Path(config['output_base_dir']) / 'checkpoints' / 'best_model.pth'
            logger.info(f"Using checkpoint from default location: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state - handle both DataParallel and non-DataParallel checkpoints
    state_dict = checkpoint['model_state_dict']
    
    # Check if state_dict has 'module.' prefix (from DataParallel)
    has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
    
    # Strip 'module.' prefix if present
    if has_module_prefix:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict
        logger.info("Stripped 'module.' prefix from checkpoint state_dict")
    
    # Load state dict
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    logger.info("Model loaded successfully")
    
    # Get epoch from checkpoint
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Checkpoint from epoch: {epoch}")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    
    # Test loop
    logger.info("Starting testing...")
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_image_paths = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
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
    
    # Calculate comprehensive metrics
    logger.info("Calculating comprehensive metrics...")
    metrics = metrics_calculator.compute_all_metrics(
        predictions=all_predictions,
        targets=all_targets,
        probabilities=all_probabilities
    )
    
    # Log metrics
    logger.info("=" * 50)
    logger.info("Test Results")
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
    
    # Get classification report
    classification_report_str = metrics_calculator.get_classification_report(
        predictions=all_predictions,
        targets=all_targets,
        output_dict=False
    )
    logger.info("\nClassification Report:")
    logger.info(classification_report_str)
    
    # Compute confusion matrices
    logger.info("Computing confusion matrices...")
    cm = metrics_calculator.compute_confusion_matrix(
        predictions=all_predictions,
        targets=all_targets,
        normalize=None
    )
    
    cm_normalized = metrics_calculator.compute_confusion_matrix(
        predictions=all_predictions,
        targets=all_targets,
        normalize='true'
    )
    
    # Generate plots
    logger.info("Generating plots...")
    plot_generator = PlotGenerator(output_dir=output_dirs['metrics'])
    
    plot_generator.plot_confusion_matrix(
        confusion_matrix=cm,
        class_names=class_names,
        normalize=False,
        save_name='test_confusion_matrix.png'
    )
    
    plot_generator.plot_confusion_matrix(
        confusion_matrix=cm_normalized,
        class_names=class_names,
        normalize=True,
        save_name='test_confusion_matrix_normalized.png'
    )
    
    plot_generator.plot_per_class_metrics(
        per_class_metrics=per_class_metrics,
        save_name='test_per_class_metrics.png'
    )
    
    # Save predictions to CSV
    logger.info("Saving predictions to CSV...")
    predictions_df = pd.DataFrame({
        'true_label': [idx_to_class[idx] for idx in all_targets],
        'predicted_label': [idx_to_class[idx] for idx in all_predictions],
        'correct': [all_targets[i] == all_predictions[i] for i in range(len(all_predictions))]
    })
    
    # Add probability columns
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = all_probabilities[:, i]
    
    predictions_path = Path(output_dirs['predictions']) / 'test_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Save metrics to JSON
    metrics_output = {
        'epoch': epoch,
        'loss': avg_loss,
        **metrics,
        'per_class_metrics': per_class_metrics,
        'classification_report': classification_report_str
    }
    
    metrics_path = Path(output_dirs['metrics']) / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        Path(output_dirs['metrics']) / 'test_metrics.csv',
        index=False
    )
    
    # Save per-class metrics to CSV
    per_class_df = pd.DataFrame(per_class_metrics).T
    per_class_df.to_csv(
        Path(output_dirs['metrics']) / 'test_per_class_metrics.csv',
        index_label='class'
    )
    
    logger.info("=" * 50)
    logger.info("Testing completed successfully")
    logger.info("=" * 50)
    logger.info(f"Test samples: {total_samples}")
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1-score (macro): {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ResNet image classifier')
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
    
    test(config_path=args.config, checkpoint_path=args.checkpoint)
