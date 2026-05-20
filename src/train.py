"""
Training script for the ResNet image classification project.

This script handles the complete training process including data loading,
model initialization, training loop, validation, checkpointing, early stopping,
and metric tracking.
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.seed import set_seed, get_worker_init_fn
from utils.metrics import MetricsCalculator, AverageMeter
from utils.checkpoint import CheckpointManager
from utils.early_stopping import EarlyStoppingCallback
from utils.plots import PlotGenerator
from utils.output_manager import create_timestamped_output_dirs
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


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
    
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    scheduler_params: Optional[Dict] = None
) -> Optional[object]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_name: Name of the scheduler ('cosine', 'step', 'plateau', 'none')
        scheduler_params: Optional parameters for the scheduler
    
    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == 'none':
        return None
    elif scheduler_name.lower() == 'cosine':
        T_max = scheduler_params.get('T_max', 100) if scheduler_params else 100
        eta_min = scheduler_params.get('eta_min', 0.00001) if scheduler_params else 0.00001
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_name.lower() == 'step':
        step_size = scheduler_params.get('step_size', 30) if scheduler_params else 30
        gamma = scheduler_params.get('gamma', 0.1) if scheduler_params else 0.1
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name.lower() == 'plateau':
        patience = scheduler_params.get('patience', 10) if scheduler_params else 10
        factor = scheduler_params.get('factor', 0.1) if scheduler_params else 0.1
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    gradient_clipping: Optional[float] = None,
    logging_frequency: int = 10,
    logger: Optional[object] = None
) -> tuple:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        scaler: Optional gradient scaler for mixed precision
        gradient_clipping: Optional gradient clipping value
        logging_frequency: Log every N batches
        logger: Logger instance
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        accuracy = (predicted == labels).float().mean().item()
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        accuracy_meter.update(accuracy, images.size(0))
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{accuracy_meter.avg:.4f}'
        })
        
        # Log batch-wise
        if logger and (batch_idx + 1) % logging_frequency == 0:
            logger.info(
                f"Epoch {epoch} - Batch {batch_idx + 1}/{len(train_loader)} - "
                f"Loss: {loss_meter.avg:.4f}, Accuracy: {accuracy_meter.avg:.4f}"
            )
    
    return loss_meter.avg, accuracy_meter.avg


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    logger: Optional[object] = None
) -> tuple:
    """
    Validate the model for one epoch.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger instance
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean().item()
            
            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            accuracy_meter.update(accuracy, images.size(0))
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{accuracy_meter.avg:.4f}'
            })
    
    if logger:
        logger.info(
            f"Epoch {epoch} - Validation - "
            f"Loss: {loss_meter.avg:.4f}, Accuracy: {accuracy_meter.avg:.4f}"
        )
    
    return loss_meter.avg, accuracy_meter.avg


def train(config_path: str):
    """
    Main training function.
    
    Args:
        config_path: Path to the config.yaml file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup basic console logger first
    logger = setup_logger(
        name="ResNetClassifier",
        log_dir=None,
        log_level=config['logging']['log_level']
    )
    
    # Create timestamped output directories
    output_dirs = create_timestamped_output_dirs(base_output_dir=config['output_base_dir'])
    logger.info(f"Created timestamped output directories in: {Path(config['output_base_dir']) / Path(output_dirs['logs']).parent.name}")
    
    # Setup full logger with file output
    log_dir = output_dirs['logs']
    logger = setup_logger(
        name="ResNetClassifier",
        log_dir=log_dir,
        log_level=config['logging']['log_level']
    )
    
    logger.info("=" * 50)
    logger.info("Starting Training")
    logger.info("=" * 50)
    
    # Log system information
    #logger.log_system_info()
    
    # Set random seed for reproducibility
    set_seed(config['dataset']['random_seed'], deterministic=True)
    logger.info(f"Random seed set to {config['dataset']['random_seed']}")
    
    # Determine device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    
    logger.info(f"Using device: {device}")
    
    # Output directories already created by create_timestamped_output_dirs
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
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
    
    num_classes = len(class_to_idx)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class mapping: {class_to_idx}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        resnet_variant=config['model']['resnet_variant'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    )
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Move model to device
    model = model.to(device)
    
    # Handle multi-GPU
    if device == 'cuda' and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    logger.info(f"Optimizer: {config['training']['optimizer']}")
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training']['scheduler'],
        scheduler_params=config['training'].get('scheduler_params', None)
    )
    logger.info(f"Scheduler: {config['training']['scheduler']}")
    
    # Setup mixed precision training
    scaler = GradScaler() if config['training']['mixed_precision'] and device == 'cuda' else None
    if scaler:
        logger.info("Mixed precision training enabled")
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=output_dirs['checkpoints'],
        save_best_only=config['checkpoint']['save_best_only'],
        checkpoint_frequency=config['checkpoint']['checkpoint_frequency']
    )
    
    # Setup early stopping
    early_stopping = None
    if config['early_stopping']['enabled']:
        early_stopping = EarlyStoppingCallback(
            patience=config['early_stopping']['patience'],
            mode='min' if config['early_stopping']['mode'] == 'min' else 'max',
            verbose=True,
            restore_best_weights=True
        )
        logger.info(f"Early stopping enabled (patience: {config['early_stopping']['patience']})")
    
    # Setup plot generator
    plot_generator = PlotGenerator(output_dir=output_dirs['plots'])
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Training loop
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start_time = time.time()
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'=' * 50}")
        
        # Train
        train_loss, train_accuracy = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            gradient_clipping=config['training']['gradient_clipping'],
            logging_frequency=config['logging']['logging_frequency'],
            logger=logger
        )
        
        # Validate
        if val_loader is not None:
            val_loss, val_accuracy = validate_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                logger=logger
            )
        else:
            val_loss, val_accuracy = train_loss, train_accuracy
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'learning_rate': current_lr
        }
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
        
        # Early stopping check
        if early_stopping is not None:
            monitor_metric = val_loss if config['early_stopping']['monitor_metric'] == 'val_loss' else val_accuracy
            if early_stopping(monitor_metric, model):
                logger.info("Early stopping triggered")
                early_stopping.restore_model(model)
                break
        
        # Generate plots every 5 epochs
        if epoch % 5 == 0:
            plot_generator.plot_loss_curves(
                history['train_loss'],
                history['val_loss']
            )
            plot_generator.plot_accuracy_curves(
                history['train_accuracy'],
                history['val_accuracy']
            )
    
    # Save final plots
    logger.info("Generating final plots...")
    plot_generator.plot_loss_curves(
        history['train_loss'],
        history['val_loss'],
        save_name='final_loss_curves.png'
    )
    plot_generator.plot_accuracy_curves(
        history['train_accuracy'],
        history['val_accuracy'],
        save_name='final_accuracy_curves.png'
    )
    plot_generator.plot_combined_metrics(
        history['train_loss'],
        history['val_loss'],
        history['train_accuracy'],
        history['val_accuracy'],
        save_name='final_combined_metrics.png'
    )
    
    # Save training history to CSV with correct column order
    history_df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
    history_df.to_csv(
        Path(output_dirs['metrics']) / 'training_history.csv',
        index=False
    )
    logger.info("Training history saved to CSV")
    
    # Save training history to JSON
    checkpoint_manager.save_metrics_json(
        metrics={
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_loss': best_val_loss
        },
        filename='final_metrics.json'
    )
    
    logger.info("=" * 50)
    logger.info("Training completed successfully")
    logger.info("=" * 50)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet image classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    train(config_path=args.config)
