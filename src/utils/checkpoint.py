"""
Checkpoint utility module for the ResNet image classification project.

This module provides functions for saving and loading model checkpoints,
including model state, optimizer state, scheduler state, and training metrics.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import json


class CheckpointManager:
    """
    A class to manage model checkpoints during training.
    
    This class handles saving and loading of model checkpoints, including
    model state, optimizer state, scheduler state, epoch information, and metrics.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = False,
        checkpoint_frequency: int = 5
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only save the best model
            checkpoint_frequency: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.checkpoint_frequency = checkpoint_frequency
        self.best_metric = float('inf') if save_best_only else None
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer state to save
            scheduler: The scheduler state to save (optional)
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
            filename: Optional custom filename for the checkpoint
        
        Returns:
            str: Path to the saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Determine filename
        if filename is None:
            if is_best:
                filename = 'best_model.pth'
            else:
                filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as last model
        if not is_best:
            last_checkpoint_path = self.checkpoint_dir / 'last_model.pth'
            torch.save(checkpoint, last_checkpoint_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: The model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load the checkpoint on
        
        Returns:
            Dict: Dictionary containing epoch and metrics
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model state loaded from {checkpoint_path}")
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Optimizer state loaded")
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Scheduler state loaded")
        
        # Return epoch and metrics
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }
        
        self.logger.info(f"Checkpoint loaded: epoch {info['epoch']}")
        return info
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to the latest checkpoint or None if no checkpoint exists
        """
        # Check for last_model.pth first
        last_checkpoint = self.checkpoint_dir / 'last_model.pth'
        if last_checkpoint.exists():
            return last_checkpoint
        
        # Otherwise, find the most recent checkpoint
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return checkpoints[0]
        
        return None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get the path to the best checkpoint.
        
        Returns:
            Path to the best checkpoint or None if no checkpoint exists
        """
        best_checkpoint = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint.exists():
            return best_checkpoint
        return None
    
    def should_save_checkpoint(
        self,
        epoch: int,
        current_metric: float,
        mode: str = 'min'
    ) -> bool:
        """
        Determine if a checkpoint should be saved based on current metric.
        
        Args:
            epoch: Current epoch number
            current_metric: Current metric value
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        
        Returns:
            bool: Whether to save the checkpoint
        """
        # Save periodically
        if not self.save_best_only and epoch % self.checkpoint_frequency == 0:
            return True
        
        # Save if it's the best model
        if self.save_best_only:
            if mode == 'min':
                is_best = current_metric < self.best_metric
            else:
                is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                return True
        
        return False
    
    def save_metrics_json(self, metrics: Dict[str, float], filename: str = 'metrics.json'):
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Name of the JSON file
        """
        metrics_path = self.checkpoint_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_path}")
    
    def load_metrics_json(self, filename: str = 'metrics.json') -> Dict[str, float]:
        """
        Load metrics from a JSON file.
        
        Args:
            filename: Name of the JSON file
        
        Returns:
            Dictionary of metrics
        """
        metrics_path = self.checkpoint_dir / filename
        if not metrics_path.exists():
            return {}
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        self.logger.info(f"Metrics loaded from {metrics_path}")
        return metrics
    
    def delete_old_checkpoints(self, keep_last_n: int = 5):
        """
        Delete old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Delete old checkpoints
        for checkpoint in checkpoints[keep_last_n:]:
            checkpoint.unlink()
            self.logger.info(f"Deleted old checkpoint: {checkpoint}")


if __name__ == "__main__":
    # Test the checkpoint manager
    import tempfile
    import torch.nn as nn
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        
        # Create a simple model
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Initialize checkpoint manager
        manager = CheckpointManager(str(checkpoint_dir), save_best_only=False, checkpoint_frequency=2)
        
        # Save a checkpoint
        metrics = {'train_loss': 0.5, 'val_loss': 0.6, 'train_accuracy': 0.9, 'val_accuracy': 0.85}
        manager.save_checkpoint(model, optimizer, None, epoch=1, metrics=metrics, is_best=True)
        
        # Load the checkpoint
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        info = manager.load_checkpoint(
            checkpoint_dir / 'best_model.pth',
            new_model,
            new_optimizer,
            device='cpu'
        )
        
        print(f"Loaded checkpoint from epoch {info['epoch']}")
        print(f"Metrics: {info['metrics']}")
        
        # Test metrics JSON
        manager.save_metrics_json(metrics)
        loaded_metrics = manager.load_metrics_json()
        print(f"Loaded metrics: {loaded_metrics}")
