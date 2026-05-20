"""
Early stopping utility module for the ResNet image classification project.

This module provides an EarlyStopping class to monitor a metric during training
and stop training when the metric stops improving, preventing overfitting.
"""

import torch
import numpy as np
from typing import Optional
import logging


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric has stopped improving.
    
    This class monitors a specified metric and stops training if the metric
    doesn't improve for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize the EarlyStopping class.
        
        Args:
            patience: Number of epochs to wait before stopping if no improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics to minimize (e.g., loss), 'max' for metrics to maximize (e.g., accuracy)
            verbose: Whether to print messages about early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize counters and best score
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        
        # Determine comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop based on the current score.
        
        Args:
            current_score: Current value of the monitored metric
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch - save the score
            self.best_score = current_score
            return False
        
        # Check if current score is better than best score
        if self.is_better(current_score, self.best_score):
            # Improvement detected
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                self.logger.info(
                    f"Metric improved: {self.best_score:.6f}. "
                    f"Resetting counter."
                )
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"No improvement in metric. "
                    f"Counter: {self.counter}/{self.patience}"
                )
            
            # Check if patience is exceeded
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.warning(
                        f"Early stopping triggered. "
                        f"No improvement for {self.patience} epochs."
                    )
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def get_best_score(self) -> Optional[float]:
        """
        Get the best score observed so far.
        
        Returns:
            Best score or None if no score has been recorded
        """
        return self.best_score
    
    def get_counter(self) -> int:
        """
        Get the current counter value.
        
        Returns:
            Current counter value
        """
        return self.counter


class EarlyStoppingCallback:
    """
    A callback class for integrating early stopping with training loops.
    
    This class provides additional functionality like saving the best model
    and restoring the best model state when early stopping is triggered.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
        restore_best_weights: bool = True
    ):
        """
        Initialize the EarlyStoppingCallback.
        
        Args:
            patience: Number of epochs to wait before stopping if no improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
            verbose: Whether to print messages about early stopping
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.early_stopping = EarlyStopping(patience, min_delta, mode, verbose)
        self.restore_best_weights = restore_best_weights
        self.best_model_state: Optional[dict] = None
        self.logger = logging.getLogger(__name__)
    
    def __call__(
        self,
        current_score: float,
        model: torch.nn.Module
    ) -> bool:
        """
        Check if training should stop and optionally save best model state.
        
        Args:
            current_score: Current value of the monitored metric
            model: The model being trained
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        should_stop = self.early_stopping(current_score)
        
        # Save best model state if improvement detected
        if self.restore_best_weights and not should_stop:
            if self.early_stopping.best_score == current_score:
                self.best_model_state = model.state_dict().copy()
                self.logger.info("Best model state saved.")
        
        return should_stop
    
    def restore_model(self, model: torch.nn.Module):
        """
        Restore the model to the best state if early stopping was triggered.
        
        Args:
            model: The model to restore
        """
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            self.logger.info("Model restored to best state.")
    
    def reset(self):
        """Reset the early stopping callback state."""
        self.early_stopping.reset()
        self.best_model_state = None


if __name__ == "__main__":
    # Test the early stopping
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with minimizing mode (e.g., for loss)
    print("Testing EarlyStopping with mode='min'")
    early_stop = EarlyStopping(patience=3, mode='min', verbose=True)
    
    # Simulate training with improving then plateauing loss
    losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5]
    for epoch, loss in enumerate(losses):
        print(f"Epoch {epoch + 1}: Loss = {loss}")
        if early_stop(loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"\nBest score: {early_stop.get_best_score()}")
    
    # Test with maximizing mode (e.g., for accuracy)
    print("\n" + "=" * 50)
    print("Testing EarlyStopping with mode='max'")
    early_stop = EarlyStopping(patience=3, mode='max', verbose=True)
    
    # Simulate training with improving then plateauing accuracy
    accuracies = [0.7, 0.75, 0.8, 0.85, 0.85, 0.85, 0.85]
    for epoch, acc in enumerate(accuracies):
        print(f"Epoch {epoch + 1}: Accuracy = {acc}")
        if early_stop(acc):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"\nBest score: {early_stop.get_best_score()}")
