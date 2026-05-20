"""
Metrics utility module for the ResNet image classification project.

This module provides functions for calculating various classification metrics
including accuracy, precision, recall, F1-score, and confusion matrix generation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from typing import Dict, List, Tuple, Optional, Union
import logging


class MetricsCalculator:
    """
    A class to calculate and track various classification metrics.
    
    This class provides methods to compute accuracy, precision, recall, F1-score,
    and other metrics for multi-class classification tasks.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            num_classes: Number of classes in the classification task
            class_names: Optional list of class names for better interpretability
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.logger = logging.getLogger(__name__)
    
    def calculate_accuracy(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate accuracy score.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
        
        Returns:
            float: Accuracy score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return accuracy_score(targets, predictions)
    
    def calculate_precision(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        average: str = 'macro'
    ) -> float:
        """
        Calculate precision score.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            average: Averaging method ('macro', 'micro', 'weighted', None)
        
        Returns:
            float: Precision score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return precision_score(targets, predictions, average=average, zero_division=0)
    
    def calculate_recall(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        average: str = 'macro'
    ) -> float:
        """
        Calculate recall score.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            average: Averaging method ('macro', 'micro', 'weighted', None)
        
        Returns:
            float: Recall score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return recall_score(targets, predictions, average=average, zero_division=0)
    
    def calculate_f1(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        average: str = 'macro'
    ) -> float:
        """
        Calculate F1-score.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            average: Averaging method ('macro', 'micro', 'weighted', None)
        
        Returns:
            float: F1-score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return f1_score(targets, predictions, average=average, zero_division=0)
    
    def calculate_balanced_accuracy(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calculate balanced accuracy score.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
        
        Returns:
            float: Balanced accuracy score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return balanced_accuracy_score(targets, predictions)
    
    def calculate_roc_auc(
        self,
        probabilities: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        multi_class: str = 'ovr'
    ) -> Optional[float]:
        """
        Calculate ROC-AUC score for multi-class classification.
        
        Args:
            probabilities: Predicted probabilities (shape: [n_samples, n_classes])
            targets: Ground truth class labels
            multi_class: Multi-class handling ('ovr' or 'ovo')
        
        Returns:
            float: ROC-AUC score or None if calculation fails
        """
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        try:
            return roc_auc_score(targets, probabilities, multi_class=multi_class)
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC-AUC: {e}")
            return None
    
    def compute_confusion_matrix(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
        
        Returns:
            np.ndarray: Confusion matrix
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        return confusion_matrix(targets, predictions, labels=range(self.num_classes), normalize=normalize)
    
    def get_classification_report(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        output_dict: bool = True
    ) -> Union[str, Dict]:
        """
        Generate classification report.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            output_dict: Whether to return report as dictionary
        
        Returns:
            Classification report (string or dictionary)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        target_names = self.class_names if self.class_names else None
        return classification_report(
            targets,
            predictions,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def compute_all_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
            probabilities: Optional predicted probabilities for ROC-AUC
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            'accuracy': self.calculate_accuracy(predictions, targets),
            'precision_macro': self.calculate_precision(predictions, targets, average='macro'),
            'precision_micro': self.calculate_precision(predictions, targets, average='micro'),
            'recall_macro': self.calculate_recall(predictions, targets, average='macro'),
            'recall_micro': self.calculate_recall(predictions, targets, average='micro'),
            'f1_macro': self.calculate_f1(predictions, targets, average='macro'),
            'f1_micro': self.calculate_f1(predictions, targets, average='micro'),
            'balanced_accuracy': self.calculate_balanced_accuracy(predictions, targets)
        }
        
        # Calculate ROC-AUC if probabilities are provided
        if probabilities is not None:
            roc_auc = self.calculate_roc_auc(probabilities, targets)
            if roc_auc is not None:
                metrics['roc_auc'] = roc_auc
        
        return metrics
    
    def get_per_class_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
        
        Returns:
            Dictionary with per-class metrics
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        report = self.get_classification_report(predictions, targets, output_dict=True)
        
        per_class_metrics = {}
        for class_name in self.class_names:
            if class_name in report:
                per_class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        return per_class_metrics


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training epochs.
    """
    
    def __init__(self):
        """Initialize the AverageMeter."""
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.
        
        Args:
            val: New value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


if __name__ == "__main__":
    # Test the metrics calculator
    num_classes = 3
    class_names = ['cat', 'dog', 'bird']
    
    # Create dummy predictions and targets
    predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    targets = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 1])
    probabilities = np.random.rand(10, 3)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    calculator = MetricsCalculator(num_classes, class_names)
    
    # Calculate all metrics
    metrics = calculator.compute_all_metrics(predictions, targets, probabilities)
    print("All Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Get per-class metrics
    per_class = calculator.get_per_class_metrics(predictions, targets)
    print("\nPer-class Metrics:")
    for class_name, class_metrics in per_class.items():
        print(f"{class_name}: {class_metrics}")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"\nAverageMeter average: {meter.avg}")
