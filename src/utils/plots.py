"""
Plots utility module for the ResNet image classification project.

This module provides functions for generating and saving training plots,
including loss curves, accuracy curves, learning rate curves, and confusion matrices.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class PlotGenerator:
    """
    A class to generate and save various training plots.
    
    This class provides methods to create loss curves, accuracy curves,
    learning rate curves, and confusion matrices with professional styling.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the plot generator.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epochs: Optional[List[int]] = None,
        save_name: str = 'loss_curves.png'
    ):
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            epochs: Optional list of epoch numbers
            save_name: Name of the file to save
        """
        if epochs is None:
            epochs = list(range(1, len(train_losses) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Loss curves saved to {save_path}")
    
    def plot_accuracy_curves(
        self,
        train_accuracies: List[float],
        val_accuracies: List[float],
        epochs: Optional[List[int]] = None,
        save_name: str = 'accuracy_curves.png'
    ):
        """
        Plot training and validation accuracy curves.
        
        Args:
            train_accuracies: List of training accuracies
            val_accuracies: List of validation accuracies
            epochs: Optional list of epoch numbers
            save_name: Name of the file to save
        """
        if epochs is None:
            epochs = list(range(1, len(train_accuracies) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracies, label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Accuracy curves saved to {save_path}")
    
    def plot_learning_rate_curve(
        self,
        learning_rates: List[float],
        epochs: Optional[List[int]] = None,
        save_name: str = 'learning_rate_curve.png'
    ):
        """
        Plot learning rate curve over epochs.
        
        Args:
            learning_rates: List of learning rates
            epochs: Optional list of epoch numbers
            save_name: Name of the file to save
        """
        if epochs is None:
            epochs = list(range(1, len(learning_rates) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, learning_rates, linewidth=2, marker='o', markersize=4, color='green')
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Learning rate curve saved to {save_path}")
    
    def plot_combined_metrics(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        learning_rates: Optional[List[float]] = None,
        save_name: str = 'combined_metrics.png'
    ):
        """
        Plot combined metrics in a single figure with subplots.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_accuracies: List of training accuracies
            val_accuracies: List of validation accuracies
            learning_rates: Optional list of learning rates
            save_name: Name of the file to save
        """
        epochs = list(range(1, len(train_losses) + 1))
        
        if learning_rates is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(epochs, train_losses, label='Training Loss', linewidth=2, marker='o', markersize=4)
        axes[0].plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, train_accuracies, label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Learning rate curve (if provided)
        if learning_rates is not None:
            axes[2].plot(epochs, learning_rates, linewidth=2, marker='o', markersize=4, color='green')
            axes[2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[2].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
            axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale('log')
            
            # Remove the fourth subplot
            axes[3].remove()
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Combined metrics plot saved to {save_path}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        save_name: str = 'confusion_matrix.png'
    ):
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the confusion matrix
            save_name: Name of the file to save
        """
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict[str, Dict[str, float]],
        save_name: str = 'per_class_metrics.png'
    ):
        """
        Plot per-class metrics (precision, recall, F1-score).
        
        Args:
            per_class_metrics: Dictionary of per-class metrics
            save_name: Name of the file to save
        """
        class_names = list(per_class_metrics.keys())
        precisions = [per_class_metrics[name]['precision'] for name in class_names]
        recalls = [per_class_metrics[name]['recall'] for name in class_names]
        f1_scores = [per_class_metrics[name]['f1-score'] for name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.figure(figsize=(14, 8))
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Per-Class Metrics', fontsize=14, fontweight='bold')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Per-class metrics plot saved to {save_path}")


if __name__ == "__main__":
    # Test the plot generator
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_gen = PlotGenerator(temp_dir)
        
        # Generate dummy data
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.55, 0.5]
        train_accuracies = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82]
        val_accuracies = [0.48, 0.58, 0.68, 0.72, 0.75, 0.78]
        learning_rates = [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005]
        
        # Test individual plots
        plot_gen.plot_loss_curves(train_losses, val_losses)
        plot_gen.plot_accuracy_curves(train_accuracies, val_accuracies)
        plot_gen.plot_learning_rate_curve(learning_rates)
        plot_gen.plot_combined_metrics(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates)
        
        # Test confusion matrix
        cm = np.array([[50, 5, 2], [3, 45, 7], [1, 4, 48]])
        class_names = ['cat', 'dog', 'bird']
        plot_gen.plot_confusion_matrix(cm, class_names, normalize=False)
        plot_gen.plot_confusion_matrix(cm, class_names, normalize=True, save_name='confusion_matrix_normalized.png')
        
        # Test per-class metrics
        per_class_metrics = {
            'cat': {'precision': 0.92, 'recall': 0.88, 'f1-score': 0.90},
            'dog': {'precision': 0.82, 'recall': 0.85, 'f1-score': 0.83},
            'bird': {'precision': 0.85, 'recall': 0.86, 'f1-score': 0.85}
        }
        plot_gen.plot_per_class_metrics(per_class_metrics)
        
        print(f"Plots generated successfully in {temp_dir}")
