# ResNet Image Classification Project

A production-grade PyTorch image classification project using ResNet architecture with comprehensive training, validation, and testing capabilities.

> ### Author  
> **Dr. Amit Chougule (PhD)**  

## Project Overview

This project provides a complete, modular, and production-ready implementation for training ResNet models for image classification tasks. It features:

- **Config-driven system**: All parameters controlled via YAML configuration
- **Multiple ResNet variants**: Support for ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152
- **Advanced training features**: Mixed precision training, gradient clipping, learning rate scheduling
- **Comprehensive metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices
- **Early stopping**: Automatic stopping to prevent overfitting
- **Checkpoint management**: Save and load model checkpoints with full state
- **Professional logging**: Console and file logging with timestamps
- **High-quality plots**: Training curves, confusion matrices, per-class metrics
- **Reproducibility**: Deterministic training with seed control

## Project Structure

```
project_root/
│
├── config/
│   └── config.yaml              # Configuration file
│
├── data/
│   ├── train.csv                # Training data CSV
│   ├── val.csv                  # Validation data CSV
│   └── test.csv                 # Test data CSV
│
├── src/
│   ├── datasets/
│   │   └── dataset.py           # Custom dataset class
│   │
│   ├── models/
│   │   └── model.py             # ResNet model definition
│   │
│   ├── utils/
│   │   ├── logger.py            # Logging utilities
│   │   ├── metrics.py           # Metrics calculation
│   │   ├── plots.py             # Plot generation
│   │   ├── seed.py              # Reproducibility utilities
│   │   ├── checkpoint.py        # Checkpoint management
│   │   └── early_stopping.py    # Early stopping implementation
│   │
│   ├── train.py                 # Training script
│   ├── validate.py              # Validation script
│   └── test.py                  # Testing script
│
├── outputs/
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # Training logs
│   ├── metrics/                 # Metrics and plots
│   ├── plots/                   # Training plots
│   └── predictions/             # Test predictions
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Using UV (Recommended)

```bash
# Install UV if not already installed
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

## Configuration

All parameters are controlled through `config/config.yaml`. Key configuration sections:

### Paths
- `train_csv`: Path to training CSV file
- `val_csv`: Path to validation CSV file
- `test_csv`: Path to test CSV file

### Dataset
- `batch_size`: Batch size for training (default: 32)
- `num_workers`: Number of data loading workers (default: 4)
- `image_size`: Target image size (default: 224)
- `random_seed`: Random seed for reproducibility (default: 42)

### Model
- `resnet_variant`: ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
- `num_classes`: Number of output classes
- `pretrained`: Use pretrained weights (true/false)
- `class_names`: List of class names

### Training
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `optimizer`: Optimizer (adam, sgd, adamw)
- `scheduler`: Learning rate scheduler (cosine, step, plateau, none)
- `mixed_precision`: Enable mixed precision training
- `gradient_clipping`: Gradient clipping value

### Early Stopping
- `enabled`: Enable early stopping (true/false)
- `patience`: Patience in epochs
- `monitor_metric`: Metric to monitor (val_loss, val_accuracy)
- `mode`: min or max

## CSV Format

The CSV files must contain the following columns:

```csv
image_full_path,image_name,class_label
/path/to/images,image_001.jpg,class_0
/path/to/images,image_002.jpg,class_1
```

- `image_full_path`: Directory path containing the images
- `image_name`: Name of the image file
- `class_label`: Class label for the image

**Important**: The dataset loader combines `image_full_path + image_name` to create the full absolute path to each image.

## Usage

### Training

To train the model:

```bash
python src/train.py --config config/config.yaml
```

This will:
- Load data from CSV files
- Initialize the ResNet model
- Train for the specified number of epochs
- Save checkpoints periodically
- Generate training plots
- Apply early stopping if enabled

### Validation

To validate the model on the validation set:

```bash
python src/validate.py --config config/config.yaml
```

Optional: Specify a custom checkpoint:

```bash
python src/validate.py --config config/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

This will:
- Load the trained model
- Evaluate on validation data
- Calculate comprehensive metrics
- Generate confusion matrices
- Save results to JSON and CSV

### Testing

To test the model on the test set:

```bash
python src/test.py --config config/config.yaml
```

Optional: Specify a custom checkpoint:

```bash
python src/test.py --config config/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

This will:
- Load the trained model
- Evaluate on test data
- Calculate comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generate confusion matrices
- Save predictions to CSV
- Generate classification report

## Output Files

### Checkpoints
- `best_model.pth`: Best model based on validation metric
- `last_model.pth`: Model from the last epoch
- `checkpoint_epoch_N.pth`: Periodic checkpoints

### Logs
- `training.log`: Complete training log with timestamps

### Metrics
- `training_history.csv`: Training loss, accuracy, learning rate per epoch
- `validation_metrics.json`: Validation metrics
- `test_metrics.json`: Test metrics
- `test_predictions.csv`: Predictions for test set

### Plots
- `loss_curves.png`: Training and validation loss curves
- `accuracy_curves.png`: Training and validation accuracy curves
- `learning_rate_curve.png`: Learning rate schedule
- `combined_metrics.png`: Combined metrics visualization
- `confusion_matrix.png`: Confusion matrix
- `confusion_matrix_normalized.png`: Normalized confusion matrix
- `per_class_metrics.png`: Per-class precision, recall, F1-score

## Features

### Data Augmentation
Training data includes:
- Random resized crop
- Random horizontal flip
- Random rotation
- Color jitter

### Supported Image Formats
- RGB images
- Grayscale images (automatically converted to RGB)

### Model Variants
- ResNet18: Lightweight, fast training
- ResNet34: Balanced performance
- ResNet50: Standard choice, good accuracy
- ResNet101: High accuracy, slower training
- ResNet152: Maximum accuracy, slowest training

### Optimizers
- Adam: Adaptive learning rate
- SGD: Momentum-based
- AdamW: Adam with weight decay correction

### Learning Rate Schedulers
- Cosine Annealing: Smooth decay
- Step: Step-wise decay
- Plateau: Reduce on plateau
- None: Constant learning rate

## Advanced Features

### Mixed Precision Training
Enables faster training with reduced memory usage using automatic mixed precision (AMP). Requires CUDA.

### Gradient Clipping
Prevents exploding gradients by clipping gradient norms.

### Multi-GPU Support
Automatically uses multiple GPUs if available using DataParallel.

### Reproducibility
Deterministic training with:
- Fixed random seeds
- CuDNN deterministic mode
- Worker initialization for data loading

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size` in config
- Enable mixed precision training
- Use a smaller ResNet variant

### Slow Training
- Increase `num_workers` for data loading
- Enable mixed precision training
- Use a smaller ResNet variant
- Reduce image size

### Poor Accuracy
- Increase training epochs
- Adjust learning rate
- Try different optimizer
- Enable data augmentation
- Use pretrained weights

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependency list.
