"""
Dataset module for the ResNet image classification project.

This module provides a custom PyTorch Dataset class for loading images from CSV files,
with support for data augmentation, normalization, and handling of various image formats.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import logging
import os


class ImageClassificationDataset(Dataset):
    """
    A custom PyTorch Dataset for image classification from CSV files.
    
    This dataset loads image paths and labels from a CSV file, applies
    transformations, and returns processed images and labels.
    """
    
    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        split: str = 'train',
        normalization_mean: List[float] = [0.485, 0.456, 0.406],
        normalization_std: List[float] = [0.229, 0.224, 0.225],
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file containing image information
            image_size: Target size for image resizing
            split: Dataset split ('train', 'val', or 'test')
            normalization_mean: Mean values for normalization
            normalization_std: Standard deviation values for normalization
            class_to_idx: Optional dictionary mapping class names to indices
        """
        self.csv_path = Path(csv_path)
        self.image_size = image_size
        self.split = split
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.logger = logging.getLogger(__name__)
        
        # Load CSV file
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data = pd.read_csv(self.csv_path)
        self.logger.info(f"Loaded {len(self.data)} samples from {csv_path}")
        
        # Validate CSV columns
        required_columns = ['image_full_path', 'image_name', 'class_label']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"CSV file must contain column: {col}")
        
        # Build class mapping
        if class_to_idx is None:
            unique_classes = sorted(self.data['class_label'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Validate image paths and filter out missing files
        self._validate_image_paths()
        
        # Setup transformations
        self.transform = self._get_transform()
        
        self.logger.info(f"Dataset initialized with {len(self)} valid samples")
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Class mapping: {self.class_to_idx}")
    
    def _validate_image_paths(self):
        """
        Validate that all image paths exist and filter out missing files.
        Logs skipped samples and updates the dataframe.
        """
        valid_samples = []
        skipped_count = 0
        
        for idx, row in self.data.iterrows():
            image_path = Path(row['image_full_path']) / row['image_name']
            
            if image_path.exists():
                valid_samples.append(row)
            else:
                skipped_count += 1
                self.logger.warning(f"Image not found, skipping: {image_path}")
        
        if skipped_count > 0:
            self.logger.warning(f"Skipped {skipped_count} samples due to missing images")
        
        self.data = pd.DataFrame(valid_samples)
        
        if len(self.data) == 0:
            raise ValueError("No valid images found in the dataset")
    
    def _get_transform(self) -> transforms.Compose:
        """
        Get the appropriate transformation pipeline based on the dataset split.
        
        Returns:
            transforms.Compose: Transformation pipeline
        """
        # Common transformations
        common_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalization_mean,
                std=self.normalization_std
            )
        ]
        
        if self.split == 'train':
            # Training transformations with augmentation
            train_transforms = [
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.normalization_mean,
                    std=self.normalization_std
                )
            ]
            return transforms.Compose(train_transforms)
        else:
            # Validation/test transformations (no augmentation)
            return transforms.Compose(common_transforms)
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get row from dataframe
        row = self.data.iloc[idx]
        
        # Construct full image path
        image_path = Path(row['image_full_path']) / row['image_name']
        
        # Load image
        try:
            image = Image.open(image_path)
            
            # Convert grayscale to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            image = self.transform(image)
            
            # Get label
            class_label = row['class_label']
            label = self.class_to_idx[class_label]
            
            return image, label
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            blank_image = torch.zeros(3, self.image_size, self.image_size)
            return blank_image, 0
    
    def get_class_names(self) -> List[str]:
        """
        Get the list of class names.
        
        Returns:
            List of class names
        """
        return list(self.class_to_idx.keys())
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of samples per class.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = self.data['class_label'].value_counts().to_dict()
        return distribution


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    normalization_mean: List[float] = [0.485, 0.456, 0.406],
    normalization_std: List[float] = [0.229, 0.224, 0.225],
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, int]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        test_csv: Path to test CSV file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        image_size: Target size for image resizing
        normalization_mean: Mean values for normalization
        normalization_std: Standard deviation values for normalization
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx)
    """
    logger = logging.getLogger(__name__)
    
    # Create training dataset
    train_dataset = ImageClassificationDataset(
        csv_path=train_csv,
        image_size=image_size,
        split='train',
        normalization_mean=normalization_mean,
        normalization_std=normalization_std
    )
    
    # Create training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    logger.info(f"Training dataloader created: {len(train_loader)} batches")
    
    # Get class mapping from training set
    class_to_idx = train_dataset.class_to_idx
    
    # Create validation dataset (if CSV exists)
    val_loader = None
    if Path(val_csv).exists():
        val_dataset = ImageClassificationDataset(
            csv_path=val_csv,
            image_size=image_size,
            split='val',
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            class_to_idx=class_to_idx
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        logger.info(f"Validation dataloader created: {len(val_loader)} batches")
    
    # Create test dataset (if CSV exists)
    test_loader = None
    if Path(test_csv).exists():
        test_dataset = ImageClassificationDataset(
            csv_path=test_csv,
            image_size=image_size,
            split='test',
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            class_to_idx=class_to_idx
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        logger.info(f"Test dataloader created: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, class_to_idx


if __name__ == "__main__":
    # Test the dataset
    import tempfile
    import os
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy CSV file
        csv_path = Path(temp_dir) / "test.csv"
        data = {
            'image_full_path': [temp_dir] * 5,
            'image_name': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'class_label': ['cat', 'dog', 'cat', 'bird', 'dog']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        # Create dummy images
        for img_name in data['image_name']:
            img_path = Path(temp_dir) / img_name
            img = Image.new('RGB', (100, 100), color='red')
            img.save(img_path)
        
        # Test dataset
        dataset = ImageClassificationDataset(
            csv_path=str(csv_path),
            image_size=224,
            split='train'
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Class mapping: {dataset.class_to_idx}")
        print(f"Class distribution: {dataset.get_class_distribution()}")
        
        # Test dataloader
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
            train_csv=str(csv_path),
            val_csv=str(csv_path),
            test_csv=str(csv_path),
            batch_size=2,
            num_workers=0
        )
        
        print(f"Train loader batches: {len(train_loader)}")
        
        # Test loading a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")
            break
