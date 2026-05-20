"""
Seed utility module for reproducibility in the ResNet image classification project.

This module provides functions to set random seeds for various libraries to ensure
reproducible results across different runs.
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across multiple libraries.
    
    This function sets seeds for Python's random module, NumPy, PyTorch CPU and GPU,
    and configures CuDNN for deterministic behavior when possible.
    
    Args:
        seed: Random seed value (default: 42)
        deterministic: Whether to enable deterministic behavior in CuDNN (default: True)
                      Note: This may impact performance and is not guaranteed across all operations
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure CuDNN for deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable benchmark mode for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_worker_init_fn(seed: int):
    """
    Create a worker initialization function for DataLoader workers.
    
    This ensures that each DataLoader worker has a different but deterministic
    random seed, which is important for reproducible data loading.
    
    Args:
        seed: Base random seed value
    
    Returns:
        Function to initialize workers with deterministic seeds
    """
    def worker_init_fn(worker_id: int):
        """
        Initialize worker with a deterministic seed based on worker_id.
        
        Args:
            worker_id: ID of the worker process
        """
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def verify_reproducibility():
    """
    Verify that reproducibility settings are correctly configured.
    
    This function checks the current state of random seed settings and logs
    warnings if deterministic behavior might not be guaranteed.
    """
    print("=" * 50)
    print("Reproducibility Settings Verification")
    print("=" * 50)
    print(f"CuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    
    if not torch.backends.cudnn.deterministic:
        print("WARNING: CuDNN deterministic mode is disabled. Results may not be fully reproducible.")
    
    if torch.backends.cudnn.benchmark:
        print("WARNING: CuDNN benchmark mode is enabled. This may improve performance but reduces reproducibility.")
    
    print("=" * 50)


if __name__ == "__main__":
    # Test the seed functions
    set_seed(42, deterministic=True)
    verify_reproducibility()
    
    # Test worker init function
    worker_init = get_worker_init_fn(42)
    print("Worker initialization function created successfully")
