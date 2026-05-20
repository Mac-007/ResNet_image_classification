"""
Output directory manager for timestamp-based folder creation.

This module provides functionality to create timestamp-based output directories
for organizing training/testing runs.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict


def create_timestamped_output_dirs(base_output_dir: str = "outputs") -> Dict[str, str]:
    """
    Create timestamp-based output directories for a training/testing run.
    
    Args:
        base_output_dir: Base directory for outputs (default: "outputs")
    
    Returns:
        Dictionary with keys 'checkpoints', 'logs', 'metrics', 'plots', 'predictions'
        and values as the full paths to the timestamped subdirectories
    """
    # Generate timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped base directory
    timestamp_dir = Path(base_output_dir) / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'checkpoints': timestamp_dir / 'checkpoints',
        'logs': timestamp_dir / 'logs',
        'metrics': timestamp_dir / 'metrics',
        'plots': timestamp_dir / 'plots',
        'predictions': timestamp_dir / 'predictions'
    }
    
    # Create all subdirectories
    for subdir_path in subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)
    
    # Return string paths
    return {key: str(path) for key, path in subdirs.items()}


def get_latest_output_dir(base_output_dir: str = "outputs") -> str:
    """
    Get the most recent timestamped output directory.
    
    Args:
        base_output_dir: Base directory for outputs (default: "outputs")
    
    Returns:
        Path string to the most recent timestamped directory, or None if none exist
    """
    base_path = Path(base_output_dir)
    if not base_path.exists():
        return None
    
    # Get all directories in base path
    timestamp_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not timestamp_dirs:
        return None
    
    # Sort by modification time and return the most recent
    latest_dir = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest_dir)


if __name__ == "__main__":
    # Test the function
    output_dirs = create_timestamped_output_dirs()
    print("Created timestamped output directories:")
    for key, path in output_dirs.items():
        print(f"  {key}: {path}")
