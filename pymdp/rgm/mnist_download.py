"""
MNIST Dataset Preprocessor
========================

Downloads and preprocesses MNIST dataset for the Renormalization Generative Model.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict

from rgm.utils import RGMLogging

class MNISTPreprocessor:
    """Handles MNIST dataset download and preprocessing."""
    
    def __init__(self, data_dir: Path, exp_state=None):
        """
        Initialize MNIST preprocessor.
        
        Args:
            data_dir: Directory to store dataset
            exp_state: Optional experiment state manager
        """
        self.data_dir = data_dir
        self.exp_state = exp_state
        self.logger = RGMLogging.get_logger("rgm.mnist")
        
    def prepare_datasets(self) -> Dict:
        """
        Download and preprocess MNIST dataset.
        
        Returns:
            Dictionary containing train and test datasets
        """
        try:
            # Define basic transforms
            transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Simple resize without antialiasing
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            # Download and load training set
            train_dataset = torchvision.datasets.MNIST(
                root=str(self.data_dir),
                train=True,
                download=True,
                transform=transform
            )
            
            # Download and load test set
            test_dataset = torchvision.datasets.MNIST(
                root=str(self.data_dir),
                train=False,
                download=True,
                transform=transform
            )
            
            self.logger.info(f"MNIST datasets prepared: {len(train_dataset)} training, {len(test_dataset)} test samples")
            
            return {
                'train': train_dataset,
                'test': test_dataset
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare MNIST dataset: {str(e)}")
            raise