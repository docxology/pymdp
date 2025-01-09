"""
RGM Data Loader
============

Data loading utilities for the Renormalization Generative Model.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np

from .rgm_logging import RGMLogging

class RGMDataLoader:
    """Handles data loading and preprocessing for RGM."""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = RGMLogging.get_logger("rgm.data_loader")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data_loaders(self, mnist_data: Dict) -> Dict[str, DataLoader]:
        """
        Prepare data loaders for training and validation.
        
        Args:
            mnist_data: Dictionary containing MNIST data
            
        Returns:
            Dictionary of data loaders
        """
        try:
            # Convert data to tensors
            train_images = torch.FloatTensor(mnist_data['train'].data.numpy())
            train_labels = torch.LongTensor(mnist_data['train'].targets.numpy())
            test_images = torch.FloatTensor(mnist_data['test'].data.numpy())
            test_labels = torch.LongTensor(mnist_data['test'].targets.numpy())
            
            # Normalize images to [0, 1]
            train_images = train_images / 255.0
            test_images = test_images / 255.0
            
            # Reshape images to (batch_size, channels, height, width)
            train_images = train_images.unsqueeze(1)
            test_images = test_images.unsqueeze(1)
            
            # Create datasets
            train_dataset = TensorDataset(train_images, train_labels)
            test_dataset = TensorDataset(test_images, test_labels)
            
            # Split training data into train and validation
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            self.logger.info(f"✓ Prepared data loaders:")
            self.logger.info(f"   - Training samples: {len(train_dataset):,}")
            self.logger.info(f"   - Validation samples: {len(val_dataset):,}")
            self.logger.info(f"   - Test samples: {len(test_dataset):,}")
            
            return {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare data loaders: {str(e)}")
            raise 