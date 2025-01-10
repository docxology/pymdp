"""
RGM Data Manager
==============

Manages data loading and preprocessing for RGM.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_data_utils import RGMDataUtils
from .rgm_core_utils import RGMCoreUtils

class RGMMNISTDataset(Dataset):
    """Custom MNIST dataset with RGM-specific preprocessing"""
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, config: Dict):
        """
        Initialize dataset.
        
        Args:
            data: Image data tensor
            targets: Target labels tensor
            config: Data configuration dictionary
        """
        self.data = data
        self.targets = targets
        self.config = config['data']['mnist']
        self.data_utils = RGMDataUtils(config)
        
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item"""
        # Get image and target
        image = self.data[idx].float()
        target = self.targets[idx]
        
        # Apply preprocessing
        image = self.data_utils.preprocess_image(image, training=True)
        
        return image, target

class RGMDataManager:
    """Manages data loading and preprocessing"""
    
    def __init__(self, config: Dict):
        """
        Initialize data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = RGMExperimentUtils.get_logger('data_manager')
        self.config = config
        self.experiment = RGMExperimentUtils.get_experiment()
        self.data_utils = RGMDataUtils(config)
        
        # Create data directory
        self.data_dir = self.experiment['dirs']['root'] / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and test data loaders.
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        try:
            self.logger.info("Loading MNIST datasets...")
            
            # Load datasets
            train_set = datasets.MNIST(
                root=str(self.data_dir),
                train=True,
                download=True
            )
            test_set = datasets.MNIST(
                root=str(self.data_dir),
                train=False,
                download=True
            )
            
            # Convert to tensors
            train_data = torch.from_numpy(
                train_set.data.numpy()
            ).unsqueeze(1).float() / 255.0
            train_targets = torch.from_numpy(train_set.targets.numpy())
            
            test_data = torch.from_numpy(
                test_set.data.numpy()
            ).unsqueeze(1).float() / 255.0
            test_targets = torch.from_numpy(test_set.targets.numpy())
            
            # Create custom datasets
            train_dataset = RGMMNISTDataset(
                train_data,
                train_targets,
                self.config
            )
            test_dataset = RGMMNISTDataset(
                test_data,
                test_targets,
                self.config
            )
            
            # Create data loaders
            batch_config = self.config['data']['mnist']['batching']
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_config['batch_size'],
                shuffle=batch_config['shuffle'],
                num_workers=batch_config['num_workers'],
                pin_memory=batch_config['pin_memory'],
                drop_last=batch_config['drop_last']
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_config['batch_size'],
                shuffle=False,
                num_workers=batch_config['num_workers'],
                pin_memory=batch_config['pin_memory']
            )
            
            self.logger.info(
                f"Loaded {len(train_dataset)} training samples, "
                f"{len(test_dataset)} test samples"
            )
            
            return train_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise
            
    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Preprocess batch of images.
        
        Args:
            batch: Batch of images
            
        Returns:
            Preprocessed batch
        """
        try:
            # Get preprocessing config
            preproc = self.config['data']['mnist']['preprocessing']
            
            # Center if enabled
            if preproc['center']:
                batch = batch - batch.mean(dim=(2, 3), keepdim=True)
                
            # Standardize if enabled
            if preproc['standardize']:
                batch = batch / (batch.std(dim=(2, 3), keepdim=True) + 1e-8)
                
            return batch
            
        except Exception as e:
            self.logger.error(f"Error preprocessing batch: {str(e)}")
            raise
            
    def save_dataset_info(self):
        """Save dataset information"""
        try:
            info = {
                'train_size': len(self.train_dataset),
                'test_size': len(self.test_dataset),
                'input_shape': list(self.train_dataset[0][0].shape),
                'n_classes': 10,
                'preprocessing': self.config['data']['mnist']['preprocessing'],
                'augmentation': self.config['data']['mnist']['augmentation']
            }
            
            # Save info
            info_path = self.experiment['dirs']['config'] / "dataset_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
                
            self.logger.info(f"Saved dataset info to: {info_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving dataset info: {str(e)}")
            raise 