"""RGM Data Loader

Handles data loading and preprocessing for the RGM MNIST model.
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .rgm_logging import RGMLogging

class RGMDataLoader:
    """
    Data loader for RGM MNIST model.
    
    Features:
    - Configurable data augmentation and normalization
    - Automatic train/val/test splitting
    - Dimension validation
    - Batch size handling
    
    Args:
        config: Data configuration dictionary
        data_dir: Directory to store/load datasets
    """
    
    def __init__(self,
                 config: dict,
                 data_dir: Optional[str] = None):
        """Initialize the data loader."""
        self.logger = RGMLogging.get_logger("rgm.data")
        self._validate_config(config)
        self.config = config
        
        # Set data directory
        self.data_dir = Path(data_dir) if data_dir else Path.home() / '.torch'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract parameters
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"].get("num_workers", 4)
        self.input_dim = config["architecture"]["input_dim"]
        
        # Set up transforms
        self.transform = self._get_transforms()
        
        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Prepare data
        self._prepare_data()
        
        self.logger.info("✓ Data loader initialized successfully")
        
    def _prepare_data(self) -> None:
        """Prepare data loaders internally."""
        self.train_loader, self.val_loader, self.test_loader = self.prepare_data()
    
    def get_train_loader(self) -> DataLoader:
        """Get the training data loader.
        
        Returns:
            DataLoader for training data
        """
        if self.train_loader is None:
            self._prepare_data()
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get the validation data loader.
        
        Returns:
            DataLoader for validation data
        """
        if self.val_loader is None:
            self._prepare_data()
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """Get the test data loader.
        
        Returns:
            DataLoader for test data
        """
        if self.test_loader is None:
            self._prepare_data()
        return self.test_loader
    
    def _validate_config(self, config: dict) -> None:
        """Validate data configuration."""
        required_keys = {
            "data": ["batch_size", "train_split", "val_split"],
            "architecture": ["input_dim"]
        }
        
        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(
                        f"Missing config key: {section}.{key}"
                    )
                    
        # Validate splits
        splits = [
            config["data"]["train_split"],
            config["data"]["val_split"]
        ]
        if not all(0 <= s <= 1 for s in splits) or sum(splits) > 1:
            raise ValueError(
                "Invalid split values. Must be between 0 and 1 "
                "and sum to less than or equal to 1."
            )
            
        # Validate batch size
        if config["data"]["batch_size"] < 1:
            raise ValueError(
                f"Batch size must be positive, "
                f"got {config['data']['batch_size']}"
            )
            
    def _get_transforms(self) -> transforms.Compose:
        """Create transform pipeline."""
        transform_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
        ]
        
        # Add normalization if specified
        if "normalization" in self.config["data"]:
            norm_config = self.config["data"]["normalization"]
            transform_list.append(
                transforms.Normalize(
                    norm_config["mean"],
                    norm_config["std"]
                )
            )
            
        # Add data augmentation if specified
        if self.config["data"].get("augmentation", False):
            aug_config = self.config["data"]["augmentation"]
            if aug_config.get("random_rotation", False):
                transform_list.insert(
                    0,
                    transforms.RandomRotation(
                        aug_config["rotation_degrees"]
                    )
                )
            if aug_config.get("random_crop", False):
                transform_list.insert(
                    0,
                    transforms.RandomCrop(
                        28,
                        padding=aug_config["crop_padding"]
                    )
                )
                
        return transforms.Compose(transform_list)
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Download and load MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        # Split training data into train and validation
        train_size = int(self.config["data"]["train_split"] * len(train_dataset))
        val_size = int(self.config["data"]["val_split"] * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        
        train_dataset, val_dataset, extra = random_split(
            train_dataset,
            [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Log dataset sizes
        self.logger.info(
            f"Dataset sizes: "
            f"train={len(train_dataset)}, "
            f"val={len(val_dataset)}, "
            f"test={len(test_dataset)}"
        )
        
        return train_loader, val_loader, test_loader
        
    def validate_dimensions(self, batch: torch.Tensor) -> None:
        """
        Validate batch dimensions.
        
        Args:
            batch: Input batch tensor
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if batch.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor [batch_size, {self.input_dim}], "
                f"got shape {batch.shape}"
            )
            
        if batch.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {batch.shape[1]}"
            )
            
    def get_sample_batch(self) -> torch.Tensor:
        """
        Get a sample batch for testing.
        
        Returns:
            Sample batch tensor
        """
        train_loader, _, _ = self.prepare_data()
        sample_batch = next(iter(train_loader))[0]
        self.validate_dimensions(sample_batch)
        return sample_batch 