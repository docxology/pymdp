"""
RGM Data Utilities
================

Handles data processing and augmentation for RGM.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_core_utils import RGMCoreUtils

class RGMDataUtils:
    """Handles data processing and augmentation"""
    
    def __init__(self, config: Dict):
        """
        Initialize data utilities.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = RGMExperimentUtils.get_logger('data_utils')
        self.config = config['data']
        self.core = RGMCoreUtils()
        
        # Initialize transforms
        self.transforms = self._initialize_transforms()
        
    def _initialize_transforms(self) -> Dict[str, T.Compose]:
        """Initialize data transforms"""
        try:
            transforms = {}
            
            # Training transforms with augmentation
            train_transforms = []
            if self.config['augmentation']['enabled']:
                train_transforms.extend([
                    T.RandomRotation(
                        degrees=self.config['augmentation']['rotation_range'],
                        fill=0
                    ),
                    T.RandomAffine(
                        degrees=0,
                        translate=(
                            self.config['augmentation']['width_shift'],
                            self.config['augmentation']['height_shift']
                        ),
                        scale=(
                            1 - self.config['augmentation']['zoom_range'],
                            1 + self.config['augmentation']['zoom_range']
                        ),
                        fill=0
                    )
                ])
                
            # Add base transforms
            train_transforms.extend(self._get_base_transforms())
            transforms['train'] = T.Compose(train_transforms)
            
            # Test transforms (no augmentation)
            transforms['test'] = T.Compose(self._get_base_transforms())
            
            return transforms
            
        except Exception as e:
            self.logger.error(f"Error initializing transforms: {str(e)}")
            raise
            
    def _get_base_transforms(self) -> List[torch.nn.Module]:
        """Get base transforms for all data"""
        try:
            transforms = []
            
            # Resize
            transforms.append(
                T.Resize(
                    self.config['input']['size'],
                    antialias=True
                )
            )
            
            # Add preprocessing
            if self.config['preprocessing']['normalize']:
                transforms.append(T.Normalize((0.5,), (0.5,)))
                
            if self.config['preprocessing']['center']:
                transforms.append(lambda x: x - x.mean())
                
            if self.config['preprocessing']['standardize']:
                transforms.append(
                    lambda x: x / (x.std() + 1e-8)
                )
                
            return transforms
            
        except Exception as e:
            self.logger.error(f"Error getting base transforms: {str(e)}")
            raise
            
    def process_batch(self, batch: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Process batch of images.
        
        Args:
            batch: Batch of images
            training: Whether in training mode
            
        Returns:
            Processed batch
        """
        try:
            # Apply transforms
            transform = self.transforms['train'] if training else self.transforms['test']
            processed = transform(batch)
            
            # Ensure proper shape and type
            if len(processed.shape) == 3:
                processed = processed.unsqueeze(1)  # Add channel dimension
                
            return processed.float()
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            raise
            
    def preprocess_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image],
                        training: bool = False) -> torch.Tensor:
        """
        Preprocess single image.
        
        Args:
            image: Input image
            training: Whether in training mode
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert to tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            elif isinstance(image, Image.Image):
                image = TF.to_tensor(image)
                
            # Add batch dimension if needed
            if len(image.shape) == 2:
                image = image.unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == 3:
                image = image.unsqueeze(0)
                
            # Process batch
            return self.process_batch(image, training)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise
            
    def create_batches(self, data: torch.Tensor, batch_size: int,
                      shuffle: bool = True) -> List[torch.Tensor]:
        """
        Create batches from data tensor.
        
        Args:
            data: Input data tensor
            batch_size: Size of batches
            shuffle: Whether to shuffle data
            
        Returns:
            List of batch tensors
        """
        try:
            # Get indices
            indices = torch.randperm(len(data)) if shuffle else torch.arange(len(data))
            
            # Create batches
            batches = []
            for i in range(0, len(data), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = data[batch_indices]
                batches.append(batch)
                
            return batches
            
        except Exception as e:
            self.logger.error(f"Error creating batches: {str(e)}")
            raise
            
    def validate_data(self, data: torch.Tensor) -> Tuple[bool, List[str]]:
        """
        Validate data tensor.
        
        Args:
            data: Data tensor to validate
            
        Returns:
            Tuple of (is_valid, messages)
        """
        try:
            messages = []
            
            # Check shape
            if len(data.shape) not in [3, 4]:  # [B, H, W] or [B, C, H, W]
                messages.append(f"Invalid data shape: {data.shape}")
                
            # Check values
            if torch.isnan(data).any():
                messages.append("Data contains NaN values")
                
            if torch.isinf(data).any():
                messages.append("Data contains Inf values")
                
            # Check range if normalized
            if self.config['preprocessing']['normalize']:
                if data.min() < -1.1 or data.max() > 1.1:
                    messages.append(f"Data range invalid: [{data.min():.2f}, {data.max():.2f}]")
                    
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            raise
            
    def save_batch_info(self, batch: torch.Tensor, save_dir: Path):
        """
        Save batch information.
        
        Args:
            batch: Batch tensor
            save_dir: Directory to save info
        """
        try:
            info = {
                'shape': list(batch.shape),
                'dtype': str(batch.dtype),
                'device': str(batch.device),
                'statistics': {
                    'min': float(batch.min()),
                    'max': float(batch.max()),
                    'mean': float(batch.mean()),
                    'std': float(batch.std())
                },
                'memory': {
                    'size_mb': float(batch.element_size() * batch.nelement() / 1024 / 1024)
                }
            }
            
            # Save info
            info_path = save_dir / "batch_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
                
            self.logger.info(f"Saved batch info to: {info_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch info: {str(e)}")
            raise