"""
MNIST-Specific RGM Implementation
===============================

This module extends the base RGM implementation with MNIST-specific functionality.
It handles MNIST data preprocessing, digit-specific inference, and specialized
visualization methods.

Key features:
1. MNIST data preprocessing
2. Digit-specific state initialization
3. Specialized visualization methods
4. Performance metrics for digit recognition
"""

import numpy as np
import torch
import torchvision
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .rgm_base import RGMBase

class RGMMNIST(RGMBase):
    """RGM implementation specialized for MNIST digit recognition"""
    
    def __init__(self, config: Dict):
        """
        Initialize MNIST-specific RGM model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger('rgm.mnist')
        
        # MNIST-specific parameters
        self.batch_size = config['data']['mnist']['batch_size']
        self.augmentation = config['data']['mnist']['augmentation']
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._initialize_data_loaders()
        
        # Performance tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'confusion_matrix': None
        }
        
    def _initialize_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Initialize MNIST data loaders"""
        try:
            # Define transforms
            transforms = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
            
            # Add augmentation if enabled
            if self.augmentation['enabled']:
                transforms.extend([
                    torchvision.transforms.RandomRotation(self.augmentation['random_rotation']),
                    torchvision.transforms.RandomAffine(
                        0, translate=(
                            self.augmentation['random_translation'],
                            self.augmentation['random_translation']
                        )
                    )
                ])
                
            transform = torchvision.transforms.Compose(transforms)
            
            # Load datasets
            train_dataset = torchvision.datasets.MNIST(
                'data', train=True, download=True,
                transform=transform
            )
            
            val_dataset = torchvision.datasets.MNIST(
                'data', train=False, download=True,
                transform=transform
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config['data']['mnist']['num_workers']
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config['data']['mnist']['num_workers']
            )
            
            self.logger.info("Initialized MNIST data loaders")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data loaders: {str(e)}")
            raise
            
    def preprocess_batch(self, batch: torch.Tensor) -> np.ndarray:
        """
        Preprocess MNIST batch for RGM processing.
        
        Args:
            batch: MNIST image batch (B, 1, 28, 28)
            
        Returns:
            Preprocessed batch (B, 784)
        """
        try:
            # Flatten images
            batch = batch.view(batch.size(0), -1)
            
            # Convert to numpy
            batch = batch.numpy()
            
            # Scale to [0, 1]
            batch = (batch - batch.min()) / (batch.max() - batch.min() + 1e-8)
            
            return batch
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess batch: {str(e)}")
            raise
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        try:
            epoch_loss = 0.0
            n_batches = 0
            
            for batch, labels in self.train_loader:
                # Preprocess batch
                x = self.preprocess_batch(batch)
                
                # Update states
                self.update_states(x)
                
                # Compute reconstruction loss
                loss = self.compute_loss(x)
                epoch_loss += loss
                n_batches += 1
                
            avg_loss = epoch_loss / n_batches
            self.metrics['train_loss'].append(avg_loss)
            
            return {'loss': avg_loss}
            
        except Exception as e:
            self.logger.error(f"Training epoch failed: {str(e)}")
            raise
            
    def validate(self) -> Dict[str, float]:
        """Perform validation"""
        try:
            val_loss = 0.0
            n_batches = 0
            all_preds = []
            all_labels = []
            
            for batch, labels in self.val_loader:
                # Preprocess batch
                x = self.preprocess_batch(batch)
                
                # Forward pass
                activations = self.forward_pass(x)
                
                # Get predictions
                preds = np.argmax(activations[-1], axis=1)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
                # Compute loss
                loss = self.compute_loss(x)
                val_loss += loss
                n_batches += 1
                
            # Compute metrics
            avg_loss = val_loss / n_batches
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            conf_matrix = confusion_matrix(all_labels, all_preds)
            
            # Update metrics
            self.metrics['val_loss'].append(avg_loss)
            self.metrics['accuracy'].append(accuracy)
            self.metrics['confusion_matrix'] = conf_matrix
            
            return {
                'loss': avg_loss,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise
            
    def compute_loss(self, x: np.ndarray) -> float:
        """
        Compute reconstruction loss.
        
        Args:
            x: Input data
            
        Returns:
            Reconstruction loss
        """
        try:
            # Forward pass
            activations = self.forward_pass(x)
            
            # Backward pass
            reconstructions = self.backward_pass(activations[-1])
            
            # Compute MSE loss
            loss = np.mean((x - reconstructions[0]) ** 2)
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Loss computation failed: {str(e)}")
            raise
            
    def visualize_results(self, save_dir: Path):
        """Generate visualizations of results"""
        try:
            # Create visualization directory
            vis_dir = save_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            # Plot loss curves
            self._plot_loss_curves(vis_dir)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(vis_dir)
            
            # Plot sample reconstructions
            self._plot_reconstructions(vis_dir)
            
            self.logger.info(f"Saved visualizations to {vis_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {str(e)}")
            raise
            
    def _plot_loss_curves(self, save_dir: Path):
        """Plot training and validation loss curves"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['train_loss'], label='Train Loss')
            plt.plot(self.metrics['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.savefig(save_dir / 'loss_curves.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot loss curves: {str(e)}")
            raise
            
    def _plot_confusion_matrix(self, save_dir: Path):
        """Plot confusion matrix"""
        try:
            if self.metrics['confusion_matrix'] is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(self.metrics['confusion_matrix'], cmap='viridis')
                plt.colorbar()
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')
                plt.savefig(save_dir / 'confusion_matrix.png')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Failed to plot confusion matrix: {str(e)}")
            raise
            
    def _plot_reconstructions(self, save_dir: Path):
        """Plot sample reconstructions"""
        try:
            # Get random batch
            batch, _ = next(iter(self.val_loader))
            x = self.preprocess_batch(batch)
            
            # Get reconstructions
            activations = self.forward_pass(x)
            reconstructions = self.backward_pass(activations[-1])
            
            # Plot original vs reconstruction
            n_samples = min(10, len(x))
            fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
            
            for i in range(n_samples):
                # Original
                axes[0, i].imshow(x[i].reshape(28, 28), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_title('Original')
                    
                # Reconstruction
                axes[1, i].imshow(reconstructions[0][i].reshape(28, 28), cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_title('Reconstruction')
                    
            plt.tight_layout()
            plt.savefig(save_dir / 'reconstructions.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot reconstructions: {str(e)}")
            raise 