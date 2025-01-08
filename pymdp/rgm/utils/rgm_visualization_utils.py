"""
Visualization Utilities
=====================

Utilities for RGM visualization and plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class RGMVisualizationUtils:
    """Visualization utilities for RGM analysis"""
    
    @staticmethod
    def plot_hierarchical_structure(hierarchy: Dict,
                                  output_path: Path,
                                  figsize: Tuple[int, int] = (15, 10)):
        """Plot hierarchical model structure"""
        plt.figure(figsize=figsize)
        
        # Plot levels
        for level_idx, (name, level) in enumerate(hierarchy.items()):
            plt.subplot(2, 2, level_idx + 1)
            RGMVisualizationUtils._plot_level_structure(level, name)
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def plot_receptive_fields(model: Dict,
                            level: int,
                            output_path: Path,
                            n_fields: int = 8):
        """Plot receptive fields at given level"""
        fields = RGMVisualizationUtils._extract_receptive_fields(model, level)
        
        plt.figure(figsize=(15, 5))
        for i in range(min(n_fields, len(fields))):
            plt.subplot(2, 4, i + 1)
            plt.imshow(fields[i], cmap='RdBu_r')
            plt.axis('off')
            plt.title(f'Field {i+1}')
            
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def plot_learning_curves(history: Dict,
                           output_path: Path,
                           figsize: Tuple[int, int] = (12, 8)):
        """Plot learning curves"""
        plt.figure(figsize=figsize)
        
        # Plot ELBO
        plt.subplot(2, 2, 1)
        plt.plot(history['elbo'], label='ELBO')
        plt.title('Evidence Lower Bound')
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['accuracy'], label='Accuracy')
        plt.title('Classification Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        
        # Plot sample efficiency
        plt.subplot(2, 2, 3)
        plt.plot(history['samples_processed'], history['accuracy'])
        plt.title('Sample Efficiency')
        plt.xlabel('Samples Processed')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(predictions: List[int],
                            true_labels: List[int],
                            output_path: Path,
                            figsize: Tuple[int, int] = (10, 8)):
        """Plot confusion matrix"""
        cm = np.zeros((10, 10))
        for pred, true in zip(predictions, true_labels):
            cm[true, pred] += 1
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def plot_example_reconstructions(model: Dict,
                                   images: torch.Tensor,
                                   output_path: Path,
                                   n_examples: int = 8):
        """Plot original and reconstructed images"""
        plt.figure(figsize=(15, 6))
        
        for i in range(min(n_examples, len(images))):
            # Original
            plt.subplot(2, n_examples, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Original')
            
            # Reconstruction
            plt.subplot(2, n_examples, i + n_examples + 1)
            reconstruction = RGMVisualizationUtils._reconstruct_image(model, images[i])
            plt.imshow(reconstruction.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Reconstructed')
            
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def _plot_level_structure(level: Dict, name: str):
        """Plot structure of a single level"""
        dims = level['dimensions']
        plt.bar(range(len(dims)), list(dims.values()))
        plt.xticks(range(len(dims)), list(dims.keys()), rotation=45)
        plt.title(f'Level: {name}')
        plt.ylabel('Dimension Size')
    
    @staticmethod
    def _extract_receptive_fields(model: Dict,
                                level: int) -> List[np.ndarray]:
        """Extract receptive fields from model"""
        # Get likelihood mapping for level
        mapping = model[f'D{level}']
        
        # Reshape into image patches
        patch_size = int(np.sqrt(mapping.shape[0] // 3))  # Assuming RGB
        fields = []
        
        for i in range(mapping.shape[1]):
            field = mapping[:, i].reshape(3, patch_size, patch_size)
            fields.append(field.cpu().numpy())
            
        return fields
    
    @staticmethod
    def _reconstruct_image(model: Dict,
                          image: torch.Tensor) -> torch.Tensor:
        """Reconstruct image through model hierarchy"""
        # Forward pass through hierarchy
        current = image
        for level in range(len(model['hierarchy'])):
            current = model[f'D{level}'].t() @ current.flatten()
            current = current.reshape(-1, 1)
            
        # Backward pass for reconstruction
        for level in reversed(range(len(model['hierarchy']))):
            current = model[f'D{level}'] @ current
            
        return current.reshape_as(image) 