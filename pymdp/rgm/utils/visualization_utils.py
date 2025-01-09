"""
RGM Visualization Utilities
=========================

Utilities for visualizing RGM model inputs, outputs and states.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union

class RGMVisualizationUtils:
    """Utilities for RGM visualizations."""
    
    @staticmethod
    def plot_matrix(
        matrix: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        cmap: str = 'RdBu'
    ):
        """Plot single matrix with colorbar."""
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap=cmap)
        plt.colorbar()
        if title:
            plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_matrix_grid(
        matrices: List[np.ndarray],
        titles: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        n_cols: int = 3
    ):
        """Plot grid of matrices."""
        n_matrices = len(matrices)
        n_rows = (n_matrices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4*n_cols, 3*n_rows)
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
            
        for idx, matrix in enumerate(matrices):
            row = idx // n_cols
            col = idx % n_cols
            im = axes[row, col].imshow(matrix, cmap='RdBu')
            plt.colorbar(im, ax=axes[row, col])
            
            if titles and idx < len(titles):
                axes[row, col].set_title(titles[idx])
                
        # Hide empty subplots
        for idx in range(n_matrices, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_mnist_grid(
        images: torch.Tensor,
        reconstructions: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None,
        n_samples: int = 10,
        title: Optional[str] = None
    ):
        """Plot grid of MNIST images and reconstructions."""
        # Reshape to 28x28
        images = images.reshape(-1, 28, 28).cpu().numpy()
        if reconstructions is not None:
            reconstructions = reconstructions.reshape(-1, 28, 28).cpu().numpy()
            
        n_rows = 2 if reconstructions is not None else 1
        n_samples = min(n_samples, len(images))
        
        fig, axes = plt.subplots(
            n_rows, n_samples,
            figsize=(2*n_samples, 2*n_rows)
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
            
        for i in range(n_samples):
            # Original
            axes[0,i].imshow(images[i], cmap='gray')
            axes[0,i].axis('off')
            if i == 0:
                axes[0,i].set_title('Input')
                
            # Reconstruction
            if reconstructions is not None:
                axes[1,i].imshow(reconstructions[i], cmap='gray')
                axes[1,i].axis('off')
                if i == 0:
                    axes[1,i].set_title('Reconstruction')
                    
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 