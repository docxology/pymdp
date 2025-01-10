"""
Matrix Visualization Utilities

This module provides utilities for visualizing and analyzing matrices
used in the RGM model, including heatmaps, histograms, and statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

class RGMMatrixVisualizationUtils:
    """Utilities for visualizing RGM matrices."""
    
    def __init__(self):
        """Initialize visualization utilities."""
        pass
        
    def plot_matrix_analysis(self,
                           matrix: np.ndarray,
                           title: str,
                           save_path: Path,
                           figsize: Optional[Tuple[int, int]] = None) -> None:
        """Create comprehensive matrix analysis plot.
        
        Args:
            matrix: 2D numpy array to analyze
            title: Plot title
            save_path: Path to save visualization
            figsize: Optional figure size (width, height)
        """
        # Set up figure
        fig = plt.figure(figsize=figsize or (15, 5))
        fig.suptitle(title, fontsize=14)
        
        # Plot heatmap
        ax1 = plt.subplot(131)
        im = ax1.imshow(matrix, cmap='viridis')
        plt.colorbar(im, ax=ax1)
        ax1.set_title('Matrix Heatmap')
        
        # Plot histogram
        ax2 = plt.subplot(132)
        ax2.hist(matrix.flatten(), bins=50, density=True)
        ax2.set_title('Value Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        
        # Plot statistics
        ax3 = plt.subplot(133)
        ax3.axis('off')
        stats = [
            f"Shape: {matrix.shape}",
            f"Mean: {matrix.mean():.4f}",
            f"Std: {matrix.std():.4f}",
            f"Min: {matrix.min():.4f}",
            f"Max: {matrix.max():.4f}",
            f"Sparsity: {(matrix == 0).mean():.2%}"
        ]
        ax3.text(0.1, 0.5, '\n'.join(stats),
                transform=ax3.transAxes,
                fontfamily='monospace')
        ax3.set_title('Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def plot_matrix_comparison(self,
                             matrix1: np.ndarray,
                             matrix2: np.ndarray,
                             title1: str,
                             title2: str,
                             save_path: Path,
                             figsize: Optional[Tuple[int, int]] = None) -> None:
        """Create comparison plot between two matrices.
        
        Args:
            matrix1: First matrix to compare
            matrix2: Second matrix to compare
            title1: Title for first matrix
            title2: Title for second matrix
            save_path: Path to save visualization
            figsize: Optional figure size (width, height)
        """
        # Validate shapes
        if matrix1.shape != matrix2.shape:
            raise ValueError(
                f"Matrix shapes don't match: {matrix1.shape} vs {matrix2.shape}"
            )
            
        # Set up figure
        fig = plt.figure(figsize=figsize or (15, 10))
        
        # Plot first matrix
        ax1 = plt.subplot(231)
        im1 = ax1.imshow(matrix1, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(f'{title1}\nHeatmap')
        
        # Plot second matrix
        ax2 = plt.subplot(232)
        im2 = ax2.imshow(matrix2, cmap='viridis')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(f'{title2}\nHeatmap')
        
        # Plot difference
        ax3 = plt.subplot(233)
        diff = matrix2 - matrix1
        im3 = ax3.imshow(diff, cmap='RdBu')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('Difference\nHeatmap')
        
        # Plot histograms
        ax4 = plt.subplot(234)
        ax4.hist(matrix1.flatten(), bins=50, alpha=0.5,
                label=title1, density=True)
        ax4.hist(matrix2.flatten(), bins=50, alpha=0.5,
                label=title2, density=True)
        ax4.set_title('Value Distributions')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        
        # Plot difference histogram
        ax5 = plt.subplot(235)
        ax5.hist(diff.flatten(), bins=50, density=True)
        ax5.set_title('Difference Distribution')
        ax5.set_xlabel('Difference')
        ax5.set_ylabel('Density')
        
        # Plot statistics
        ax6 = plt.subplot(236)
        ax6.axis('off')
        stats = [
            f"Shape: {matrix1.shape}",
            f"Mean Diff: {diff.mean():.4f}",
            f"Std Diff: {diff.std():.4f}",
            f"Max Abs Diff: {np.abs(diff).max():.4f}",
            f"Correlation: {np.corrcoef(matrix1.flatten(), matrix2.flatten())[0,1]:.4f}"
        ]
        ax6.text(0.1, 0.5, '\n'.join(stats),
                transform=ax6.transAxes,
                fontfamily='monospace')
        ax6.set_title('Comparison Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def plot_matrix_evolution(self,
                            matrices: list,
                            titles: list,
                            save_path: Path,
                            figsize: Optional[Tuple[int, int]] = None) -> None:
        """Create visualization of matrix evolution over time.
        
        Args:
            matrices: List of matrices showing evolution
            titles: List of titles for each matrix
            save_path: Path to save visualization
            figsize: Optional figure size (width, height)
        """
        if len(matrices) != len(titles):
            raise ValueError("Number of matrices must match number of titles")
            
        n = len(matrices)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        
        # Set up figure
        fig = plt.figure(figsize=figsize or (4*cols, 4*rows))
        
        # Plot each matrix
        for i, (matrix, title) in enumerate(zip(matrices, titles)):
            ax = plt.subplot(rows, cols, i+1)
            im = ax.imshow(matrix, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()