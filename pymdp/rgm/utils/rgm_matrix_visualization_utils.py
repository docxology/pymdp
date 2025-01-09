"""
RGM Matrix Visualization Utilities
===============================

Visualization utilities for the Renormalization Generative Model matrices.
"""

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

class RGMMatrixVisualizationUtils:
    """Visualization utilities for RGM matrices."""
    
    @staticmethod
    def plot_matrix(matrix: np.ndarray, title: str = "", save_path: Optional[Path] = None,
                   figsize: Tuple[int, int] = (10, 8), show_colorbar: bool = True):
        """
        Plot a matrix as a heatmap.
        
        Args:
            matrix: Matrix to visualize
            title: Title for the plot
            save_path: Optional path to save the plot
            figsize: Figure size (width, height)
            show_colorbar: Whether to show colorbar
        """
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot matrix
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            
            # Add colorbar if requested
            if show_colorbar:
                plt.colorbar(im, ax=ax)
            
            # Set title
            ax.set_title(title)
            
            # Add axis labels
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show plot
            if save_path:
                # Ensure directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error plotting matrix: {str(e)}")
            plt.close('all')  # Clean up any open figures
            
    @staticmethod
    def plot_matrix_grid(matrices: dict, save_path: Path, 
                        ncols: int = 3, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot multiple matrices in a grid layout.
        
        Args:
            matrices: Dictionary of matrices to plot
            save_path: Path to save the plot
            ncols: Number of columns in the grid
            figsize: Figure size (width, height)
        """
        try:
            n_matrices = len(matrices)
            nrows = (n_matrices + ncols - 1) // ncols
            
            # Create figure and axes
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            if nrows == 1:
                axes = axes.reshape(1, -1)
            
            # Plot each matrix
            for idx, (name, matrix) in enumerate(matrices.items()):
                row = idx // ncols
                col = idx % ncols
                ax = axes[row, col]
                
                im = ax.imshow(matrix, cmap='viridis', aspect='auto')
                ax.set_title(name)
                plt.colorbar(im, ax=ax)
                
            # Remove empty subplots
            for idx in range(n_matrices, nrows * ncols):
                row = idx // ncols
                col = idx % ncols
                fig.delaxes(axes[row, col])
                
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error plotting matrix grid: {str(e)}")
            plt.close('all')  # Clean up any open figures
            
    @staticmethod
    def plot_matrix_analysis(matrix: np.ndarray, title: str, save_path: Path):
        """
        Create detailed analysis plot for a matrix.
        
        Args:
            matrix: Matrix to analyze
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original matrix heatmap
            im0 = axes[0, 0].imshow(matrix, cmap='viridis', aspect='auto')
            axes[0, 0].set_title(f"{title} - Heatmap")
            plt.colorbar(im0, ax=axes[0, 0])
            
            # Value distribution histogram
            axes[0, 1].hist(matrix.flatten(), bins=50, density=True)
            axes[0, 1].set_title(f"{title} - Value Distribution")
            axes[0, 1].set_xlabel("Value")
            axes[0, 1].set_ylabel("Density")
            
            # Row-wise statistics
            row_means = matrix.mean(axis=1)
            row_stds = matrix.std(axis=1)
            x = np.arange(len(row_means))
            axes[1, 0].errorbar(x, row_means, yerr=row_stds, fmt='o', capsize=5)
            axes[1, 0].set_title(f"{title} - Row Statistics")
            axes[1, 0].set_xlabel("Row Index")
            axes[1, 0].set_ylabel("Mean ± Std")
            
            # Column-wise statistics
            col_means = matrix.mean(axis=0)
            col_stds = matrix.std(axis=0)
            x = np.arange(len(col_means))
            axes[1, 1].errorbar(x, col_means, yerr=col_stds, fmt='o', capsize=5)
            axes[1, 1].set_title(f"{title} - Column Statistics")
            axes[1, 1].set_xlabel("Column Index")
            axes[1, 1].set_ylabel("Mean ± Std")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error plotting matrix analysis: {str(e)}")
            plt.close('all')