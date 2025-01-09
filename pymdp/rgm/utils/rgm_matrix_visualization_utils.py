"""
RGM Matrix Visualization Utilities
================================

Utilities for visualizing RGM matrices and results.
Provides functions for:
- Matrix heatmaps
- Factor analysis plots
- State evolution plots
- Inference results visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .rgm_experiment_utils import RGMExperimentUtils

class RGMMatrixVisualizationUtils:
    """Utilities for visualizing RGM matrices and results"""
    
    def __init__(self):
        """Initialize visualization utilities"""
        self.logger = RGMExperimentUtils.get_logger('visualizer')
        
        # Set up matplotlib style with modern, clean aesthetics
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': [10, 8],
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': '#f0f0f0',
            'figure.facecolor': 'white',
            'lines.linewidth': 2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        })
        
    def generate_visualizations(self, matrices: Dict[str, np.ndarray], save_dir: Path):
        """
        Generate visualizations for RGM matrices.
        
        Args:
            matrices: Dictionary of matrices to visualize
            save_dir: Directory to save visualizations
        """
        try:
            self.logger.info("Generating matrix visualizations...")
            
            # Create visualizations directory
            vis_dir = save_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # Generate heatmaps for each matrix
            for name, matrix in matrices.items():
                self.logger.debug(f"Generating heatmap for {name} matrix...")
                fig = self._generate_heatmap(matrix, name)
                
                # Save figure
                save_path = vis_dir / f"{name}_heatmap.png"
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                
            # Generate factor analysis plots
            self.logger.debug("Generating factor analysis plots...")
            self._generate_factor_plots(matrices, vis_dir)
            
            # Generate state evolution plots
            self.logger.debug("Generating state evolution plots...")
            self._generate_state_plots(matrices, vis_dir)
            
            self.logger.info(f"Visualizations saved to: {vis_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def _generate_heatmap(self, matrix: np.ndarray, name: str) -> plt.Figure:
        """Generate heatmap visualization"""
        try:
            # Create figure
            fig, ax = plt.subplots()
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Set title and labels
            ax.set_title(f"{name} Matrix")
            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")
            
            # Add grid
            ax.grid(False)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error generating heatmap for {name}: {str(e)}")
            raise
            
    def _generate_factor_plots(self, matrices: Dict[str, np.ndarray], save_dir: Path):
        """Generate factor analysis plots"""
        try:
            # Get factor matrices
            A = matrices.get('A')
            B = matrices.get('B')
            
            if A is None or B is None:
                self.logger.warning("Missing factor matrices, skipping factor plots")
                return
                
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot factor distributions
            ax1.hist(A.flatten(), bins=50, alpha=0.5, label='A')
            ax1.hist(B.flatten(), bins=50, alpha=0.5, label='B')
            ax1.set_title("Factor Matrix Distributions")
            ax1.set_xlabel("Value")
            ax1.set_ylabel("Count")
            ax1.legend()
            
            # Plot factor correlations
            ax2.scatter(A.flatten(), B.flatten(), alpha=0.1)
            ax2.set_title("Factor Matrix Correlations")
            ax2.set_xlabel("A Matrix Values")
            ax2.set_ylabel("B Matrix Values")
            
            # Save figure
            save_path = save_dir / "factor_analysis.png"
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Error generating factor plots: {str(e)}")
            raise
            
    def _generate_state_plots(self, matrices: Dict[str, np.ndarray], save_dir: Path):
        """Generate state evolution plots"""
        try:
            # Get state matrices
            D = matrices.get('D')
            E = matrices.get('E')
            
            if D is None or E is None:
                self.logger.warning("Missing state matrices, skipping state plots")
                return
                
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot state distributions
            ax1.hist(D.flatten(), bins=50, alpha=0.5, label='D')
            ax1.hist(E.flatten(), bins=50, alpha=0.5, label='E')
            ax1.set_title("State Matrix Distributions")
            ax1.set_xlabel("Value")
            ax1.set_ylabel("Count")
            ax1.legend()
            
            # Plot state correlations
            ax2.scatter(D.flatten(), E.flatten(), alpha=0.1)
            ax2.set_title("State Matrix Correlations")
            ax2.set_xlabel("D Matrix Values")
            ax2.set_ylabel("E Matrix Values")
            
            # Save figure
            save_path = save_dir / "state_evolution.png"
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Error generating state plots: {str(e)}")
            raise