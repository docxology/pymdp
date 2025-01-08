"""
RGM Matrix Visualization Utilities
================================

Handles visualization of RGM matrices and their properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_svd_utils import RGMSVDUtils

class RGMMatrixVisualizationUtils:
    """Handles matrix visualization"""
    
    def __init__(self):
        """Initialize visualization utilities"""
        self.logger = RGMExperimentUtils.get_logger('matrix_viz')
        self.experiment = RGMExperimentUtils.get_experiment()
        self.svd = RGMSVDUtils()
        
        # Set style
        plt.style.use('seaborn')
        
    def generate_visualizations(self, matrices: Dict[str, np.ndarray], output_dir: Path):
        """
        Generate visualizations for all matrices.
        
        Args:
            matrices: Dictionary of matrices to visualize
            output_dir: Output directory for visualizations
        """
        try:
            self.logger.info("Generating matrix visualizations...")
            
            # Create visualization directory
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Generate visualizations for each matrix
            for name, matrix in matrices.items():
                self._visualize_matrix(name, matrix, viz_dir)
                
            # Generate comparison plots
            self._generate_comparison_plots(matrices, viz_dir)
            
            self.logger.info(f"Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def _visualize_matrix(self, name: str, matrix: np.ndarray, output_dir: Path):
        """Generate visualizations for a single matrix"""
        try:
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, cmap='viridis', center=0)
            plt.title(f"{name} Matrix Heatmap")
            plt.savefig(output_dir / f"{name}_heatmap.png")
            plt.close()
            
            # Plot singular value spectrum if 2D
            if matrix.ndim == 2:
                self._plot_singular_values(name, matrix, output_dir)
                
            # Plot distribution
            plt.figure(figsize=(8, 6))
            sns.histplot(matrix.flatten(), bins=50)
            plt.title(f"{name} Matrix Distribution")
            plt.savefig(output_dir / f"{name}_distribution.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error visualizing matrix {name}: {str(e)}")
            raise
            
    def _plot_singular_values(self, name: str, matrix: np.ndarray, output_dir: Path):
        """Plot singular value spectrum"""
        try:
            # Compute SVD
            spectrum = self.svd.analyze_spectrum(matrix)
            
            # Plot spectrum
            plt.figure(figsize=(8, 6))
            plt.semilogy(spectrum['singular_values'])
            plt.title(f"{name} Matrix Singular Values")
            plt.xlabel("Index")
            plt.ylabel("Singular Value")
            plt.grid(True)
            plt.savefig(output_dir / f"{name}_spectrum.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting singular values for {name}: {str(e)}")
            raise
            
    def _generate_comparison_plots(self, matrices: Dict[str, np.ndarray], output_dir: Path):
        """Generate comparison plots between matrices"""
        try:
            # Compare condition numbers
            condition_numbers = {}
            for name, matrix in matrices.items():
                if matrix.ndim == 2:
                    condition_numbers[name] = np.linalg.cond(matrix)
                    
            if condition_numbers:
                plt.figure(figsize=(10, 6))
                plt.bar(condition_numbers.keys(), condition_numbers.values())
                plt.title("Matrix Condition Numbers")
                plt.yscale('log')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "condition_numbers.png")
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error generating comparison plots: {str(e)}")
            raise