"""
RGM Visualization Module

This module provides visualization utilities for the RGM model,
including matrix rendering and analysis tools.
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt

from .utils.rgm_matrix_visualization_utils import RGMMatrixVisualizationUtils
from .utils.rgm_gnn_matrix_factory import RGMGNNMatrixFactory

class RGMRenderer:
    """Renderer for RGM model components and results."""
    
    def __init__(self,
                 exp_dir: Path,
                 config: dict,
                 device: Optional[torch.device] = None):
        """Initialize the renderer.
        
        Args:
            exp_dir: Experiment directory for saving outputs
            config: Model configuration dictionary
            device: Optional torch device
        """
        self.exp_dir = Path(exp_dir)
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create visualization directory
        self.vis_dir = self.exp_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize utilities
        self.matrix_vis = RGMMatrixVisualizationUtils()
        self.matrix_factory = RGMGNNMatrixFactory()
        
    def generate_matrices(self) -> Dict[str, torch.Tensor]:
        """Generate initial matrices based on configuration.
        
        Returns:
            Dictionary mapping matrix names to initialized tensors
        """
        matrices = self.matrix_factory.create_matrices(self.config)
        
        # Move matrices to device
        for name, matrix in matrices.items():
            matrices[name] = matrix.to(self.device)
            
        # Visualize initial matrices
        self.visualize_matrices(matrices, prefix="initial")
        
        return matrices
        
    def visualize_matrices(self,
                         matrices: Dict[str, torch.Tensor],
                         prefix: str = "") -> None:
        """Create visualizations for a set of matrices.
        
        Args:
            matrices: Dictionary of matrices to visualize
            prefix: Optional prefix for output filenames
        """
        for name, matrix in matrices.items():
            # Convert to numpy for visualization
            matrix_np = matrix.cpu().numpy()
            
            # Create visualization
            save_path = self.vis_dir / f"{prefix}_matrix_{name.lower()}.png"
            self.matrix_vis.plot_matrix_analysis(
                matrix=matrix_np,
                title=f"Matrix {name}",
                save_path=save_path
            )
            
    def visualize_matrix_evolution(self,
                                 initial_matrices: Dict[str, torch.Tensor],
                                 final_matrices: Dict[str, torch.Tensor]) -> None:
        """Visualize how matrices evolved during training.
        
        Args:
            initial_matrices: Matrices at start of training
            final_matrices: Matrices at end of training
        """
        for name in initial_matrices:
            # Get corresponding matrices
            init_matrix = initial_matrices[name].cpu().numpy()
            final_matrix = final_matrices[name].cpu().numpy()
            
            # Create comparison visualization
            save_path = self.vis_dir / f"evolution_matrix_{name.lower()}.png"
            self.matrix_vis.plot_matrix_comparison(
                matrix1=init_matrix,
                matrix2=final_matrix,
                title1="Initial",
                title2="Final",
                save_path=save_path
            )
            
    def visualize_mnist_samples(self,
                              samples: torch.Tensor,
                              reconstructions: torch.Tensor,
                              save_name: str) -> None:
        """Visualize MNIST samples and their reconstructions.
        
        Args:
            samples: Original MNIST samples [N, 784]
            reconstructions: Reconstructed samples [N, 784]
            save_name: Name for the output file
        """
        # Select first 10 samples
        n_samples = min(10, samples.shape[0])
        samples = samples[:n_samples]
        reconstructions = reconstructions[:n_samples]
        
        # Reshape to images
        samples = samples.reshape(-1, 28, 28).cpu().numpy()
        reconstructions = reconstructions.reshape(-1, 28, 28).cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        # Plot original samples
        for i, ax in enumerate(axes[0]):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Original', pad=10)
                
        # Plot reconstructions
        for i, ax in enumerate(axes[1]):
            ax.imshow(reconstructions[i], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Reconstructed', pad=10)
                
        plt.tight_layout()
        
        # Save visualization
        save_path = self.vis_dir / f"{save_name}.png"
        plt.savefig(save_path)
        plt.close()
        
    def visualize_training_progress(self,
                                  metrics: Dict[str, list],
                                  save_name: str = "training_curves") -> None:
        """Create visualization of training progress.
        
        Args:
            metrics: Dictionary mapping metric names to lists of values
            save_name: Name for the output file
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress', fontsize=14)
        
        # Plot loss curves
        if 'train_loss' in metrics and 'val_loss' in metrics:
            ax = axes[0, 0]
            ax.plot(metrics['train_loss'], label='Train')
            ax.plot(metrics['val_loss'], label='Validation')
            ax.set_title('Total Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            
        # Plot reconstruction loss
        if 'recon_loss' in metrics:
            ax = axes[0, 1]
            ax.plot(metrics['recon_loss'])
            ax.set_title('Reconstruction Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True)
            
        # Plot state prediction loss
        if 'state_loss' in metrics:
            ax = axes[1, 0]
            ax.plot(metrics['state_loss'])
            ax.set_title('State Prediction Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True)
            
        # Plot learning rate
        if 'learning_rate' in metrics:
            ax = axes[1, 1]
            ax.plot(metrics['learning_rate'])
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('LR')
            ax.grid(True)
            
        plt.tight_layout()
        
        # Save visualization
        save_path = self.vis_dir / f"{save_name}.png"
        plt.savefig(save_path)
        plt.close() 