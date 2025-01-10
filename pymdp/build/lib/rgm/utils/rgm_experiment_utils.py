"""
RGM Experiment Utilities

This module provides utilities for managing RGM experiments,
including directory setup, checkpointing, and result analysis.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class RGMExperimentUtils:
    """Utilities for managing RGM experiments."""
    
    def __init__(self, exp_dir: Path):
        """Initialize experiment utilities.
        
        Args:
            exp_dir: Root experiment directory
        """
        self.exp_dir = Path(exp_dir)
        self._setup_directories()
        
    def _setup_directories(self):
        """Create experiment directory structure."""
        # Create main directories
        dirs = [
            self.exp_dir,
            self.exp_dir / "checkpoints",
            self.exp_dir / "logs",
            self.exp_dir / "results",
            self.exp_dir / "visualizations"
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            
    def save_checkpoint(self,
                       state: Dict[str, Any],
                       name: str,
                       is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            state: Dictionary containing model state
            name: Name for the checkpoint
            is_best: Whether this is the best model so far
        """
        # Save checkpoint
        checkpoint_dir = self.exp_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save(state, checkpoint_path)
        
        # If best model, create a copy
        if is_best:
            best_path = checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            
    def load_checkpoint(self,
                       name: str,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            name: Name of checkpoint to load
            device: Optional device to load checkpoint to
            
        Returns:
            Dictionary containing model state
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self.exp_dir / "checkpoints" / f"{name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        return torch.load(checkpoint_path, map_location=device)
        
    def save_metrics(self, metrics: Dict[str, float], name: str) -> None:
        """Save evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            name: Name for the metrics file
        """
        metrics_dir = self.exp_dir / "results"
        metrics_path = metrics_dir / f"{name}.json"
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def plot_training_curves(self,
                           metrics: Dict[str, list],
                           save_name: str) -> None:
        """Plot training curves from collected metrics.
        
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
        
        # Save plot
        save_path = self.exp_dir / "visualizations" / f"{save_name}.png"
        plt.savefig(save_path)
        plt.close()
        
    def analyze_matrix_evolution(self,
                               initial_matrices: Dict[str, torch.Tensor],
                               final_matrices: Dict[str, torch.Tensor],
                               save_name: str) -> None:
        """Analyze how matrices evolved during training.
        
        Args:
            initial_matrices: Dictionary of matrices at start of training
            final_matrices: Dictionary of matrices at end of training
            save_name: Name for the output file
        """
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle('Matrix Evolution Analysis', fontsize=14)
        
        # For each matrix type
        for i, (name, init_matrix) in enumerate(initial_matrices.items()):
            final_matrix = final_matrices[name]
            
            # Convert to numpy for analysis
            init_np = init_matrix.cpu().numpy()
            final_np = final_matrix.cpu().numpy()
            
            # Compute changes
            diff = final_np - init_np
            rel_change = np.abs(diff) / (np.abs(init_np) + 1e-8)
            
            # Plot
            ax = plt.subplot(1, 3, i+1)
            im = ax.imshow(rel_change, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'Matrix {name}\nRelative Change')
            
        plt.tight_layout()
        
        # Save analysis
        save_path = self.exp_dir / "visualizations" / f"{save_name}.png"
        plt.savefig(save_path)
        plt.close()