"""
RGM Model State
=============

Manages the state of the Renormalization Generative Model during training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Iterator, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np

from .rgm_logging import RGMLogging
from .visualization_utils import RGMVisualizationUtils

class RGMModelState(nn.Module):
    """
    Maintains the computational state of the RGM model.
    
    Inherits from nn.Module to support PyTorch training pipeline.
    
    Handles:
    - Matrix storage and access
    - Level state management
    - Parameter management for training
    - Forward/backward passes
    - Device placement
    - Dynamic variable tracking
    """
    
    def __init__(self, matrices: Dict[str, torch.Tensor], device: Optional[torch.device] = None):
        """
        Initialize model state.
        
        Args:
            matrices: Dictionary of model matrices (R/G/L)
            device: Computation device (CPU/CUDA)
        """
        super().__init__()
        self.logger = RGMLogging.get_logger("rgm.model_state")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Register matrices as parameters
        for name, matrix in matrices.items():
            self.register_parameter(
                name, 
                nn.Parameter(matrix.to(self.device))
            )
        
        # Initialize hierarchical states
        self.states = {}
        for level in range(3):
            state_size = getattr(self, f"L{level}").shape[0]
            self.states[f"level{level}"] = torch.zeros(
                state_size, 
                device=self.device,
                requires_grad=True
            )
            
        # Initialize dynamic variables
        self.prediction_errors = {f"level{i}": None for i in range(3)}
        self.predictions = {f"level{i}": None for i in range(3)}
        self.factors = {f"level{i}": None for i in range(3)}
        
        # Training state
        self.training = True
        self.epoch = 0
        self.global_step = 0
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.prediction_loss = nn.MSELoss()
        self.sparsity_loss = nn.L1Loss()
        
        # Visualization directory
        self.vis_dir = None
        
        self.logger.debug(
            f"Initialized model state on {self.device} with "
            f"{len(matrices)} matrices and {len(self.states)} state tensors"
        )
        
    def set_visualization_dir(self, vis_dir: Path):
        """Set directory for saving visualizations."""
        self.vis_dir = vis_dir
        self.vis_dir.mkdir(exist_ok=True)
        
    def visualize_batch(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], batch_idx: int):
        """
        Visualize input batch and model outputs.
        
        Args:
            x: Input batch (B, 784)
            outputs: Model outputs dictionary
            batch_idx: Current batch index
        """
        if not self.vis_dir:
            return
            
        # Reshape images to 28x28
        input_images = x.reshape(-1, 28, 28).cpu().numpy()
        recon_images = outputs['predictions']['level0'].reshape(-1, 28, 28).detach().cpu().numpy()
        
        # Create figure with input-reconstruction pairs
        n_samples = min(5, len(input_images))
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        for i in range(n_samples):
            # Original
            axes[0,i].imshow(input_images[i], cmap='gray')
            axes[0,i].axis('off')
            if i == 0:
                axes[0,i].set_title('Input')
                
            # Reconstruction
            axes[1,i].imshow(recon_images[i], cmap='gray')
            axes[1,i].axis('off')
            if i == 0:
                axes[1,i].set_title('Reconstruction')
                
        plt.tight_layout()
        save_path = self.vis_dir / f"batch_{batch_idx:04d}.png"
        plt.savefig(save_path)
        plt.close()
        
        # Log reconstruction error
        mse = F.mse_loss(
            torch.tensor(input_images), 
            torch.tensor(recon_images)
        ).item()
        self.logger.debug(f"Batch {batch_idx} MSE: {mse:.4f}")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, 784)
            
        Returns:
            Dictionary containing:
            - output: Final output tensor (batch_size, 10)
            - states: Intermediate states
            - predictions: Generated predictions
            - errors: Prediction errors
            - loss: Total loss value
        """
        batch_size = x.shape[0]
        
        # Bottom-up pass (Recognition)
        h0 = F.relu(x @ self.R0.t())  # Level 0 state
        h1 = F.relu(h0 @ self.R1.t())  # Level 1 state
        h2 = F.relu(h1 @ self.R2.t())  # Level 2 state
        
        # Store states
        states = {
            'level0': h0,
            'level1': h1,
            'level2': h2
        }
        
        # Top-down pass (Generation)
        p2 = F.relu(h2 @ self.G2.t())  # Level 2 prediction
        p1 = F.relu(p2 @ self.G1.t())  # Level 1 prediction
        p0 = torch.sigmoid(p1 @ self.G0.t())  # Level 0 prediction (image reconstruction)
        
        # Store predictions
        predictions = {
            'level0': p0,
            'level1': p1,
            'level2': p2
        }
        
        # Compute prediction errors
        e0 = x - p0
        e1 = h0 - p1
        e2 = h1 - p2
        
        # Store prediction errors
        errors = {
            'level0': e0,
            'level1': e1,
            'level2': e2
        }
        
        # Compute losses
        recon_loss = self.reconstruction_loss(p0, x)
        pred_loss = (
            self.prediction_loss(p1, h0) + 
            self.prediction_loss(p2, h1)
        )
        sparsity_loss = (
            self.sparsity_loss(h0, torch.zeros_like(h0)) +
            self.sparsity_loss(h1, torch.zeros_like(h1)) +
            self.sparsity_loss(h2, torch.zeros_like(h2))
        )
        
        total_loss = recon_loss + pred_loss + 0.1 * sparsity_loss
        
        return {
            'output': h2,
            'states': states,
            'predictions': predictions,
            'errors': errors,
            'loss': total_loss,
            'loss_components': {
                'reconstruction': recon_loss.item(),
                'prediction': pred_loss.item(),
                'sparsity': sparsity_loss.item()
            }
        }
        
    def get_parameters(self) -> Iterator[nn.Parameter]:
        """Get trainable parameters for optimizer."""
        return self.parameters()
        
    def get_matrix(self, name: str) -> torch.Tensor:
        """Get matrix by name."""
        if not hasattr(self, name):
            raise KeyError(f"Matrix not found: {name}")
        return getattr(self, name)
        
    def get_state(self, level: int) -> torch.Tensor:
        """Get state tensor for given level."""
        key = f"level{level}"
        if key not in self.states:
            raise KeyError(f"State not found: {key}")
        return self.states[key]
        
    def set_state(self, level: int, state: torch.Tensor):
        """
        Set state tensor for given level.
        
        Args:
            level: Hierarchy level
            state: New state tensor
        """
        key = f"level{level}"
        if key not in self.states:
            raise KeyError(f"Invalid level: {level}")
        self.states[key] = state.to(self.device)
        
    def reset_states(self):
        """Reset all state tensors to zero."""
        for level in range(3):
            state_size = self.matrices[f"L{level}"].shape[0]
            self.states[f"level{level}"] = torch.zeros(
                state_size,
                device=self.device,
                requires_grad=True
            )
        self.logger.debug("Reset all state tensors to zero")
        
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        
    def state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            Dictionary containing model state
        """
        return {
            'matrices': self.matrices,
            'states': self.states,
            'epoch': self.epoch,
            'global_step': self.global_step
        }
        
    def load_state_dict(self, state_dict: Dict):
        """
        Load state from dictionary.
        
        Args:
            state_dict: Dictionary containing model state
        """
        self.matrices = {
            k: v.to(self.device) 
            for k, v in state_dict['matrices'].items()
        }
        self.states = {
            k: v.to(self.device)
            for k, v in state_dict['states'].items()
        }
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']