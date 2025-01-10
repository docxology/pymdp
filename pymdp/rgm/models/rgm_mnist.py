"""RGM MNIST Model

This module implements the RGM model for MNIST digit recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any

class RGMMNISTModel(nn.Module):
    """RGM model for MNIST digit recognition.
    
    This model implements a Relational Graph Machine for MNIST digit recognition,
    using recognition, generative, and lateral matrices for information processing.
    
    Args:
        config: Model configuration dictionary
        device: Device to place the model on
        matrices: Pre-initialized matrices (optional)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        matrices: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ):
        """Initialize the RGM MNIST model."""
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Get architecture configuration
        arch_config = config['architecture']
        self.input_dim = arch_config['input_dim']
        self.hidden_dims = arch_config['hidden_dims']
        self.latent_dim = arch_config['latent_dim']
        self.matrix_size = arch_config['matrix_size']
        
        # Initialize or load matrices
        if matrices is not None:
            self.recognition_matrices = nn.ParameterDict({
                name: nn.Parameter(matrix)
                for name, matrix in matrices['recognition'].items()
            })
            
            self.generative_matrices = nn.ParameterDict({
                name: nn.Parameter(matrix)
                for name, matrix in matrices['generative'].items()
            })
            
            self.lateral_matrices = nn.ParameterDict({
                name: nn.Parameter(matrix)
                for name, matrix in matrices['lateral'].items()
            })
        else:
            # Initialize matrices from scratch if not provided
            from rgm.utils.matrix_init import initialize_matrices
            matrices = initialize_matrices(config, device)
            
            self.recognition_matrices = nn.ParameterDict({
                name: nn.Parameter(matrix)
                for name, matrix in matrices['recognition'].items()
            })
            
            self.generative_matrices = nn.ParameterDict({
                name: nn.Parameter(matrix)
                for name, matrix in matrices['generative'].items()
            })
            
            self.lateral_matrices = nn.ParameterDict({
                name: nn.Parameter(matrix)
                for name, matrix in matrices['lateral'].items()
            })
        
        # Initialize layers
        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
        # Layer normalization (if enabled)
        if self.config['model']['use_layer_norm']:
            self.layer_norms = nn.ModuleDict({
                'input': nn.LayerNorm(self.input_dim),
                **{
                    f'hidden_{i}': nn.LayerNorm(dim)
                    for i, dim in enumerate(self.hidden_dims)
                },
                'latent': nn.LayerNorm(self.latent_dim)
            })
        
        # Batch normalization (if enabled)
        if self.config['architecture']['use_batch_norm']:
            self.batch_norms = nn.ModuleDict({
                'input': nn.BatchNorm1d(self.input_dim),
                **{
                    f'hidden_{i}': nn.BatchNorm1d(dim)
                    for i, dim in enumerate(self.hidden_dims)
                },
                'latent': nn.BatchNorm1d(self.latent_dim)
            })
        
        # Dropout
        self.dropout = nn.Dropout(self.config['model']['dropout'])
        
        # Activation function
        activation_name = self.config['architecture']['activation'].lower()
        if activation_name == 'relu':
            self.activation = F.relu
        elif activation_name == 'leakyrelu':
            self.activation = F.leaky_relu
        elif activation_name == 'tanh':
            self.activation = F.tanh
        elif activation_name == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple containing:
            - loss: Total loss value
            - reconstruction_loss: Reconstruction loss component
            - state_loss: State prediction loss component
        """
        batch_size = x.shape[0]
        
        # Recognition pass (bottom-up)
        hidden_states = self._recognition_pass(x)
        
        # Generative pass (top-down)
        reconstructions = self._generative_pass(hidden_states)
        
        # Compute losses
        recon_loss = nn.functional.mse_loss(reconstructions, x)
        state_loss = self._compute_state_loss(hidden_states)
        total_loss = recon_loss + state_loss
        
        return total_loss, recon_loss, state_loss
    
    def _recognition_pass(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Recognition (bottom-up) pass through the network."""
        states = {'input': x}
        current = x
        
        # Process through hidden layers
        for i, dim in enumerate(self.hidden_dims):
            # Apply recognition matrix
            current = torch.matmul(
                self.recognition_matrices[f'hidden_{i}'],
                current.t()
            ).t()
            
            # Apply lateral connections
            if self.config['model']['use_residual']:
                lateral = torch.matmul(
                    self.lateral_matrices[f'hidden_{i}'],
                    current.t()
                ).t()
                current = current + lateral
            
            # Apply normalization and activation
            if self.config['architecture']['use_batch_norm']:
                current = self.batch_norms[f'hidden_{i}'](current)
            if self.config['model']['use_layer_norm']:
                current = self.layer_norms[f'hidden_{i}'](current)
            
            current = self.activation(current)
            current = self.dropout(current)
            
            states[f'hidden_{i}'] = current
        
        # Final transformation to latent space
        latent = torch.matmul(
            self.recognition_matrices['latent'],
            current.t()
        ).t()
        
        if self.config['model']['use_residual']:
            lateral = torch.matmul(
                self.lateral_matrices['latent'],
                latent.t()
            ).t()
            latent = latent + lateral
        
        if self.config['architecture']['use_batch_norm']:
            latent = self.batch_norms['latent'](latent)
        if self.config['model']['use_layer_norm']:
            latent = self.layer_norms['latent'](latent)
        
        latent = self.activation(latent)
        states['latent'] = latent
        
        return states
    
    def _generative_pass(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generative (top-down) pass through the network."""
        current = states['latent']
        
        # Process through hidden layers in reverse
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            # Apply generative matrix
            current = torch.matmul(
                self.generative_matrices[f'hidden_{i}'],
                current.t()
            ).t()
            
            # Apply lateral connections
            if self.config['model']['use_residual']:
                lateral = torch.matmul(
                    self.lateral_matrices[f'hidden_{i}'],
                    current.t()
                ).t()
                current = current + lateral
            
            # Apply normalization and activation
            if self.config['architecture']['use_batch_norm']:
                current = self.batch_norms[f'hidden_{i}'](current)
            if self.config['model']['use_layer_norm']:
                current = self.layer_norms[f'hidden_{i}'](current)
            
            current = self.activation(current)
            current = self.dropout(current)
        
        # Final reconstruction
        reconstruction = torch.matmul(
            self.generative_matrices['input'],
            current.t()
        ).t()
        
        return reconstruction
    
    def _compute_state_loss(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute state prediction loss."""
        state_loss = 0.0
        
        # Compute prediction errors between adjacent layers
        for i in range(len(self.hidden_dims)):
            pred = torch.matmul(
                self.generative_matrices[f'hidden_{i}'],
                states[f'hidden_{i}'].t()
            ).t()
            
            target = states[f'hidden_{i+1}'] if i < len(self.hidden_dims) - 1 else states['latent']
            state_loss += nn.functional.mse_loss(pred, target)
        
        return state_loss
    
    def get_matrices(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get the current state of all matrices.
        
        Returns:
            Dictionary containing all matrices
        """
        return {
            'recognition': {
                name: matrix.detach()
                for name, matrix in self.recognition_matrices.items()
            },
            'generative': {
                name: matrix.detach()
                for name, matrix in self.generative_matrices.items()
            },
            'lateral': {
                name: matrix.detach()
                for name, matrix in self.lateral_matrices.items()
            }
        }
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Dict[str, Any],
        device: torch.device
    ) -> 'RGMMNISTModel':
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(config=config, device=device, matrices=checkpoint['matrices'])
        model.load_state_dict(checkpoint['model_state'])
        return model
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on provided data.
        
        Args:
            data_loader: DataLoader containing evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_state_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                loss, recon_loss, state_loss = self(batch)
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_state_loss += state_loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'state_loss': total_state_loss / num_batches
        } 