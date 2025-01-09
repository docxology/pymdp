"""
RGM Model State Manager
===================

Manages the state of the Renormalization Generative Model based on 
renormalization group principles and the Free Energy Principle.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np

from .rgm_logging import RGMLogging

class RGMModelState:
    """Manages RGM model state and parameters."""
    
    def __init__(self, matrices: Dict[str, torch.Tensor], config: Dict):
        """Initialize model state."""
        self.logger = RGMLogging.get_logger("rgm.model_state")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate and reshape input matrices
        self._validate_matrices(matrices)
        matrices = self._reshape_matrices(matrices)
        
        # Initialize model components
        self.state = {
            'matrices': {k: v.to(self.device) for k, v in matrices.items()},
            'parameters': nn.ParameterDict(),
            'buffers': {},
            'config': config,
            'hierarchy_levels': self._determine_hierarchy_levels(matrices)
        }
        
        # Initialize trainable parameters
        self._initialize_parameters()
        
        # Build the model
        self.model = RGMModel(self.state).to(self.device)
        
    def _validate_matrices(self, matrices: Dict[str, torch.Tensor]):
        """Validate matrix dimensions and relationships."""
        # Check matrix existence
        required = ['A0', 'B0', 'D0']
        for name in required:
            if name not in matrices:
                raise ValueError(f"Missing required matrix: {name}")
        
        # Expected dimensions for each level
        dimensions = {
            0: {'input': 784, 'latent': 256},
            1: {'input': 256, 'latent': 64},
            2: {'input': 64, 'latent': 16}
        }
        
        # Validate dimensions
        levels = sum(1 for k in matrices if k.startswith('A'))
        for l in range(levels):
            A = matrices[f'A{l}']
            B = matrices[f'B{l}']
            D = matrices[f'D{l}']
            
            expected_input = dimensions[l]['input']
            expected_latent = dimensions[l]['latent']
            
            # Check recognition matrix (A)
            if A.size() != (expected_input, expected_latent):
                raise ValueError(
                    f"A{l} has wrong dimensions: {A.size()}, "
                    f"expected ({expected_input}, {expected_latent})"
                )
            
            # Check generative matrix (B)
            if B.size() != (expected_latent, expected_input):
                raise ValueError(
                    f"B{l} has wrong dimensions: {B.size()}, "
                    f"expected ({expected_latent}, {expected_input})"
                )
            
            # Check lateral connections (D)
            if D.size() != (expected_latent, expected_latent):
                raise ValueError(
                    f"D{l} must be square {expected_latent}x{expected_latent}, "
                    f"got {D.size()}"
                )
            
            # Additional checks
            if not torch.allclose(D, D.t()):
                self.logger.warning(f"D{l} is not symmetric")
            
            if torch.any(D < 0):
                self.logger.warning(f"D{l} contains negative values")
                
    def _reshape_matrices(self, matrices: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reshape matrices for proper dimensions."""
        reshaped = {}
        for name, matrix in matrices.items():
            if name.startswith('A'):
                # Recognition matrices (bottom-up)
                level = int(name[1])
                input_size = 784 if level == 0 else 256 // (4 ** (level-1))
                output_size = 256 // (4 ** level)
                reshaped[name] = matrix.reshape(input_size, output_size)
                
            elif name.startswith('B'):
                # Generative matrices (top-down)
                level = int(name[1])
                input_size = 256 // (4 ** level)
                output_size = 784 if level == 0 else 256 // (4 ** (level-1))
                reshaped[name] = matrix.reshape(input_size, output_size)
                
            elif name.startswith('D'):
                # Lateral connection matrices
                level = int(name[1])
                size = 256 // (4 ** level)
                reshaped[name] = matrix.reshape(size, size)
                
        return reshaped
        
    def _determine_hierarchy_levels(self, matrices: Dict[str, torch.Tensor]) -> int:
        """Determine number of hierarchy levels from matrices."""
        levels = 0
        while f'A{levels}' in matrices:
            levels += 1
        return levels
        
    def _initialize_parameters(self):
        """Initialize trainable parameters."""
        # Initialize hierarchy level parameters
        for level in range(self.state['hierarchy_levels']):
            # Weight matrices for each level
            self.state['parameters'][f'W{level}'] = nn.Parameter(
                torch.randn_like(self.state['matrices'][f'A{level}']) * 0.01,
                requires_grad=True
            )
            
            # Bias terms
            self.state['parameters'][f'b{level}'] = nn.Parameter(
                torch.zeros(self.state['matrices'][f'B{level}'].size(0)),
                requires_grad=True
            )
            
            # Scale factors
            self.state['parameters'][f'scale{level}'] = nn.Parameter(
                torch.ones(1),
                requires_grad=True
            )
        
        # Initialize global parameters
        self.state['parameters']['global_scale'] = nn.Parameter(
            torch.ones(1),
            requires_grad=True
        )
        
        self.logger.info(f"✓ Initialized {len(self.state['parameters'])} parameter groups")
        
    def get_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return list(self.state['parameters'].values())
        
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'matrices': {k: v.cpu() for k, v in self.state['matrices'].items()},
            'parameters': {k: v.cpu() for k, v in self.state['parameters'].items()},
            'buffers': {k: v.cpu() for k, v in self.state['buffers'].items()},
            'config': self.state['config'],
            'hierarchy_levels': self.state['hierarchy_levels']
        }
        torch.save(checkpoint, path)
        
    @classmethod
    def load_checkpoint(cls, path: Path, device: Optional[torch.device] = None):
        """Load model checkpoint."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        model_state = cls(checkpoint['matrices'], checkpoint['config'])
        
        # Load parameters and buffers
        for k, v in checkpoint['parameters'].items():
            model_state.state['parameters'][k] = nn.Parameter(v.to(device))
        for k, v in checkpoint['buffers'].items():
            model_state.state['buffers'][k] = v.to(device)
            
        return model_state

class RGMModel(nn.Module):
    """RGM Neural Network Model."""
    
    def __init__(self, state: Dict):
        """Initialize RGM model."""
        super().__init__()
        self.state = state
        self.hierarchy_levels = state['hierarchy_levels']
        
        # Create neural network layers
        self.layers = nn.ModuleList([
            RGMHierarchyLevel(
                state['matrices'][f'A{i}'],
                state['matrices'][f'B{i}'],
                state['matrices'][f'D{i}'],
                state['parameters'][f'W{i}'],
                state['parameters'][f'b{i}'],
                state['parameters'][f'scale{i}']
            )
            for i in range(self.hierarchy_levels)
        ])
        
        # Add batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(state['matrices'][f'B{i}'].size(0))
            for i in range(self.hierarchy_levels)
        ])
        
        # Add dropout
        self.dropout = nn.Dropout(p=state['config'].get('dropout_rate', 0.1))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        batch_size = x.size(0)
        outputs = {'layer_outputs': [], 'latents': []}
        current = x
        
        # Forward through hierarchy
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            current, latent = layer(current)
            current = bn(current)
            current = self.dropout(current)
            outputs['layer_outputs'].append(current)
            outputs['latents'].append(latent)
        
        # Final reconstruction
        outputs['reconstructed'] = self.layers[0].reconstruct(outputs['latents'][0])
        
        return outputs

class RGMHierarchyLevel(nn.Module):
    """Single level of the RGM hierarchy."""
    
    def __init__(self, A: torch.Tensor, B: torch.Tensor, D: torch.Tensor,
                 W: nn.Parameter, b: nn.Parameter, scale: nn.Parameter):
        """Initialize hierarchy level."""
        super().__init__()
        self.A = A
        self.B = B
        self.D = D
        self.W = W
        self.b = b
        self.scale = scale
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through hierarchy level."""
        # Compute latent representation
        latent = self.activation(torch.matmul(x, self.W) + self.b)
        latent = latent * self.scale
        
        # Transform through level matrices
        output = torch.matmul(latent, self.B)
        
        return output, latent
        
    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent representation."""
        return torch.matmul(latent, self.A)