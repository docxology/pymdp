"""
RGM Matrix Handler
===============

Handles matrix operations and validation for the Renormalization Generative Model.

Matrix Hierarchy
--------------
The RGM uses a hierarchical structure of matrices for bottom-up recognition,
top-down generation, and lateral connections:

Recognition Matrices (R):
- R0: [256, 784] - Level 0 → Level 1 (feature extraction)
- R1: [64, 256]  - Level 1 → Level 2 (part composition)
- R2: [16, 64]   - Level 2 → Output (digit classification)

Generative Matrices (G):
- G0: [784, 256] - Level 1 → Level 0 (image reconstruction)
- G1: [256, 64]  - Level 2 → Level 1 (part generation)
- G2: [64, 16]   - Output → Level 2 (digit generation)

Lateral Matrices (L):
- L0: [256, 256] - Level 0 lateral connections
- L1: [64, 64]   - Level 1 lateral connections
- L2: [16, 16]   - Level 2 lateral connections

Matrix Operations
---------------
1. Recognition (Bottom-up):
   x₁ = R₀x₀ + L₀x₀  # Level 0 → 1
   x₂ = R₁x₁ + L₁x₁  # Level 1 → 2
   y  = R₂x₂ + L₂x₂  # Level 2 → Output

2. Generation (Top-down):
   x̂₂ = G₂y          # Output → Level 2
   x̂₁ = G₁x̂₂        # Level 2 → 1
   x̂₀ = G₀x̂₁        # Level 1 → 0

Note: All operations handle batched inputs [batch_size, feature_dim]
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .rgm_logging import RGMLogging

class RGMMatrixHandler:
    """Handles matrix operations for RGM."""
    
    def __init__(self):
        """Initialize matrix handler."""
        self.logger = RGMLogging.get_logger("rgm.matrix_handler")
        
        # Define expected dimensions
        self.dimensions = {
            'input': 784,    # 28x28 MNIST
            'level0': 256,   # First level features
            'level1': 64,    # Second level features
            'level2': 16,    # Third level features
            'output': 16     # Classification output
        }
        
        # Define expected matrix shapes
        self.matrix_shapes = {
            # Recognition matrices
            'R0': (self.dimensions['level0'], self.dimensions['input']),
            'R1': (self.dimensions['level1'], self.dimensions['level0']),
            'R2': (self.dimensions['level2'], self.dimensions['level1']),
            
            # Generative matrices
            'G0': (self.dimensions['input'], self.dimensions['level0']),
            'G1': (self.dimensions['level0'], self.dimensions['level1']),
            'G2': (self.dimensions['level1'], self.dimensions['level2']),
            
            # Lateral matrices
            'L0': (self.dimensions['level0'], self.dimensions['level0']),
            'L1': (self.dimensions['level1'], self.dimensions['level1']),
            'L2': (self.dimensions['level2'], self.dimensions['level2'])
        }
        
        # Input/output dimensions for each level
        self.level_dims = {
            0: (self.dimensions['input'], self.dimensions['level0']),
            1: (self.dimensions['level0'], self.dimensions['level1']),
            2: (self.dimensions['level1'], self.dimensions['level2'])
        }
        
    def _validate_batch_input(self, x: torch.Tensor, expected_feature_dim: int, name: str) -> None:
        """
        Validate batch input dimensions.
        
        Args:
            x: Input tensor [batch_size, feature_dim]
            expected_feature_dim: Expected feature dimension
            name: Name for error messages
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        if len(x.shape) != 2:
            raise ValueError(
                f"Expected 2D input tensor [batch_size, {expected_feature_dim}] for {name}, "
                f"got shape {x.shape}"
            )
            
        if x.shape[1] != expected_feature_dim:
            raise ValueError(
                f"Invalid feature dimension for {name}: "
                f"expected {expected_feature_dim}, got {x.shape[1]}"
            )
            
    def _reshape_if_needed(self, x: torch.Tensor, expected_feature_dim: int, name: str) -> torch.Tensor:
        """
        Reshape input tensor if needed to match expected dimensions.
        
        Args:
            x: Input tensor
            expected_feature_dim: Expected feature dimension
            name: Name for error messages
            
        Returns:
            Reshaped tensor [batch_size, feature_dim]
        """
        if len(x.shape) == 1:
            # Single sample, add batch dimension
            x = x.unsqueeze(0)
            
        elif len(x.shape) > 2:
            # Try to flatten extra dimensions
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            
        if x.shape[1] != expected_feature_dim:
            raise ValueError(
                f"Cannot reshape input for {name}: "
                f"expected feature dim {expected_feature_dim}, got {x.shape[1]}"
            )
            
        return x
        
    def validate_matrix(self, matrix: torch.Tensor, name: str) -> None:
        """
        Validate matrix dimensions.
        
        Args:
            matrix: Matrix to validate
            name: Matrix name (e.g., 'R0', 'G1', 'L2')
            
        Raises:
            ValueError: If matrix dimensions don't match expectations
        """
        expected_shape = self.matrix_shapes.get(name)
        if expected_shape is None:
            raise ValueError(f"Unknown matrix name: {name}")
            
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Invalid shape for matrix {name}: "
                f"expected {expected_shape}, got {matrix.shape}"
            )
            
    def validate_matrices(self, matrices: Dict[str, torch.Tensor]) -> None:
        """
        Validate all matrices in the hierarchy.
        
        Args:
            matrices: Dictionary of matrices
            
        Raises:
            ValueError: If any matrix is invalid
        """
        for name, matrix in matrices.items():
            self.validate_matrix(matrix, name)
            
        # Validate recognition-generative pairs
        for level in range(3):
            r_name = f'R{level}'
            g_name = f'G{level}'
            if matrices[r_name].shape != matrices[g_name].t().shape:
                raise ValueError(
                    f"Recognition-generative matrix pair mismatch at level {level}: "
                    f"{r_name} shape {matrices[r_name].shape} != "
                    f"{g_name}ᵀ shape {matrices[g_name].t().shape}"
                )
                
        self.logger.info("✓ All matrices validated successfully")
        
    def initialize_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Initialize matrices with correct dimensions.
        
        Returns:
            Dictionary of initialized matrices
        """
        matrices = {}
        
        try:
            for name, shape in self.matrix_shapes.items():
                if name.startswith('L'):
                    # Initialize lateral matrices as identity
                    matrices[name] = torch.eye(shape[0])
                else:
                    # Initialize other matrices with Xavier/Glorot
                    matrices[name] = torch.empty(shape)
                    nn.init.xavier_uniform_(matrices[name])
                    
            self.validate_matrices(matrices)
            return matrices
            
        except Exception as e:
            self.logger.error(f"Failed to initialize matrices: {str(e)}")
            raise
            
    def matrix_multiply(self, matrix: torch.Tensor, x: torch.Tensor, name: str) -> torch.Tensor:
        """
        Perform matrix multiplication with proper batch handling.
        
        Args:
            matrix: Weight matrix [out_dim, in_dim]
            x: Input tensor [batch_size, in_dim]
            name: Matrix name for error messages
            
        Returns:
            Output tensor [batch_size, out_dim]
        """
        try:
            # Validate and reshape input if needed
            x = self._reshape_if_needed(x, matrix.shape[1], name)
            self._validate_batch_input(x, matrix.shape[1], name)
            
            # Perform matrix multiplication
            return torch.matmul(x, matrix.t())
            
        except Exception as e:
            self.logger.error(f"Matrix multiplication failed for {name}: {str(e)}")
            raise
            
    def apply_recognition(self, x: torch.Tensor, matrices: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply recognition (bottom-up) pass.
        
        Args:
            x: Input tensor [batch_size, 784]
            matrices: Dictionary of matrices
            
        Returns:
            List of activations at each level [batch_size, dim]
        """
        try:
            # Validate input
            x = self._reshape_if_needed(x, self.dimensions['input'], 'input')
            self._validate_batch_input(x, self.dimensions['input'], 'input')
            
            activations = [x]
            
            # Level 0 → 1
            x1 = self.matrix_multiply(matrices['R0'], x, 'R0')
            x1 = x1 + self.matrix_multiply(matrices['L0'], x1, 'L0')
            activations.append(x1)
            
            # Level 1 → 2
            x2 = self.matrix_multiply(matrices['R1'], x1, 'R1')
            x2 = x2 + self.matrix_multiply(matrices['L1'], x2, 'L1')
            activations.append(x2)
            
            # Level 2 → Output
            y = self.matrix_multiply(matrices['R2'], x2, 'R2')
            y = y + self.matrix_multiply(matrices['L2'], y, 'L2')
            activations.append(y)
            
            return activations
            
        except Exception as e:
            self.logger.error(f"Recognition pass failed: {str(e)}")
            raise
            
    def apply_generation(self, y: torch.Tensor, matrices: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply generation (top-down) pass.
        
        Args:
            y: Top-level tensor [batch_size, 16]
            matrices: Dictionary of matrices
            
        Returns:
            List of reconstructions at each level [batch_size, dim]
        """
        try:
            # Validate input
            y = self._reshape_if_needed(y, self.dimensions['output'], 'output')
            self._validate_batch_input(y, self.dimensions['output'], 'output')
            
            reconstructions = [y]
            
            # Output → Level 2
            x2 = self.matrix_multiply(matrices['G2'], y, 'G2')
            reconstructions.append(x2)
            
            # Level 2 → 1
            x1 = self.matrix_multiply(matrices['G1'], x2, 'G1')
            reconstructions.append(x1)
            
            # Level 1 → 0
            x0 = self.matrix_multiply(matrices['G0'], x1, 'G0')
            reconstructions.append(x0)
            
            return reconstructions[::-1]  # Reverse to match forward order
            
        except Exception as e:
            self.logger.error(f"Generation pass failed: {str(e)}")
            raise 