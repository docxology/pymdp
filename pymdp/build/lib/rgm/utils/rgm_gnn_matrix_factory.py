"""
GNN Matrix Factory

This module provides utilities for creating and initializing matrices
used in the GNN component of the RGM model.
"""

from typing import Dict, Any
import torch
import torch.nn as nn

class RGMGNNMatrixFactory:
    """Factory for creating GNN matrices with proper initialization."""
    
    def __init__(self):
        """Initialize the matrix factory."""
        pass
        
    def create_matrices(self, specs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Create matrices according to specifications.
        
        Args:
            specs: Dictionary containing matrix specifications
            
        Returns:
            Dictionary mapping matrix names to initialized tensors
            
        Raises:
            ValueError: If specifications are invalid
        """
        matrices = {}
        
        # Get matrix specifications
        matrix_specs = specs['matrices']
        
        # Create recognition matrices
        for name, dims in matrix_specs['recognition'].items():
            matrices[name] = self._initialize_recognition_matrix(dims)
            
        # Create generative matrices
        for name, dims in matrix_specs['generative'].items():
            matrices[name] = self._initialize_generative_matrix(dims)
            
        # Create lateral matrices
        for name, dims in matrix_specs['lateral'].items():
            matrices[name] = self._initialize_lateral_matrix(dims)
            
        return matrices
        
    def _initialize_recognition_matrix(self, dims: list) -> torch.Tensor:
        """Initialize recognition matrix with Xavier uniform initialization.
        
        Args:
            dims: List containing [output_dim, input_dim]
            
        Returns:
            Initialized matrix tensor
        """
        matrix = torch.empty(dims[0], dims[1])
        nn.init.xavier_uniform_(matrix, gain=nn.init.calculate_gain('relu'))
        return matrix
        
    def _initialize_generative_matrix(self, dims: list) -> torch.Tensor:
        """Initialize generative matrix with Xavier uniform initialization.
        
        Args:
            dims: List containing [output_dim, input_dim]
            
        Returns:
            Initialized matrix tensor
        """
        matrix = torch.empty(dims[0], dims[1])
        nn.init.xavier_uniform_(matrix, gain=nn.init.calculate_gain('tanh'))
        return matrix
        
    def _initialize_lateral_matrix(self, dims: list) -> torch.Tensor:
        """Initialize lateral matrix with identity + small noise.
        
        Args:
            dims: List containing [dim, dim] (must be square)
            
        Returns:
            Initialized matrix tensor
            
        Raises:
            ValueError: If dimensions are not equal
        """
        if dims[0] != dims[1]:
            raise ValueError(
                f"Lateral matrix must be square: got dims {dims}"
            )
            
        # Start with identity
        matrix = torch.eye(dims[0])
        
        # Add small uniform noise
        noise = torch.empty(dims[0], dims[1])
        nn.init.uniform_(noise, -0.01, 0.01)
        matrix += noise
        
        return matrix