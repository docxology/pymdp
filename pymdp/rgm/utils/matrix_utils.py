"""
Matrix Utilities
==============

Utility functions for matrix operations in RGM.
"""

import torch
from typing import Tuple, Optional

def initialize_matrix(name: str, shape: Tuple[int, int], init_config: dict) -> torch.Tensor:
    """
    Initialize matrix according to configuration.
    
    Args:
        name: Matrix name (e.g., 'A0', 'B0', 'D0')
        shape: Matrix shape as (rows, cols)
        init_config: Initialization configuration
        
    Returns:
        Initialized matrix
    """
    method = init_config['method']
    
    if method == 'orthogonal':
        gain = init_config.get('gain', 1.0)
        matrix = initialize_orthogonal(shape, gain)
        
    elif method == 'normal':
        mean = init_config.get('mean', 0.0)
        std = init_config.get('std', 0.01)
        matrix = initialize_normal(shape, mean, std)
        
    elif method == 'identity':
        noise_std = init_config.get('noise_std', 0.001)
        matrix = initialize_identity_plus_noise(shape, noise_std)
        
    else:
        raise ValueError(f"Unknown initialization method: {method}")
        
    return matrix

def initialize_orthogonal(shape: Tuple[int, int], gain: float = 1.0) -> torch.Tensor:
    """Initialize matrix with orthogonal weights."""
    matrix = torch.nn.init.orthogonal_(
        torch.empty(shape),
        gain=gain
    )
    return matrix

def initialize_normal(
    shape: Tuple[int, int], 
    mean: float = 0.0, 
    std: float = 0.01
) -> torch.Tensor:
    """Initialize matrix with normal distribution."""
    matrix = torch.nn.init.normal_(
        torch.empty(shape),
        mean=mean,
        std=std
    )
    return matrix

def initialize_identity_plus_noise(
    shape: Tuple[int, int], 
    noise_std: float = 0.001
) -> torch.Tensor:
    """Initialize matrix as identity plus noise."""
    if shape[0] != shape[1]:
        raise ValueError(f"Identity initialization requires square matrix, got shape {shape}")
    matrix = torch.eye(shape[0])
    noise = torch.randn_like(matrix) * noise_std
    return matrix + noise

def ensure_positive_definite(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Ensure matrix is positive definite."""
    # Make symmetric
    matrix = 0.5 * (matrix + matrix.t())
    
    # Add small positive diagonal
    matrix = matrix + torch.eye(matrix.shape[0]) * epsilon
    
    return matrix

def orthogonalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonalize matrix using QR decomposition."""
    Q, R = torch.linalg.qr(matrix)
    return Q 