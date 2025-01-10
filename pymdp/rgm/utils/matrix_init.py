"""RGM Matrix Initialization

This module handles the initialization of matrices for the RGM model.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List

def initialize_matrices(config: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Initialize matrices for RGM model.
    
    Args:
        config: Model configuration dictionary
        device: Device to initialize matrices on
        
    Returns:
        Dictionary containing initialized matrices
    """
    arch_config = config['architecture']
    model_config = config['model']
    
    # Get dimensions
    input_dim = arch_config['input_dim']
    hidden_dims = arch_config['hidden_dims']
    latent_dim = arch_config['latent_dim']
    matrix_size = arch_config['matrix_size']
    
    # Initialize matrices
    matrices = {}
    
    # Recognition matrices (bottom-up)
    matrices['recognition'] = _initialize_recognition_matrices(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        matrix_size=matrix_size,
        init_method=model_config['initialization']['method'],
        gain=model_config['initialization']['gain'],
        device=device
    )
    
    # Generative matrices (top-down)
    matrices['generative'] = _initialize_generative_matrices(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        matrix_size=matrix_size,
        init_method=model_config['initialization']['method'],
        gain=model_config['initialization']['gain'],
        device=device
    )
    
    # Lateral matrices (within-layer)
    matrices['lateral'] = _initialize_lateral_matrices(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        matrix_size=matrix_size,
        init_method=model_config['initialization']['method'],
        gain=model_config['initialization']['gain'],
        device=device
    )
    
    return matrices

def _initialize_recognition_matrices(
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    matrix_size: int,
    init_method: str,
    gain: float,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Initialize recognition (bottom-up) matrices."""
    matrices = {}
    
    # Input to first hidden layer
    matrices['input'] = _init_matrix(
        (matrix_size, input_dim),
        init_method,
        gain,
        device
    )
    
    # Hidden layer matrices
    for i, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
        matrices[f'hidden_{i}'] = _init_matrix(
            (matrix_size, in_dim),
            init_method,
            gain,
            device
        )
    
    # Final hidden to latent
    matrices['latent'] = _init_matrix(
        (matrix_size, hidden_dims[-1]),
        init_method,
        gain,
        device
    )
    
    return matrices

def _initialize_generative_matrices(
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    matrix_size: int,
    init_method: str,
    gain: float,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Initialize generative (top-down) matrices."""
    matrices = {}
    
    # Latent to last hidden layer
    matrices['latent'] = _init_matrix(
        (matrix_size, latent_dim),
        init_method,
        gain,
        device
    )
    
    # Hidden layer matrices (reverse order)
    for i, (out_dim, in_dim) in enumerate(zip(hidden_dims[-2::-1], hidden_dims[::-1])):
        matrices[f'hidden_{i}'] = _init_matrix(
            (matrix_size, in_dim),
            init_method,
            gain,
            device
        )
    
    # First hidden to input
    matrices['input'] = _init_matrix(
        (matrix_size, hidden_dims[0]),
        init_method,
        gain,
        device
    )
    
    return matrices

def _initialize_lateral_matrices(
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    matrix_size: int,
    init_method: str,
    gain: float,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Initialize lateral (within-layer) matrices."""
    matrices = {}
    
    # Input layer lateral
    matrices['input'] = _init_matrix(
        (matrix_size, matrix_size),
        init_method,
        gain,
        device
    )
    
    # Hidden layer laterals
    for i, dim in enumerate(hidden_dims):
        matrices[f'hidden_{i}'] = _init_matrix(
            (matrix_size, matrix_size),
            init_method,
            gain,
            device
        )
    
    # Latent layer lateral
    matrices['latent'] = _init_matrix(
        (matrix_size, matrix_size),
        init_method,
        gain,
        device
    )
    
    return matrices

def _init_matrix(
    size: Tuple[int, int],
    method: str,
    gain: float,
    device: torch.device
) -> torch.Tensor:
    """Initialize a single matrix using specified method."""
    matrix = torch.empty(size, device=device)
    
    if method == 'xavier_uniform':
        nn.init.xavier_uniform_(matrix, gain=gain)
    elif method == 'xavier_normal':
        nn.init.xavier_normal_(matrix, gain=gain)
    elif method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(matrix, a=gain)
    elif method == 'kaiming_normal':
        nn.init.kaiming_normal_(matrix, a=gain)
    elif method == 'identity_with_noise':
        matrix = torch.eye(size[0], size[1], device=device)
        matrix += torch.randn_like(matrix) * gain
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return matrix 