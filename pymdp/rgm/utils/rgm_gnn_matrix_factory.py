"""
RGM Matrix Factory
===============

Factory class for creating and validating RGM matrices.
Handles initialization and constraints for recognition (R), generative (G), and lateral (L) matrices.
"""

import torch
from typing import Dict, List, Tuple
from pathlib import Path
import logging

from .rgm_logging import RGMLogging

class RGMMatrixFactory:
    """Factory for creating RGM matrices with proper initialization and constraints."""
    
    MATRIX_PREFIXES = {
        'recognition': 'R',
        'generative': 'G',
        'lateral': 'L'
    }
    
    def __init__(self, device: torch.device = None):
        """Initialize matrix factory."""
        self.logger = RGMLogging.get_logger("rgm.matrix_factory")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_matrices(self, specs: Dict) -> Dict[str, torch.Tensor]:
        """
        Create matrices from specifications.
        
        Args:
            specs: Dictionary containing matrix specifications
            
        Returns:
            Dictionary of initialized matrices
        """
        matrices = {}
        hierarchy = specs['hierarchy']
        matrix_specs = specs['matrices']
        
        # Validate matrix dimensions against hierarchy
        self._validate_matrix_dimensions(matrix_specs, hierarchy)
        
        # Create each type of matrix
        for matrix_type, type_specs in matrix_specs.items():
            for level in range(hierarchy['levels']):
                name = f"{matrix_type[0].upper()}{level}"  # A0, B0, D0, etc.
                if name not in type_specs:
                    raise ValueError(f"Missing {matrix_type} matrix for level {level}")
                    
                shape = type_specs[name]
                matrix = self._initialize_matrix(shape, matrix_type)
                matrices[name] = matrix
                
        return matrices
        
    def _validate_matrix_dimensions(self, matrix_specs: Dict, hierarchy: Dict):
        """Validate matrix dimensions against hierarchy specifications."""
        dimensions = hierarchy['dimensions']
        
        for level in range(hierarchy['levels']):
            level_dims = dimensions[f'level{level}']
            
            # Recognition matrices (A)
            a_name = f"A{level}"
            if a_name in matrix_specs['recognition']:
                shape = matrix_specs['recognition'][a_name]
                if shape[1] != level_dims['input']:
                    raise ValueError(
                        f"Recognition matrix {a_name} input dimension mismatch: "
                        f"expected {level_dims['input']}, got {shape[1]}"
                    )
                if shape[0] != level_dims['state']:
                    raise ValueError(
                        f"Recognition matrix {a_name} output dimension mismatch: "
                        f"expected {level_dims['state']}, got {shape[0]}"
                    )
                    
            # Generative matrices (B)
            b_name = f"B{level}"
            if b_name in matrix_specs['generative']:
                shape = matrix_specs['generative'][b_name]
                if shape[0] != level_dims['input']:
                    raise ValueError(
                        f"Generative matrix {b_name} output dimension mismatch: "
                        f"expected {level_dims['input']}, got {shape[0]}"
                    )
                if shape[1] != level_dims['state']:
                    raise ValueError(
                        f"Generative matrix {b_name} input dimension mismatch: "
                        f"expected {level_dims['state']}, got {shape[1]}"
                    )
                    
            # Lateral matrices (D)
            d_name = f"D{level}"
            if d_name in matrix_specs['lateral']:
                shape = matrix_specs['lateral'][d_name]
                if shape[0] != level_dims['state'] or shape[1] != level_dims['state']:
                    raise ValueError(
                        f"Lateral matrix {d_name} dimension mismatch: "
                        f"expected [{level_dims['state']}, {level_dims['state']}], "
                        f"got {shape}"
                    )
                    
    def _initialize_matrix(self, shape: List[int], matrix_type: str) -> torch.Tensor:
        """
        Initialize matrix with appropriate constraints.
        
        Args:
            shape: Matrix dimensions [rows, cols]
            matrix_type: Type of matrix ('recognition', 'generative', 'lateral')
            
        Returns:
            Initialized torch.Tensor with appropriate constraints
        """
        matrix = torch.empty(shape, device=self.device)
        torch.nn.init.xavier_uniform_(matrix)
        
        # Apply type-specific constraints
        if matrix_type == 'lateral':
            # Make symmetric for lateral connections
            matrix = 0.5 * (matrix + matrix.t())
        elif matrix_type == 'recognition':
            # Ensure positive weights for recognition
            matrix = torch.abs(matrix)
        elif matrix_type == 'generative':
            # Add small positive bias for generative
            matrix = matrix + 0.01
            
        return matrix
        
    def _get_matrix_name(self, matrix_type: str, level: int) -> str:
        """Get standardized matrix name."""
        prefix = self.MATRIX_PREFIXES[matrix_type]
        return f"{prefix}{level}"