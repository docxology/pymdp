"""
RGM GNN Matrix Factory
=====================

Generates matrices from GNN specifications.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_matrix_validator import RGMMatrixValidator
from .rgm_svd_utils import RGMSVDUtils
from .rgm_matrix_normalizer import RGMMatrixNormalizer

class RGMGNNMatrixFactory:
    """Generates matrices from GNN specifications"""
    
    def __init__(self):
        """Initialize GNN matrix factory"""
        self.logger = RGMExperimentUtils.get_logger('gnn_matrix_factory')
        self.validator = RGMMatrixValidator()
        self.svd = RGMSVDUtils()
        self.normalizer = RGMMatrixNormalizer()
        
    def generate_matrices(self, gnn_spec: Dict) -> Dict[str, np.ndarray]:
        """
        Generate all matrices from GNN specification.
        
        Args:
            gnn_spec: GNN specification dictionary
            
        Returns:
            Dictionary of generated matrices
        """
        try:
            self.logger.info("Generating matrices from GNN spec...")
            
            # Get hierarchy configuration
            hierarchy = gnn_spec['hierarchy']
            n_levels = hierarchy['n_levels']
            
            # Initialize matrices dictionary
            matrices = {}
            
            # Generate matrices for each level
            for level in range(n_levels):
                level_matrices = self._generate_level_matrices(
                    level,
                    hierarchy['dimensions'][f'level{level}'],
                    gnn_spec['hierarchy']['matrices']
                )
                matrices.update(level_matrices)
                
            # Validate matrices
            is_valid, messages = self.validator.validate_matrices(matrices, gnn_spec)
            if not is_valid:
                raise ValueError(f"Invalid matrices: {'; '.join(messages)}")
                
            # Normalize matrices
            matrices = self.normalizer.normalize_matrices(matrices)
            
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error generating matrices: {str(e)}")
            raise
            
    def _generate_level_matrices(self, level: int, dims: Dict, matrix_specs: Dict) -> Dict[str, np.ndarray]:
        """Generate matrices for a single level"""
        try:
            matrices = {}
            
            # Get dimensions
            state_dim = dims['state']
            factor_dim = dims['factor']
            
            # Generate A matrix (state transitions)
            A = self._initialize_matrix(
                (state_dim, factor_dim),
                matrix_specs['A'],
                f'A{level}'
            )
            matrices[f'A{level}'] = A
            
            # Generate B matrix (factor transitions)
            B = self._initialize_matrix(
                (factor_dim, factor_dim),
                matrix_specs['B'],
                f'B{level}'
            )
            matrices[f'B{level}'] = B
            
            # Generate D matrix (factor priors)
            D = self._initialize_matrix(
                (factor_dim,),
                matrix_specs['D'],
                f'D{level}'
            )
            matrices[f'D{level}'] = D
            
            # Generate E matrix for top level only
            if level == 0:
                E = self._initialize_matrix(
                    (state_dim,),
                    matrix_specs['E'],
                    'E'
                )
                matrices['E'] = E
                
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error generating level {level} matrices: {str(e)}")
            raise
            
    def _initialize_matrix(self, shape: Tuple[int, ...], spec: Dict, name: str) -> np.ndarray:
        """Initialize matrix according to specification"""
        try:
            init_config = spec['initialization']
            method = init_config['method']
            
            if method == 'random':
                matrix = self._initialize_random(shape, init_config)
            elif method == 'uniform':
                matrix = self._initialize_uniform(shape, init_config)
            elif method == 'zeros':
                matrix = np.zeros(shape)
            elif method == 'ones':
                matrix = np.ones(shape)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
                
            # Apply constraints
            matrix = self._apply_constraints(matrix, spec['constraints'], name)
            
            # Apply normalization
            if 'normalization' in spec:
                matrix = self._apply_normalization(matrix, spec['normalization'])
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error initializing matrix {name}: {str(e)}")
            raise
            
    def _initialize_random(self, shape: Tuple[int, ...], config: Dict) -> np.ndarray:
        """Initialize random matrix"""
        try:
            if config['distribution'] == 'truncated_normal':
                matrix = np.random.normal(
                    config['mean'],
                    config['std'],
                    shape
                )
                matrix = np.clip(matrix, config['min_val'], config['max_val'])
            else:
                raise ValueError(f"Unknown distribution: {config['distribution']}")
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error in random initialization: {str(e)}")
            raise
            
    def _initialize_uniform(self, shape: Tuple[int, ...], config: Dict) -> np.ndarray:
        """Initialize uniform matrix"""
        try:
            matrix = np.ones(shape)
            if len(shape) == 1:
                matrix = matrix / shape[0]
            else:
                matrix = matrix / shape[0]  # Column normalization
                
            # Add small constant for stability
            matrix = matrix + config.get('epsilon', 1e-12)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error in uniform initialization: {str(e)}")
            raise
            
    def _apply_constraints(self, matrix: np.ndarray, constraints: List[str], name: str) -> np.ndarray:
        """Apply constraints to matrix"""
        try:
            for constraint in constraints:
                if constraint == 'column_normalized':
                    matrix = self.normalizer.normalize_matrix(matrix, method='column')
                elif constraint == 'non_negative':
                    matrix = np.maximum(matrix, 0)
                elif constraint == 'symmetric':
                    if matrix.ndim == 2:
                        matrix = 0.5 * (matrix + matrix.T)
                elif constraint == 'normalized':
                    matrix = self.normalizer.normalize_matrix(matrix)
                else:
                    raise ValueError(f"Unknown constraint: {constraint}")
                    
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error applying constraints to {name}: {str(e)}")
            raise
            
    def _apply_normalization(self, matrix: np.ndarray, norm_config: Dict) -> np.ndarray:
        """Apply normalization to matrix"""
        try:
            if norm_config['type'] == 'svd':
                matrix = self.svd.condition_matrix(
                    matrix,
                    max_cond=norm_config['max_condition_number'],
                    eps=norm_config['epsilon']
                )
            else:
                raise ValueError(f"Unknown normalization type: {norm_config['type']}")
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error applying normalization: {str(e)}")
            raise