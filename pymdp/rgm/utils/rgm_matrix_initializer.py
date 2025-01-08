"""
RGM Matrix Initializer
=====================

Handles matrix initialization and conditioning for RGM.
Ensures proper initialization, normalization and numerical stability.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from rgm_experiment_utils import RGMExperimentUtils
from rgm_svd_utils import RGMSVDUtils
from rgm_matrix_normalizer import RGMMatrixNormalizer

class RGMMatrixInitializer:
    """Handles matrix initialization and conditioning"""
    
    def __init__(self, config: Dict):
        """
        Initialize matrix initializer.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = RGMExperimentUtils.get_logger('matrix_init')
        self.config = config
        self.svd = RGMSVDUtils()
        self.normalizer = RGMMatrixNormalizer()
        
    def initialize_matrices(self, hierarchy_config: Dict) -> Dict[str, np.ndarray]:
        """
        Initialize all matrices for RGM hierarchy.
        
        Args:
            hierarchy_config: Hierarchy configuration dictionary
            
        Returns:
            Dictionary of initialized matrices
        """
        try:
            matrices = {}
            n_levels = hierarchy_config['n_levels']
            
            for level in range(n_levels):
                level_matrices = self._initialize_level_matrices(
                    level, 
                    hierarchy_config['dimensions'][f'level{level}']
                )
                matrices.update(level_matrices)
                
            # Initialize output matrix if needed
            if self.config['model'].get('use_output_matrix', True):
                matrices['E'] = self._initialize_output_matrix(
                    hierarchy_config['dimensions']['level0']['state']
                )
                
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error initializing matrices: {str(e)}")
            raise
            
    def _initialize_level_matrices(self, level: int, dims: Dict) -> Dict[str, np.ndarray]:
        """Initialize matrices for a single level"""
        try:
            matrices = {}
            
            # Get dimensions
            state_dim = dims['state']
            factor_dim = dims['factor']
            
            # Initialize A matrix (state transitions)
            A = self._initialize_state_matrix(state_dim, factor_dim)
            matrices[f'A{level}'] = A
            
            # Initialize B matrix (factor transitions)
            B = self._initialize_factor_matrix(factor_dim)
            matrices[f'B{level}'] = B
            
            # Initialize D matrix (factor priors)
            D = self._initialize_prior_matrix(factor_dim)
            matrices[f'D{level}'] = D
            
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error initializing level {level} matrices: {str(e)}")
            raise
            
    def _initialize_state_matrix(self, state_dim: int, factor_dim: int) -> np.ndarray:
        """Initialize state transition matrix"""
        try:
            # Get initialization parameters
            init_config = self.config['model']['matrices']['A']
            
            # Initialize with truncated normal distribution
            matrix = np.random.normal(
                init_config['mean'],
                init_config['std'],
                (state_dim, factor_dim)
            )
            
            # Clip values
            matrix = np.clip(
                matrix,
                init_config['min_val'],
                init_config['max_val']
            )
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Column normalize
            matrix = self.normalizer.normalize_matrix(matrix, method='column')
            
            # Condition matrix
            matrix = self.svd.condition_matrix(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error initializing state matrix: {str(e)}")
            raise
            
    def _initialize_factor_matrix(self, factor_dim: int) -> np.ndarray:
        """Initialize factor transition matrix"""
        try:
            # Get initialization parameters
            init_config = self.config['model']['matrices']['B']
            
            # Initialize with truncated normal distribution
            matrix = np.random.normal(
                init_config['mean'],
                init_config['std'],
                (factor_dim, factor_dim)
            )
            
            # Clip values
            matrix = np.clip(
                matrix,
                init_config['min_val'],
                init_config['max_val']
            )
            
            # Ensure symmetry
            matrix = 0.5 * (matrix + matrix.T)
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Column normalize
            matrix = self.normalizer.normalize_matrix(matrix, method='column')
            
            # Condition matrix
            matrix = self.svd.condition_matrix(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error initializing factor matrix: {str(e)}")
            raise
            
    def _initialize_prior_matrix(self, factor_dim: int) -> np.ndarray:
        """Initialize factor prior matrix"""
        try:
            # Initialize uniform distribution
            matrix = np.ones(factor_dim)
            
            # Add small noise for symmetry breaking
            matrix += np.random.uniform(0, 0.1, factor_dim)
            
            # Normalize
            matrix = self.normalizer.normalize_matrix(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error initializing prior matrix: {str(e)}")
            raise
            
    def _initialize_output_matrix(self, state_dim: int) -> np.ndarray:
        """Initialize output mapping matrix"""
        try:
            # Initialize uniform distribution
            matrix = np.ones(state_dim)
            
            # Add small noise for symmetry breaking
            matrix += np.random.uniform(0, 0.1, state_dim)
            
            # Normalize
            matrix = self.normalizer.normalize_matrix(matrix)
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error initializing output matrix: {str(e)}")
            raise