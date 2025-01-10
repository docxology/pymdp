"""
RGM Matrix Factory
================

Handles matrix generation and initialization for RGM.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_matrix_validator import RGMMatrixValidator
from .rgm_svd_utils import RGMSVDUtils
from .rgm_matrix_normalizer import RGMMatrixNormalizer

class RGMMatrixFactory:
    """Generates matrices for RGM"""
    
    def __init__(self):
        """Initialize matrix factory"""
        self.logger = RGMExperimentUtils.get_logger('matrix_factory')
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
            
    def _generate_level_matrices(self,
                                level: int,
                                dimensions: Dict,
                                matrices: Dict) -> Dict[str, np.ndarray]:
        """
        Generate matrices for a given level.
        
        Args:
            level: Hierarchy level
            dimensions: Dimensions for the given level
            matrices: Matrices for the given level
            
        Returns:
            Dictionary of generated matrices for the given level
        """
        try:
            # Initialize matrices dictionary
            level_matrices = {}
            
            # Generate A matrix (state transitions)
            A = self._generate_A_matrix(
                level,
                dimensions['state'],
                dimensions['factor'],
                matrices['A']
            )
            level_matrices['A'] = A
            
            # Generate B matrix (factor transitions)
            B = self._generate_B_matrix(
                level,
                dimensions['factor'],
                matrices['B']
            )
            level_matrices['B'] = B
            
            # Generate D matrix (factor priors)
            D = self._generate_D_matrix(
                level,
                dimensions['factor'],
                matrices['D']
            )
            level_matrices['D'] = D
            
            return level_matrices
            
        except Exception as e:
            self.logger.error(f"Error generating level matrices: {str(e)}")
            raise
            
    def _generate_A_matrix(self,
                           level: int,
                           state_dim: int,
                           factor_dim: int,
                           A_matrix: np.ndarray) -> np.ndarray:
        """
        Generate state transition matrix for a given level.
        
        Args:
            level: Hierarchy level
            state_dim: State dimension
            factor_dim: Factor dimension
            A_matrix: State transition matrix for the given level
            
        Returns:
            State transition matrix for the given level
        """
        try:
            # Initialize with random values
            A = np.random.rand(state_dim, factor_dim)
            
            # Apply SVD normalization
            A = self.svd.normalize_matrix(A)
            
            # Validate shape and properties
            if not self.validator.validate_A_matrix(A, level, {'state': state_dim, 'factor': factor_dim}):
                raise ValueError(f"Invalid A{level} matrix")
                
            return A
            
        except Exception as e:
            self.logger.error(f"Error generating A{level} matrix: {str(e)}")
            raise
            
    def _generate_B_matrix(self,
                           level: int,
                           factor_dim: int,
                           B_matrix: np.ndarray) -> np.ndarray:
        """
        Generate factor transition matrix for a given level.
        
        Args:
            level: Hierarchy level
            factor_dim: Factor dimension
            B_matrix: Factor transition matrix for the given level
            
        Returns:
            Factor transition matrix for the given level
        """
        try:
            # Initialize with random values
            B = np.random.rand(factor_dim, factor_dim)
            
            # Apply normalization
            B = self.normalizer.normalize_matrix(B)
            
            # Validate
            if not self.validator.validate_B_matrix(B, level, {'factor': factor_dim}):
                raise ValueError(f"Invalid B{level} matrix")
                
            return B
            
        except Exception as e:
            self.logger.error(f"Error generating B{level} matrix: {str(e)}")
            raise
            
    def _generate_D_matrix(self,
                           level: int,
                           factor_dim: int,
                           D_matrix: np.ndarray) -> np.ndarray:
        """
        Generate factor prior matrix for a given level.
        
        Args:
            level: Hierarchy level
            factor_dim: Factor dimension
            D_matrix: Factor prior matrix for the given level
            
        Returns:
            Factor prior matrix for the given level
        """
        try:
            # Initialize uniform prior
            D = np.ones(factor_dim) / factor_dim
            
            # Validate
            if not self.validator.validate_D_matrix(D, level, {'factor': factor_dim}):
                raise ValueError(f"Invalid D{level} matrix")
                
            return D
            
        except Exception as e:
            self.logger.error(f"Error generating D{level} matrix: {str(e)}")
            raise
            
    def _generate_E_matrix(self, gnn_specs: Dict) -> np.ndarray:
        """Generate output mapping matrix"""
        try:
            # Get dimensions from top level
            dims = self._get_level_dimensions(gnn_specs, 3)  # Top level
            output_dim = dims['factor']  # 10 for MNIST
            
            # Initialize uniform mapping
            E = np.ones(output_dim) / output_dim
            
            # Validate
            if not self.validator.validate_E_matrix(E, dims):
                raise ValueError("Invalid E matrix")
                
            return E
            
        except Exception as e:
            self.logger.error(f"Error generating E matrix: {str(e)}")
            raise
            
    def _get_level_dimensions(self, gnn_specs: Dict, level: int) -> Dict:
        """Get dimensions for given hierarchy level"""
        try:
            level_spec = gnn_specs.get(f'level{level}')
            if not level_spec:
                raise ValueError(f"Missing specification for level {level}")
                
            dims = level_spec.get('dimensions')
            if not dims:
                raise ValueError(f"Missing dimensions for level {level}")
                
            required = ['state', 'factor']
            if not all(dim in dims for dim in required):
                raise ValueError(f"Missing required dimensions for level {level}")
                
            return dims
            
        except Exception as e:
            self.logger.error(f"Error getting dimensions: {str(e)}")
            raise
            
    def _validate_matrices(self, matrices: Dict[str, np.ndarray]):
        """Validate all generated matrices"""
        try:
            for name, matrix in matrices.items():
                # Check normalization
                if name.startswith(('A', 'B')):
                    col_sums = matrix.sum(axis=0)
                    if not np.allclose(col_sums, 1.0, rtol=1e-5):
                        raise ValueError(f"Matrix {name} not properly normalized")
                        
                # Check shapes
                if name.startswith('A'):
                    level = int(name[1])
                    dims = self._get_level_dimensions(self.gnn_specs, level)
                    expected_shape = (dims['state'], dims['factor'])
                    if matrix.shape != expected_shape:
                        raise ValueError(
                            f"Invalid shape for {name}: "
                            f"expected {expected_shape}, got {matrix.shape}"
                        )
                        
                # Log validation success
                self.logger.info(f"Validated {name} matrix: OK")
                
        except Exception as e:
            self.logger.error(f"Error validating matrices: {str(e)}")
            raise
            
    def _validate_A_matrix(self,
                          matrix: np.ndarray,
                          level: int,
                          dims: Dict) -> bool:
        """Validate A matrix properties"""
        try:
            # Check shape
            expected_shape = (dims['state'], dims['factor'])
            if matrix.shape != expected_shape:
                self.logger.error(
                    f"Invalid shape for A{level}: "
                    f"expected {expected_shape}, got {matrix.shape}"
                )
                return False
                
            # Check normalization
            if not np.allclose(matrix.sum(axis=0), 1.0, rtol=1e-5):
                self.logger.error(f"A{level} matrix not properly normalized")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating A{level} matrix: {str(e)}")
            raise
            
    def _validate_B_matrix(self,
                          matrix: np.ndarray,
                          level: int,
                          dims: Dict) -> bool:
        """Validate B matrix properties"""
        try:
            # Check shape
            factor_dim = dims['factor']
            expected_shape = (factor_dim, factor_dim)
            if matrix.shape != expected_shape:
                self.logger.error(
                    f"Invalid shape for B{level}: "
                    f"expected {expected_shape}, got {matrix.shape}"
                )
                return False
                
            # Check normalization
            if not np.allclose(matrix.sum(axis=0), 1.0, rtol=1e-5):
                self.logger.error(f"B{level} matrix not properly normalized")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating B{level} matrix: {str(e)}") 