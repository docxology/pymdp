"""
RGM Matrix Validator
==================

Validates RGM matrices and their properties.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_svd_utils import RGMSVDUtils

class RGMMatrixValidator:
    """Validates RGM matrices"""
    
    def __init__(self):
        """Initialize matrix validator"""
        self.logger = RGMExperimentUtils.get_logger('matrix_validator')
        self.svd = RGMSVDUtils()
        
    def validate_matrices(self, matrices: Dict[str, np.ndarray], config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate all matrices and their relationships.
        
        Args:
            matrices: Dictionary of matrices to validate
            config: Configuration dictionary with expected properties
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            messages = []
            
            # Check individual matrices
            for name, matrix in matrices.items():
                matrix_messages = self._validate_single_matrix(name, matrix, config)
                messages.extend(matrix_messages)
                
            # Check hierarchical relationships
            hierarchy_messages = self._validate_hierarchy(matrices, config)
            messages.extend(hierarchy_messages)
            
            # Check numerical properties
            numerical_messages = self._validate_numerical_properties(matrices)
            messages.extend(numerical_messages)
            
            # Check consistency
            consistency_messages = self._validate_consistency(matrices, config)
            messages.extend(consistency_messages)
            
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error validating matrices: {str(e)}")
            raise
            
    def _validate_single_matrix(self, name: str, matrix: np.ndarray, config: Dict) -> List[str]:
        """Validate properties of single matrix"""
        messages = []
        
        try:
            # Get matrix configuration
            matrix_config = self._get_matrix_config(name, config)
            
            # Check basic properties
            if not self._check_basic_properties(matrix, name, messages):
                return messages
                
            # Check constraints
            if not self._check_constraints(matrix, matrix_config['constraints'], name, messages):
                return messages
                
            # Check normalization
            if not self._check_normalization(matrix, matrix_config['normalization'], name, messages):
                return messages
                
            # Check conditioning
            if matrix.ndim == 2:
                if not self._check_conditioning(matrix, name, messages):
                    return messages
                    
            return messages
            
        except Exception as e:
            self.logger.error(f"Error validating matrix {name}: {str(e)}")
            messages.append(f"Error validating {name}: {str(e)}")
            return messages
            
    def _check_basic_properties(self, matrix: np.ndarray, name: str, messages: List[str]) -> bool:
        """Check basic matrix properties"""
        # Check for NaN/Inf values
        if not np.all(np.isfinite(matrix)):
            messages.append(f"Matrix {name} contains NaN/Inf values")
            return False
            
        # Check non-negativity
        if np.any(matrix < 0):
            messages.append(f"Matrix {name} contains negative values")
            return False
            
        # Check shape validity
        if matrix.ndim not in [1, 2]:
            messages.append(f"Matrix {name} has invalid dimensions: {matrix.ndim}")
            return False
            
        return True
        
    def _check_constraints(self, matrix: np.ndarray, constraints: List[str], name: str, messages: List[str]) -> bool:
        """Check matrix constraints"""
        for constraint in constraints:
            if constraint == "column_normalized":
                if not np.allclose(matrix.sum(axis=0), 1.0, rtol=1e-5):
                    messages.append(f"Matrix {name} not column normalized")
                    return False
                    
            elif constraint == "non_negative":
                if np.any(matrix < 0):
                    messages.append(f"Matrix {name} contains negative values")
                    return False
                    
            elif constraint == "symmetric":
                if matrix.ndim == 2 and not np.allclose(matrix, matrix.T, rtol=1e-5):
                    messages.append(f"Matrix {name} is not symmetric")
                    return False
                    
            elif constraint == "normalized":
                if not np.isclose(matrix.sum(), 1.0, rtol=1e-5):
                    messages.append(f"Matrix {name} not normalized")
                    return False
                    
        return True
        
    def _check_normalization(self, matrix: np.ndarray, norm_config: Dict, name: str, messages: List[str]) -> bool:
        """Check matrix normalization"""
        if norm_config['type'] == 'svd':
            # Check SVD properties
            if matrix.ndim == 2:
                U, s, Vh = np.linalg.svd(matrix)
                
                # Check condition number
                cond = s[0] / s[-1]
                if cond > norm_config['max_condition_number']:
                    messages.append(
                        f"Matrix {name} poorly conditioned: {cond:.2e}"
                    )
                    return False
                    
                # Check singular values
                if np.any(s < norm_config['epsilon']):
                    messages.append(
                        f"Matrix {name} has small singular values"
                    )
                    return False
                    
        return True
        
    def _check_conditioning(self, matrix: np.ndarray, name: str, messages: List[str]) -> bool:
        """Check matrix conditioning"""
        try:
            cond = np.linalg.cond(matrix)
            if cond > 1e4:
                messages.append(
                    f"Matrix {name} poorly conditioned: {cond:.2e}"
                )
                return False
                
            return True
            
        except np.linalg.LinAlgError:
            messages.append(f"Matrix {name} is singular")
            return False
            
    def _validate_hierarchy(self, matrices: Dict[str, np.ndarray], config: Dict) -> List[str]:
        """Validate hierarchical relationships between matrices"""
        messages = []
        hierarchy = config['hierarchy']
        
        try:
            for level in range(hierarchy['n_levels']):
                curr_dim = hierarchy['dimensions'][f'level{level}']
                
                # Check A matrix shapes
                A_curr = matrices[f'A{level}']
                if A_curr.shape != (curr_dim['state'], curr_dim['factor']):
                    messages.append(
                        f"Invalid A{level} shape: {A_curr.shape} vs "
                        f"expected {(curr_dim['state'], curr_dim['factor'])}"
                    )
                    
                # Check B matrix shapes
                B_curr = matrices[f'B{level}']
                if B_curr.shape != (curr_dim['factor'], curr_dim['factor']):
                    messages.append(
                        f"Invalid B{level} shape: {B_curr.shape} vs "
                        f"expected {(curr_dim['factor'], curr_dim['factor'])}"
                    )
                    
                # Check D matrix shapes
                D_curr = matrices[f'D{level}']
                if D_curr.shape != (curr_dim['factor'],):
                    messages.append(
                        f"Invalid D{level} shape: {D_curr.shape} vs "
                        f"expected {(curr_dim['factor'],)}"
                    )
                    
            return messages
            
        except Exception as e:
            self.logger.error(f"Error validating hierarchy: {str(e)}")
            messages.append(f"Error validating hierarchy: {str(e)}")
            return messages
            
    def _validate_numerical_properties(self, matrices: Dict[str, np.ndarray]) -> List[str]:
        """Validate numerical properties across matrices"""
        messages = []
        
        try:
            for name, matrix in matrices.items():
                # Check matrix norms
                norm = np.linalg.norm(matrix)
                if norm > 100 or norm < 0.01:
                    messages.append(
                        f"Matrix {name} has extreme norm: {norm:.2e}"
                    )
                    
                # Check sparsity
                sparsity = np.mean(matrix == 0)
                if sparsity > 0.9:
                    messages.append(
                        f"Matrix {name} is too sparse: {sparsity:.2%}"
                    )
                    
                # Check symmetry for B matrices
                if name.startswith('B'):
                    if not np.allclose(matrix, matrix.T, rtol=1e-5):
                        messages.append(f"Matrix {name} is not symmetric")
                        
            return messages
            
        except Exception as e:
            self.logger.error(f"Error validating numerical properties: {str(e)}")
            messages.append(f"Error validating numerical properties: {str(e)}")
            return messages
            
    def _validate_consistency(self, matrices: Dict[str, np.ndarray], config: Dict) -> List[str]:
        """Validate consistency between matrices and configuration"""
        messages = []
        
        try:
            # Check matrix shapes match config
            shapes = config['model']['matrix_shapes']
            for name, expected in shapes.items():
                if name not in matrices:
                    messages.append(f"Missing matrix: {name}")
                    continue
                    
                matrix = matrices[name]
                if not isinstance(expected, list):
                    expected = [expected]  # Handle 1D case
                    
                if matrix.shape != tuple(expected):
                    messages.append(
                        f"Shape mismatch for {name}: {matrix.shape} vs {tuple(expected)}"
                    )
                    
            return messages
            
        except Exception as e:
            self.logger.error(f"Error validating consistency: {str(e)}")
            messages.append(f"Error validating consistency: {str(e)}")
            return messages
            
    def _get_matrix_config(self, name: str, config: Dict) -> Dict:
        """Get configuration for specific matrix"""
        try:
            if name.startswith('A'):
                return config['hierarchy']['matrices']['A']
            elif name.startswith('B'):
                return config['hierarchy']['matrices']['B']
            elif name.startswith('D'):
                return config['hierarchy']['matrices']['D']
            elif name == 'E':
                return config['hierarchy']['matrices']['E']
            else:
                raise ValueError(f"Unknown matrix type: {name}")
                
        except Exception as e:
            self.logger.error(f"Error getting matrix config: {str(e)}")
            raise
            
    def validate_normalization(self, matrices: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
        """Validate matrix normalization"""
        try:
            messages = []
            
            for name, matrix in matrices.items():
                # Check column normalization for A and B matrices
                if name.startswith(('A', 'B')):
                    col_sums = matrix.sum(axis=0)
                    if not np.allclose(col_sums, 1.0, rtol=1e-5):
                        messages.append(
                            f"Matrix {name} not column normalized: "
                            f"max deviation = {np.max(np.abs(col_sums - 1.0)):.2e}"
                        )
                        
                # Check sum-to-one for D and E matrices
                elif name.startswith(('D', 'E')):
                    total_sum = matrix.sum()
                    if not np.isclose(total_sum, 1.0, rtol=1e-5):
                        messages.append(
                            f"Matrix {name} not normalized: sum = {total_sum:.2e}"
                        )
                        
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error validating normalization: {str(e)}")
            raise
            
    def validate_conditioning(self, matrices: Dict[str, np.ndarray],
                            max_cond: float = 1e4) -> Tuple[bool, List[str]]:
        """Validate matrix conditioning"""
        try:
            messages = []
            
            for name, matrix in matrices.items():
                if matrix.ndim == 2:
                    cond = np.linalg.cond(matrix)
                    if cond > max_cond:
                        messages.append(
                            f"Matrix {name} poorly conditioned: {cond:.2e}"
                        )
                        
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error validating conditioning: {str(e)}")
            raise