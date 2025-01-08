"""
RGM Matrix Normalizer
====================

Handles matrix normalization and conditioning for RGM.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_svd_utils import RGMSVDUtils

class RGMMatrixNormalizer:
    """Handles matrix normalization and conditioning"""
    
    def __init__(self):
        """Initialize matrix normalizer"""
        self.logger = RGMExperimentUtils.get_logger('normalizer')
        self.svd = RGMSVDUtils()
        
    def normalize_matrices(self, matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize all matrices according to their types.
        
        Args:
            matrices: Dictionary of matrices to normalize
            
        Returns:
            Dictionary of normalized matrices
        """
        try:
            normalized = {}
            
            for name, matrix in matrices.items():
                if name.startswith('A'):
                    normalized[name] = self._normalize_state_matrix(matrix, name)
                elif name.startswith('B'):
                    normalized[name] = self._normalize_factor_matrix(matrix, name)
                elif name.startswith('D'):
                    normalized[name] = self._normalize_prior_matrix(matrix, name)
                elif name == 'E':
                    normalized[name] = self._normalize_output_matrix(matrix)
                else:
                    self.logger.warning(f"Unknown matrix type: {name}")
                    normalized[name] = matrix
                    
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing matrices: {str(e)}")
            raise
            
    def _normalize_state_matrix(self, matrix: np.ndarray, name: str, eps: float = 1e-12) -> np.ndarray:
        """Normalize state transition matrix"""
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            # Column normalization
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
            
            # Condition matrix
            matrix = self.svd.condition_matrix(matrix)
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Final normalization
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
            
            # Log conditioning
            cond = np.linalg.cond(matrix)
            self.logger.info(f"Normalized {name} matrix - condition number: {cond:.2e}")
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error normalizing state matrix {name}: {str(e)}")
            raise
            
    def _normalize_factor_matrix(self, matrix: np.ndarray, name: str, eps: float = 1e-12) -> np.ndarray:
        """Normalize factor transition matrix"""
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            # Make slightly diagonal dominant
            matrix = matrix + np.eye(matrix.shape[0]) * 0.1
            
            # Column normalization
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
            
            # Condition matrix
            matrix = self.svd.condition_matrix(matrix)
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Make symmetric
            matrix = 0.5 * (matrix + matrix.T)
            
            # Final normalization
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
            
            # Log conditioning
            cond = np.linalg.cond(matrix)
            self.logger.info(f"Normalized {name} matrix - condition number: {cond:.2e}")
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error normalizing factor matrix {name}: {str(e)}")
            raise
            
    def _normalize_prior_matrix(self, matrix: np.ndarray, name: str, eps: float = 1e-12) -> np.ndarray:
        """Normalize prior matrix"""
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            # Normalize to sum to 1
            matrix = matrix / matrix.sum()
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Final normalization
            matrix = matrix / matrix.sum()
            
            self.logger.info(f"Normalized {name} matrix - sum: {matrix.sum():.6f}")
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error normalizing prior matrix {name}: {str(e)}")
            raise
            
    def _normalize_output_matrix(self, matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Normalize output mapping matrix"""
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            # Normalize to sum to 1
            matrix = matrix / matrix.sum()
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Final normalization
            matrix = matrix / matrix.sum()
            
            self.logger.info(f"Normalized E matrix - sum: {matrix.sum():.6f}")
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error normalizing output matrix: {str(e)}")
            raise
            
    def check_normalization(self, matrices: Dict[str, np.ndarray], 
                          tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
        """
        Check normalization of all matrices.
        
        Args:
            matrices: Dictionary of matrices to check
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            messages = []
            
            for name, matrix in matrices.items():
                # Check non-negativity
                if np.any(matrix < 0):
                    messages.append(f"Matrix {name} contains negative values")
                    
                # Check normalization
                if name.startswith(('A', 'B')):
                    col_sums = matrix.sum(axis=0)
                    if not np.allclose(col_sums, 1.0, rtol=tolerance):
                        messages.append(
                            f"Matrix {name} columns not normalized: "
                            f"max deviation = {np.max(np.abs(col_sums - 1.0)):.2e}"
                        )
                elif name.startswith(('D', 'E')):
                    total_sum = matrix.sum()
                    if not np.isclose(total_sum, 1.0, rtol=tolerance):
                        messages.append(
                            f"Matrix {name} not normalized: sum = {total_sum:.2e}"
                        )
                        
                # Check symmetry for B matrices
                if name.startswith('B'):
                    if not np.allclose(matrix, matrix.T, rtol=tolerance):
                        messages.append(f"Matrix {name} is not symmetric")
                        
                # Check conditioning for square matrices
                if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                    cond = np.linalg.cond(matrix)
                    if cond > 1e4:
                        messages.append(
                            f"Matrix {name} poorly conditioned: {cond:.2e}"
                        )
                        
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error checking normalization: {str(e)}")
            messages.append(f"Error checking normalization: {str(e)}")
            return False, messages