"""
RGM SVD Utilities
================

Handles SVD-based matrix operations and conditioning.
Ensures proper numerical stability and matrix properties.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import linalg
import logging

from .rgm_experiment_utils import RGMExperimentUtils

class RGMSVDUtils:
    """Handles SVD-based matrix operations"""
    
    def __init__(self):
        """Initialize SVD utilities"""
        self.logger = RGMExperimentUtils.get_logger('svd_utils')
        
    def condition_matrix(self, matrix: np.ndarray, max_cond: float = 1e4, 
                        eps: float = 1e-12) -> np.ndarray:
        """
        Condition matrix using SVD to improve numerical stability.
        
        Args:
            matrix: Matrix to condition
            max_cond: Maximum allowed condition number
            eps: Small constant for numerical stability
            
        Returns:
            Conditioned matrix
        """
        try:
            # Skip if matrix is 1D
            if matrix.ndim == 1:
                return matrix
                
            # Add small constant for stability
            matrix = matrix + eps
            
            # Compute SVD
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            
            # Get condition number
            cond = s[0] / s[-1]
            
            # Only condition if needed
            if cond > max_cond:
                self.logger.info(f"Conditioning matrix with condition number {cond:.2e}")
                
                # Compute threshold
                threshold = s[0] / max_cond
                
                # Clip singular values
                s_new = np.maximum(s, threshold)
                
                # Reconstruct matrix
                matrix = U @ np.diag(s_new) @ Vh
                
                # Log new condition number
                new_cond = s_new[0] / s_new[-1]
                self.logger.info(f"New condition number: {new_cond:.2e}")
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error conditioning matrix: {str(e)}")
            raise
            
    def normalize_matrix(self, matrix: np.ndarray, method: str = 'column',
                        eps: float = 1e-12) -> np.ndarray:
        """
        Normalize matrix using specified method.
        
        Args:
            matrix: Matrix to normalize
            method: Normalization method ('column', 'row', 'frobenius')
            eps: Small constant for numerical stability
            
        Returns:
            Normalized matrix
        """
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            if method == 'column':
                # Column normalization
                norms = np.sum(matrix, axis=0)
                norms[norms == 0] = 1.0  # Avoid division by zero
                return matrix / norms
                
            elif method == 'row':
                # Row normalization
                norms = np.sum(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # Avoid division by zero
                return matrix / norms
                
            elif method == 'frobenius':
                # Frobenius normalization
                norm = np.linalg.norm(matrix)
                if norm == 0:
                    return matrix
                return matrix / norm
                
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error normalizing matrix: {str(e)}")
            raise
            
    def analyze_spectrum(self, matrix: np.ndarray, eps: float = 1e-12) -> Dict:
        """
        Analyze singular value spectrum.
        
        Args:
            matrix: Matrix to analyze
            eps: Small constant for numerical stability
            
        Returns:
            Dictionary of spectral properties
        """
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            # Compute SVD
            _, s, _ = np.linalg.svd(matrix, full_matrices=False)
            
            # Compute properties
            properties = {
                'singular_values': s.tolist(),
                'condition_number': float(s[0] / s[-1]),
                'rank': int(np.linalg.matrix_rank(matrix)),
                'effective_rank': float(np.sum(s > 1e-10)),
                'spectral_gap': float(s[0] - s[1]) if len(s) > 1 else 0.0,
                'decay_ratio': float(s[-1] / s[0]),
                'energy_95': float(
                    np.sum(np.cumsum(s**2) / np.sum(s**2) < 0.95)
                )
            }
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error analyzing spectrum: {str(e)}")
            raise
            
    def check_stability(self, matrix: np.ndarray, eps: float = 1e-12) -> Tuple[bool, List[str]]:
        """
        Check numerical stability of matrix.
        
        Args:
            matrix: Matrix to check
            eps: Small constant for numerical stability
            
        Returns:
            Tuple of (is_stable, messages)
        """
        try:
            messages = []
            
            # Check for NaN/Inf values
            if not np.all(np.isfinite(matrix)):
                messages.append("Matrix contains NaN/Inf values")
                
            # Check conditioning
            if matrix.ndim == 2:
                cond = np.linalg.cond(matrix)
                if cond > 1e4:
                    messages.append(f"Matrix poorly conditioned: {cond:.2e}")
                    
            # Check rank deficiency
            if matrix.ndim == 2:
                rank = np.linalg.matrix_rank(matrix)
                if rank < min(matrix.shape):
                    messages.append(f"Matrix is rank deficient: {rank} < {min(matrix.shape)}")
                    
            # Check small values
            if np.any(np.abs(matrix) < eps):
                messages.append(f"Matrix contains values smaller than {eps}")
                
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error checking stability: {str(e)}")
            raise
            
    def project_to_stable(self, matrix: np.ndarray, max_cond: float = 1e4,
                         eps: float = 1e-12) -> np.ndarray:
        """
        Project matrix to numerically stable form.
        
        Args:
            matrix: Matrix to project
            max_cond: Maximum allowed condition number
            eps: Small constant for numerical stability
            
        Returns:
            Stabilized matrix
        """
        try:
            # Add small constant for stability
            matrix = matrix + eps
            
            # Condition matrix
            matrix = self.condition_matrix(matrix, max_cond)
            
            # Ensure non-negativity
            matrix = np.maximum(matrix, 0)
            
            # Normalize
            matrix = self.normalize_matrix(matrix)
            
            # Check stability
            is_stable, messages = self.check_stability(matrix)
            if not is_stable:
                self.logger.warning(f"Matrix still unstable after projection: {messages}")
                
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error projecting matrix: {str(e)}")
            raise