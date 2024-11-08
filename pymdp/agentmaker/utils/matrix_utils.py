import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MatrixUtils:
    """Utilities for handling active inference matrices"""
    
    @staticmethod
    def load_matrices(matrices_dir: Path, model_type: str) -> Dict:
        """Load and validate matrices for environment or agent"""
        try:
            matrices_dir = Path(matrices_dir) / model_type
            
            # Load matrices
            matrices = {
                'A': np.load(matrices_dir / "A_matrices.npy"),
                'B': np.load(matrices_dir / "B_matrices.npy"),
                'D': np.load(matrices_dir / "D_matrices.npy")
            }
            
            # Add C matrix for agent
            if model_type == "agent":
                matrices['C'] = np.load(matrices_dir / "C_matrices.npy")
                
            # Validate shapes
            expected_shapes = {
                'A': (3, 3),
                'B': (3, 3, 3),
                'D': (3,)
            }
            if model_type == "agent":
                expected_shapes['C'] = (3,)
                
            for name, matrix in matrices.items():
                if matrix.shape != expected_shapes[name]:
                    raise ValueError(f"Invalid shape for {name} matrix: {matrix.shape} (expected {expected_shapes[name]})")
                    
            logger.info(f"Loaded {model_type} matrices:")
            for name, matrix in matrices.items():
                logger.info(f"- {name}: shape {matrix.shape}")
                
            return matrices
            
        except Exception as e:
            logger.error(f"Error loading {model_type} matrices: {str(e)}")
            raise
            
    @staticmethod
    def normalize_matrices(matrices: Dict) -> Dict:
        """Normalize probability matrices"""
        normalized = matrices.copy()
        
        # Normalize A matrix (observation model)
        if 'A' in normalized:
            A = normalized['A']
            sums = A.sum(axis=0)
            if not np.allclose(sums, 1.0):
                logger.warning("Normalizing A matrix")
                normalized['A'] = A / sums[None, :]
                
        # Normalize B matrix (transition model)
        if 'B' in normalized:
            B = normalized['B']
            for action in range(B.shape[-1]):
                sums = B[..., action].sum(axis=0)
                if not np.allclose(sums, 1.0):
                    logger.warning(f"Normalizing B matrix for action {action}")
                    B[..., action] = B[..., action] / sums[None, :]
            normalized['B'] = B
                    
        # Normalize D matrix (initial state prior)
        if 'D' in normalized:
            D = normalized['D']
            if not np.allclose(D.sum(), 1.0):
                logger.warning("Normalizing D matrix")
                normalized['D'] = D / D.sum()
                
        return normalized 