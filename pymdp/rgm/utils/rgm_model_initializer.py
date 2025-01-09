"""
RGM Model Initializer
==================

Handles initialization of the RGM model from saved matrices and specifications.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from .rgm_logging import RGMLogging

class RGMModelInitializer:
    """Initializes RGM model components."""
    
    def __init__(self, exp_dir: Path, device: Optional[torch.device] = None):
        """
        Initialize model loader.
        
        Args:
            exp_dir: Experiment directory containing matrices
            device: Torch device to load matrices to
        """
        self.logger = RGMLogging.get_logger("rgm.model_initializer")
        self.exp_dir = Path(exp_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Load saved matrices from experiment directory.
        
        Returns:
            Dictionary of loaded matrices
        """
        try:
            matrices_dir = self.exp_dir / "matrices"
            if not matrices_dir.exists():
                raise FileNotFoundError(f"Matrices directory not found: {matrices_dir}")
                
            matrices = {}
            for matrix_file in matrices_dir.glob("*.npy"):
                name = matrix_file.stem
                matrix = torch.from_numpy(np.load(matrix_file)).to(self.device)
                matrices[name] = matrix
                
            self._validate_loaded_matrices(matrices)
            
            self.logger.info(f"Loaded {len(matrices)} matrices from {matrices_dir}")
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error loading matrices: {str(e)}")
            raise
            
    def _validate_loaded_matrices(self, matrices: Dict[str, torch.Tensor]):
        """
        Validate loaded matrices.
        
        Args:
            matrices: Dictionary of loaded matrices
            
        Raises:
            ValueError: If required matrices are missing or invalid
        """
        # Check required matrices
        required_prefixes = ['R', 'G', 'L']  # Recognition, Generative, Lateral
        for prefix in required_prefixes:
            for level in range(3):
                name = f"{prefix}{level}"
                if name not in matrices:
                    raise ValueError(f"Missing required matrix: {name}")
                    
        # Validate matrix relationships
        for level in range(3):
            # Recognition and generative matrices should be transposes
            r_mat = matrices[f"R{level}"]
            g_mat = matrices[f"G{level}"]
            if r_mat.shape != g_mat.t().shape:
                raise ValueError(
                    f"Dimension mismatch between R{level} and G{level}: "
                    f"{r_mat.shape} vs {g_mat.t().shape}"
                )
                
            # Lateral matrices should be square
            l_mat = matrices[f"L{level}"]
            if l_mat.shape[0] != l_mat.shape[1]:
                raise ValueError(
                    f"Lateral matrix L{level} not square: {l_mat.shape}"
                ) 