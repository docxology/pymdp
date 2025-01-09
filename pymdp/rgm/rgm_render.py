"""
RGM Matrix Renderer
=================

Renders matrices for RGM pipeline by generating connectivity matrices from GNN specifications.

This module is responsible for:
1. Loading GNN Specifications
   - Base specifications (rgm_base.gnn)
   - Level-specific specifications
   - MNIST-specific configurations

2. Matrix Generation
   - Creating connectivity matrices
   - Applying sparsity patterns
   - Initializing weights
   - Validating dimensions

3. Matrix Storage
   - Saving matrices to experiment directory
   - Organizing by hierarchy level
   - Maintaining matrix metadata

The generated matrices define the connectivity structure of the RGM model:
- A matrices: Forward connections (state to factor)
- B matrices: Backward connections (factor to state)
- D matrices: Factor priors
- E matrices: State priors

Component Dependencies:
- RGMExperimentState: Manages experiment directories
- RGMGNNMatrixFactory: Generates matrices from specs
- RGMGNNSpecLoader: Loads and merges specifications
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to Python path for imports
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[1]
if str(top) not in sys.path:
    sys.path.insert(0, str(top))

from rgm.utils.rgm_logging import RGMLogging
from rgm.utils.rgm_experiment_utils import RGMExperimentUtils
from rgm.utils.rgm_experiment_state import RGMExperimentState
from rgm.utils.rgm_gnn_matrix_factory import RGMGNNMatrixFactory
from rgm.utils.rgm_gnn_spec_loader import RGMGNNSpecLoader

class RGMRenderer:
    """
    Renders matrices for RGM pipeline.
    
    This class handles the complete matrix generation process:
    1. Loading and validating GNN specifications
    2. Generating matrices with proper dimensions
    3. Saving matrices to experiment directory
    
    The renderer ensures that all matrices are properly initialized
    and validated before being used in the training process.
    """
    
    def __init__(self, exp_state: RGMExperimentState):
        """
        Initialize renderer.
        
        Args:
            exp_state: Experiment state manager that provides:
                     - Directory management
                     - Logging configuration
                     - State persistence
        """
        self.exp_state = exp_state
        self.logger = RGMLogging.get_logger("rgm.renderer")
        self.matrix_factory = RGMGNNMatrixFactory()
        self.spec_loader = RGMGNNSpecLoader()
        
    def render_matrices(self) -> Path:
        """
        Render matrices from GNN specifications.
        
        This method:
        1. Loads GNN specifications from config files
        2. Generates matrices using the factory
        3. Saves matrices to experiment directory
        
        Returns:
            Path to directory containing rendered matrices
            
        Raises:
            RuntimeError: If matrix generation fails
            ValueError: If specifications are invalid
        """
        try:
            self.logger.info("Loading GNN specifications...")
            gnn_specs = self.spec_loader.load_specs()
            
            self.logger.info("Generating matrices...")
            matrices = self.matrix_factory.generate_matrices(gnn_specs)
            
            # Save matrices
            matrices_dir = self.exp_state.get_dir("matrices")
            self._save_matrices(matrices, matrices_dir)
            
            return matrices_dir
            
        except Exception as e:
            self.logger.error(f"Error rendering matrices: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
            
    def _save_matrices(self, matrices: Dict[str, np.ndarray], output_dir: Path):
        """
        Save matrices to output directory.
        
        Args:
            matrices: Dictionary mapping matrix names to numpy arrays
            output_dir: Directory to save matrices in
            
        The matrices are saved in numpy format with names matching
        their role in the hierarchy (e.g., 'A0.npy', 'B0.npy').
        """
        try:
            self.logger.info(f"Saving matrices to: {output_dir}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each matrix
            for name, matrix in matrices.items():
                matrix_file = output_dir / f"{name}.npy"
                np.save(matrix_file, matrix)
                self.logger.debug(
                    f"Saved matrix {name} with shape {matrix.shape} to {matrix_file}"
                )
                
        except Exception as e:
            self.logger.error(f"Error saving matrices: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
