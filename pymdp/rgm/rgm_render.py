"""
RGM Matrix Renderer
=================

Renders matrices from GNN specifications for the Renormalization Generative Model.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict

from rgm.utils import RGMLogging
from rgm.utils.rgm_gnn_loader import RGMGNNLoader
from rgm.utils.rgm_matrix_visualization_utils import RGMMatrixVisualizationUtils

class RGMRenderer:
    """Renders matrices for the Renormalization Generative Model."""
    
    def __init__(self, exp_dir: Path):
        """Initialize renderer."""
        self.logger = RGMLogging.get_logger("rgm.renderer")
        self.gnn_loader = RGMGNNLoader()
        self.exp_dir = Path(exp_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def render_matrices(self) -> Dict[str, torch.Tensor]:
        """Generate matrices from GNN specifications."""
        try:
            # Load and validate GNN specifications
            gnn_dir = self.exp_dir / "gnn_specs"
            if not gnn_dir.exists():
                raise FileNotFoundError(f"GNN specification directory not found: {gnn_dir}")
                
            # Load merged specification
            spec = self.gnn_loader.load_specifications(gnn_dir)
            
            # Generate matrices
            matrices = self._generate_matrices(spec['matrices'])
            
            # Save matrices and visualizations
            self._save_matrices(matrices)
            self._generate_visualizations(matrices)
            
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error rendering matrices: {str(e)}")
            raise
            
    def _generate_matrices(self, matrix_specs: Dict) -> Dict[str, torch.Tensor]:
        """Generate matrices from specifications."""
        matrices = {}
        
        for matrix_type, specs in matrix_specs.items():
            for name, shape in specs.items():
                # Initialize matrix
                matrix = torch.empty(shape, device=self.device)
                torch.nn.init.xavier_uniform_(matrix)
                
                # Apply constraints based on matrix type
                if matrix_type == 'lateral':
                    # Make symmetric
                    matrix = 0.5 * (matrix + matrix.t())
                    
                matrices[name] = matrix
                
        return matrices
        
    def _save_matrices(self, matrices: Dict[str, torch.Tensor]):
        """
        Save matrices to files.
        
        Note: Matrices are saved with legacy naming (A/B/D) for backward compatibility.
        The model initializer will convert these to standardized names (R/G/L) during loading.
        """
        matrices_dir = self.exp_dir / "matrices"
        matrices_dir.mkdir(exist_ok=True)
        
        for name, matrix in matrices.items():
            file_path = matrices_dir / f"{name}.npy"
            np.save(file_path, matrix.cpu().numpy())
            
    def _generate_visualizations(self, matrices: Dict[str, torch.Tensor]):
        """Generate visualizations of matrices."""
        try:
            vis_dir = self.exp_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # Convert tensors to numpy arrays
            numpy_matrices = {
                name: matrix.cpu().numpy() 
                for name, matrix in matrices.items()
            }
            
            # Generate individual matrix plots
            for name, matrix in numpy_matrices.items():
                # Basic heatmap
                RGMMatrixVisualizationUtils.plot_matrix(
                    matrix,
                    title=f"Matrix {name}",
                    save_path=vis_dir / f"{name}_heatmap.png"
                )
                
                # Detailed analysis
                RGMMatrixVisualizationUtils.plot_matrix_analysis(
                    matrix,
                    title=f"Matrix {name}",
                    save_path=vis_dir / f"{name}_analysis.png"
                )
            
            # Generate grid plot of all matrices
            RGMMatrixVisualizationUtils.plot_matrix_grid(
                numpy_matrices,
                save_path=vis_dir / "all_matrices_grid.png"
            )
            
            self.logger.info(f"Generated matrix visualizations in: {vis_dir}")
            
        except Exception as e:
            self.logger.warning(f"Error generating matrix visualizations: {str(e)}")
            # Continue execution even if visualization fails
