"""
RGM Matrix Renderer
=================

Renders matrices from GNN specifications for the Renormalization Generative Model.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

from rgm.utils import RGMLogging
from rgm.utils.rgm_gnn_loader import RGMGNNLoader
from rgm.utils.matrix_utils import orthogonalize_matrix, ensure_positive_definite

class RGMRenderer:
    """Renders matrices from GNN specifications."""
    
    def __init__(self, exp_dir: Path):
        """
        Initialize renderer.
        
        Args:
            exp_dir: Experiment directory containing GNN specs
        """
        self.logger = RGMLogging.get_logger("rgm.renderer")
        self.gnn_loader = RGMGNNLoader()
        self.exp_dir = Path(exp_dir)  # Ensure Path object
        
    def render_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Render matrices from GNN specifications.
        
        The process follows these steps:
        1. Load GNN specifications
        2. Validate matrix dimensions
        3. Initialize matrices with specified distributions
        4. Apply constraints (orthogonality, symmetry)
        5. Save visualizations
        
        The matrices form a hierarchical generative model that minimizes variational free energy:
        - Recognition matrices (A): Encode beliefs about hidden causes
        - Generative matrices (B): Decode predictions about observations
        - Lateral matrices (D): Encode precision-weighted prediction errors
        
        Returns:
            Dictionary of rendered matrices
        """
        try:
            # Load and validate specifications
            gnn_dir = self.exp_dir / "gnn_specs"
            if not gnn_dir.exists():
                raise FileNotFoundError(f"GNN specification directory not found: {gnn_dir}")
                
            specs = self.gnn_loader.load_specifications(gnn_dir)  # Use new method name
            self.gnn_loader.validate_specifications(specs)
            
            # Extract matrix specifications
            matrices = {}
            for spec in specs:
                if 'matrices' in spec:
                    for matrix_type, matrix_specs in spec['matrices'].items():
                        for name, shape in matrix_specs.items():
                            # Initialize matrix with FEP principles
                            matrix = self._initialize_matrix(
                                name, shape, 
                                spec['initialization']
                            )
                            matrices[name] = matrix
            
            # Apply FEP-based constraints
            matrices = self._apply_constraints(matrices, specs)
            
            # Save matrices and visualizations
            self._save_matrices(matrices)
            self._generate_visualizations(matrices)
            
            self.logger.info(f"Generated matrices: {list(matrices.keys())}")
            return matrices
            
        except Exception as e:
            self.logger.error(f"Error rendering matrices: {str(e)}")
            raise
            
    def _initialize_matrix(self, name: str, shape: list, init_config: Dict) -> torch.Tensor:
        """Initialize matrix with specified distribution."""
        method = init_config['method']
        
        if method == 'orthogonal':
            matrix = torch.randn(shape)
            matrix = orthogonalize_matrix(matrix)
            
        elif method == 'normal':
            mean = init_config.get('mean', 0.0)
            std = init_config.get('std', 0.01)
            matrix = torch.randn(shape) * std + mean
            
        elif method == 'uniform':
            a = init_config.get('min', -0.1)
            b = init_config.get('max', 0.1)
            matrix = torch.rand(shape) * (b - a) + a
            
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        return matrix
        
    def _apply_constraints(self, matrices: Dict[str, torch.Tensor], 
                         specs: list) -> Dict[str, torch.Tensor]:
        """Apply constraints to matrices."""
        for spec in specs:
            if 'constraints' not in spec:
                continue
                
            constraints = spec['constraints']
            
            # Apply normalization
            if 'normalize' in constraints:
                for name, matrix in matrices.items():
                    if name.startswith('A') and constraints['normalize'] == 'row':
                        matrices[name] = torch.nn.functional.normalize(matrix, dim=1)
                    elif name.startswith('B') and constraints['normalize'] == 'column':
                        matrices[name] = torch.nn.functional.normalize(matrix, dim=0)
            
            # Apply symmetry constraints
            if constraints.get('symmetric', False):
                for name, matrix in matrices.items():
                    if name.startswith('D'):
                        matrices[name] = 0.5 * (matrix + matrix.t())
                        
            # Ensure positive definiteness for D matrices
            if constraints.get('positive', False):
                for name, matrix in matrices.items():
                    if name.startswith('D'):
                        matrices[name] = ensure_positive_definite(matrix)
                        
        return matrices
        
    def _save_matrices(self, matrices: Dict[str, torch.Tensor]):
        """Save matrices to files."""
        save_dir = self.exp_dir / "matrices"
        save_dir.mkdir(exist_ok=True)
        
        for name, matrix in matrices.items():
            torch.save(matrix, save_dir / f"{name}.pt")
            
        self.logger.info(f"Saved matrices to: {save_dir}")
        
    def _generate_visualizations(self, matrices: Dict[str, torch.Tensor]):
        """Generate visualizations of matrices."""
        vis_dir = self.exp_dir / "matrices" / "visualizations"
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        for name, matrix in matrices.items():
            plt.figure(figsize=(10, 10))
            plt.imshow(matrix.numpy(), cmap='RdBu', aspect='auto')
            plt.colorbar()
            plt.title(f"Matrix {name}")
            plt.savefig(vis_dir / f"{name}.png")
            plt.close()
            
        self.logger.info(f"Generated matrix visualizations in: {vis_dir}") 