"""
RGM Matrix Renderer
=================

Renders and visualizes matrices from GNN specifications.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

from .utils.rgm_logging import RGMLogging
from .utils.visualization_utils import RGMVisualizationUtils

class RGMRenderer:
    """Renders matrices from GNN specifications."""
    
    def __init__(self, exp_dir: Path):
        self.logger = RGMLogging.get_logger("rgm.renderer")
        self.exp_dir = exp_dir
        self.vis_dir = exp_dir / "visualizations"
        self.matrices_dir = exp_dir / "matrices"
        
        # Create directories
        self.vis_dir.mkdir(exist_ok=True, parents=True)
        self.matrices_dir.mkdir(exist_ok=True, parents=True)
        
    def render_matrices(self) -> Dict[str, torch.Tensor]:
        """Render matrices from GNN specifications."""
        self.logger.info("\n🎨 Stage 3: Matrix Generation")
        self.logger.info("-"*80)
        self.logger.info("Rendering matrices from GNN specifications...")
        
        # Generate matrices
        matrices = self._generate_matrices()
        
        # Log matrix information
        self.logger.info("\n📊 Generated Matrices:")
        for name, matrix in matrices.items():
            self.logger.info(
                f"   • {name:<4}: Shape {matrix.shape!s:<12} "
                f"[{matrix.dtype}] "
                f"Range: [{matrix.min():.3f}, {matrix.max():.3f}]"
            )
        
        # Save matrices
        self._save_matrices(matrices)
        self.logger.info(f"\n💾 Matrices saved to: {self.matrices_dir}")
        
        # Generate visualizations
        self._visualize_matrices(matrices)
        
        # Log visualization details
        self.logger.info("\n🎨 Matrix Visualizations:")
        self.logger.info(f"   Base directory: {self.vis_dir}")
        self.logger.info("\n   Individual Matrix Plots:")
        for name in matrices:
            path = self.vis_dir / f"{name}_matrix.png"
            self.logger.info(f"   • {name:<4}: {path}")
        
        self.logger.info("\n   Combined Visualizations:")
        self.logger.info(f"   • Grid:       {self.vis_dir / 'all_matrices_grid.png'}")
        self.logger.info(f"   • Hierarchy:  {self.vis_dir / 'hierarchy_visualization.png'}")
        
        return matrices
        
    def _visualize_matrices(self, matrices: Dict[str, torch.Tensor]):
        """Generate matrix visualizations."""
        self.logger.info("\nGenerating visualizations...")
        
        # Individual matrix plots
        for name, matrix in matrices.items():
            fig_path = self.vis_dir / f"{name}_matrix.png"
            RGMVisualizationUtils.plot_matrix(
                matrix.cpu().numpy(),
                title=f"Matrix {name}",
                save_path=fig_path
            )
            self.logger.debug(f"   ↳ Generated {name} plot: {fig_path.name}")
        
        # Grid visualization
        grid_path = self.vis_dir / "all_matrices_grid.png"
        RGMVisualizationUtils.plot_matrix_grid(
            [m.cpu().numpy() for m in matrices.values()],
            titles=list(matrices.keys()),
            save_path=grid_path
        )
        self.logger.debug(f"   ↳ Generated matrix grid: {grid_path.name}")
        
        # Hierarchical visualization
        hierarchy_path = self.vis_dir / "hierarchy_visualization.png"
        self._visualize_hierarchy(matrices, hierarchy_path)
        self.logger.debug(f"   ↳ Generated hierarchy plot: {hierarchy_path.name}") 