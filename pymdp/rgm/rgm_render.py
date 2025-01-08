"""
RGM Matrix Renderer
==================

Renders matrices for RGM from GNN specifications.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Import from pymdp package
from pymdp.rgm.utils.rgm_experiment_utils import RGMExperimentUtils
from pymdp.rgm.utils.rgm_gnn_spec_loader import RGMGNNSpecLoader
from pymdp.rgm.utils.rgm_gnn_matrix_factory import RGMGNNMatrixFactory
from pymdp.rgm.utils.rgm_matrix_validator import RGMMatrixValidator
from pymdp.rgm.utils.rgm_matrix_visualization_utils import RGMMatrixVisualizationUtils

class RGMRenderer:
    """Renders matrices for RGM from GNN specifications"""
    
    def __init__(self):
        """Initialize renderer"""
        try:
            # Get experiment state
            self.experiment = RGMExperimentUtils.get_experiment()
            self.logger = RGMExperimentUtils.get_logger('renderer')
            self.logger.info("Initializing RGM renderer...")
            
            # Initialize components
            self.spec_loader = RGMGNNSpecLoader()
            self.matrix_factory = RGMGNNMatrixFactory()
            self.validator = RGMMatrixValidator()
            self.visualizer = RGMMatrixVisualizationUtils()
            
            # Create required directories
            self._create_directories()
            
            # Log initialization
            self.logger.info(f"Using experiment: {self.experiment['name']}")
            self.logger.info("RGM renderer initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize renderer: {str(e)}")
            raise
            
    def _create_directories(self):
        """Create required directories"""
        try:
            # Create validation directory if it doesn't exist
            validation_dir = self.experiment['dirs']['render'] / "validation"
            validation_dir.mkdir(exist_ok=True)
            
            # Store for error handling
            self.validation_dir = validation_dir
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {str(e)}")
            raise
            
    def render_matrices(self) -> Path:
        """
        Render all matrices from GNN specifications.
        
        Returns:
            Path to rendered matrices directory
        """
        try:
            self.logger.info("Starting matrix rendering...")
            start_time = time.time()
            
            # Load base GNN spec first
            self.logger.info("Loading base GNN spec...")
            base_spec = self._load_base_spec()
            
            # Load GNN specifications
            self.logger.info("Loading GNN specifications...")
            gnn_specs = self.spec_loader.load_gnn_specs()
            
            # Merge specifications
            merged_specs = self._merge_specs(base_spec, gnn_specs)
            
            # Generate matrices
            self.logger.info("Generating matrices...")
            matrices = self.matrix_factory.generate_matrices(merged_specs)
            
            # Validate matrices
            self.logger.info("Validating matrices...")
            self.validator.validate_matrices(matrices, merged_specs)
            
            # Save outputs
            matrices_dir = self._save_outputs(matrices, merged_specs)
            
            # Generate visualizations
            self.logger.info("Generating visualizations...")
            self.visualizer.generate_visualizations(matrices, matrices_dir)
            
            # Log matrix statistics
            self._log_matrix_statistics(matrices)
            
            duration = time.time() - start_time
            self.logger.info(f"Matrix rendering completed in {duration:.2f}s")
            
            return matrices_dir
            
        except Exception as e:
            self.logger.error(f"Error during matrix rendering: {str(e)}")
            self._save_error_state()
            raise
            
    def _load_base_spec(self) -> Dict:
        """Load base GNN specification"""
        try:
            self.logger.info("Loading GNN spec: rgm_base.gnn")
            base_path = Path(__file__).parent / "models" / "rgm_base.gnn"
            
            if not base_path.exists():
                raise FileNotFoundError("Base GNN spec not found")
                
            with open(base_path) as f:
                spec = json.load(f)
                
            return spec
            
        except Exception as e:
            self.logger.error(f"Error loading required spec rgm_base.gnn: {str(e)}")
            raise
            
    def _merge_specs(self, base_spec: Dict, additional_specs: Dict) -> Dict:
        """Merge GNN specifications"""
        try:
            merged = base_spec.copy()
            
            def deep_merge(d1: Dict, d2: Dict):
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                        
            for spec in additional_specs.values():
                deep_merge(merged, spec)
                
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging specifications: {str(e)}")
            raise
            
    def _save_outputs(self, matrices: Dict, gnn_specs: Dict) -> Path:
        """Save rendered matrices and configurations"""
        try:
            # Save matrices
            matrices_dir = self.experiment['dirs']['matrices']
            for name, matrix in matrices.items():
                np.save(matrices_dir / f"{name}.npy", matrix)
                self.logger.info(f"Saved {name} matrix: {matrix.shape}")
                
            # Save experiment configuration
            self.logger.info("Saving experiment configuration...")
            config_dir = self.experiment['dirs']['config']
            with open(config_dir / "experiment_config.json", 'w') as f:
                json.dump(gnn_specs, f, indent=2)
                
            # Save GNN specifications
            self.logger.info("Saving GNN specifications...")
            gnn_dir = self.experiment['dirs']['gnn_specs']
            for name, spec in gnn_specs.items():
                with open(gnn_dir / f"{name}.gnn", 'w') as f:
                    json.dump(spec, f, indent=2)
                    
            return matrices_dir
            
        except Exception as e:
            self.logger.error(f"Error saving outputs: {str(e)}")
            raise
            
    def _log_matrix_statistics(self, matrices: Dict):
        """Log statistics for all matrices"""
        self.logger.info("\nMatrix Statistics:")
        
        for name, matrix in matrices.items():
            stats = {
                'shape': matrix.shape,
                'min': float(np.min(matrix)),
                'max': float(np.max(matrix)),
                'mean': float(np.mean(matrix)),
                'std': float(np.std(matrix)),
                'sparsity': float(np.mean(matrix == 0)),
                'norm': float(np.linalg.norm(matrix))
            }
            
            if matrix.ndim == 2:
                stats['rank'] = int(np.linalg.matrix_rank(matrix))
                stats['condition'] = float(np.linalg.cond(matrix))
                
            self.logger.info(f"\n{name} Matrix:")
            for key, value in stats.items():
                self.logger.info(f"- {key}: {value}")
                
    def _save_error_state(self):
        """Save error state information"""
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_state': {
                    'last_processed_matrix': getattr(self, '_current_matrix', None),
                    'last_processed_spec': getattr(self, '_current_spec', None)
                }
            }
            
            error_path = self.validation_dir / "error_state.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving error state: {str(e)}")
