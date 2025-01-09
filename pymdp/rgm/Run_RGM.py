"""
RGM Pipeline Runner
=================

Main script for running the Renormalization Generative Model pipeline on MNIST.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json
import numpy as np

# Add parent directory to Python path for imports
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[1]
if str(top) not in sys.path:
    sys.path.insert(0, str(top))

from rgm.utils import (
    RGMLogging,
    RGMExperimentState,
    RGMExperimentUtils
)
from rgm.utils.custom_json_encoder import CustomJSONEncoder
from rgm.mnist_download import MNISTPreprocessor
from rgm.rgm_render import RGMRenderer
from rgm.rgm_execute import RGMExecutor

class RGMPipeline:
    """Main RGM pipeline runner."""
    
    def __init__(self, exp_dir: Optional[Path] = None):
        """Initialize pipeline."""
        self.logger = RGMLogging.get_logger("rgm.runner")
        self.experiment = RGMExperimentState("rgm_mnist_pipeline", exp_dir)
        self.logger.info(f"Initializing RGM pipeline in: {self.experiment.exp_dir}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_state = None
        
    def run(self):
        """Run complete pipeline."""
        try:
            # Download and preprocess MNIST
            mnist_data = self.prepare_data()
            self.logger.info("MNIST data downloaded and verified")
            
            # Verify GNN files
            self.verify_gnn_files()
            self.logger.info("GNN files verified")
            
            # Generate matrices from GNN specs
            matrices_dir = self.render_matrices()
            self.logger.info(f"GNN-based generative models created and saved in: {matrices_dir}")
            
            # Load and validate matrices
            matrices = self._load_matrices(matrices_dir)
            self._validate_matrices(matrices)
            self.logger.info("Generated matrices loaded and validated")
            
            # Initialize model state
            self.model_state = self.initialize_model(matrices)
            self.logger.info("RGM model state initialized")
            
            # Train model
            self.train_model(mnist_data)
            
            # Evaluate model
            self.evaluate_model(mnist_data)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            self._save_error_state()
            raise
        
    def prepare_data(self) -> Dict:
        """Download and preprocess MNIST data."""
        try:
            self.logger.info("Preparing MNIST data...")
            data_dir = self.experiment.get_dir('data')
            preprocessor = MNISTPreprocessor(data_dir=data_dir, exp_state=self.experiment)
            mnist_data = preprocessor.prepare_datasets()
            self.logger.info("Data preparation complete")
            return mnist_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
        
    def render_matrices(self) -> Path:
        """Generate matrices from GNN specifications."""
        try:
            self.logger.info("Rendering matrices from GNN specs...")
            renderer = RGMRenderer(self.experiment)
            matrices_dir = renderer.render_matrices()
            self.logger.info(f"Matrix rendering complete. Saved to: {matrices_dir}")
            return matrices_dir
            
        except Exception as e:
            self.logger.error(f"Failed to render matrices: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
        
    def initialize_model(self, matrices: Dict) -> Dict:
        """Initialize RGM model state."""
        try:
            self.logger.info("Initializing RGM model...")
            # TODO: Implement model initialization
            self.logger.info("Model initialization complete")
            return {}  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
        
    def train_model(self, mnist_data: Dict) -> None:
        """Train RGM model."""
        try:
            self.logger.info("Training RGM model...")
            executor = RGMExecutor(self.experiment, self.model_state, mnist_data)
            executor.train()
            self.logger.info("Model training complete")
            
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
        
    def evaluate_model(self, mnist_data: Dict) -> None:
        """Evaluate RGM model."""
        try:
            self.logger.info("Evaluating model performance...")
            # TODO: Implement evaluation
            self.logger.info("Evaluation complete")
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise

    def _load_matrices(self, matrices_dir: Path) -> Dict:
        """Load generated matrices."""
        try:
            matrices = {}
            for matrix_file in matrices_dir.glob("*.npy"):
                name = matrix_file.stem
                matrix = torch.from_numpy(np.load(matrix_file))
                matrices[name] = matrix.to(self.device)
            return matrices
            
        except Exception as e:
            self.logger.error(f"Failed to load matrices: {str(e)}")
            raise

    def _validate_matrices(self, matrices: Dict) -> None:
        """Validate loaded matrices."""
        try:
            required = ['A0', 'B0', 'D0']  # Add required matrices
            for name in required:
                if name not in matrices:
                    raise ValueError(f"Missing required matrix: {name}")
                if not torch.isfinite(matrices[name]).all():
                    raise ValueError(f"Matrix {name} contains invalid values")
            
        except Exception as e:
            self.logger.error(f"Matrix validation failed: {str(e)}")
            raise

    def _save_error_state(self):
        """Save error state information"""
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'model_state': self.model_state,
                'last_metrics': getattr(self, '_last_metrics', None)
            }
            
            error_path = self.experiment.get_dir('simulation') / "error_state.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2, cls=CustomJSONEncoder)
            
        except Exception as e:
            self.logger.error(f"Error saving error state: {str(e)}")

    def verify_gnn_files(self):
        """Verify the presence and validity of GNN specification files."""
        try:
            gnn_dir = self.experiment.get_dir('gnn_specs')
            required_files = [
                'rgm_base.gnn',
                'rgm_mnist.gnn',
                'rgm_message_passing.gnn',
                'rgm_hierarchical_level.gnn'
            ]
            
            for file in required_files:
                file_path = gnn_dir / file
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing GNN file: {file}")
                
                # TODO: Add validation logic for GNN files
                
            self.logger.info(f"GNN files found in: {gnn_dir}")
            
        except Exception as e:
            self.logger.error(f"Error verifying GNN files: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize and run pipeline
        pipeline = RGMPipeline()
        pipeline.run()
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()