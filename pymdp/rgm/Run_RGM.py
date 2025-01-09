"""
RGM Pipeline Runner
=================

Main script for running the RGM (Recursive Generative Model) pipeline on MNIST.

Pipeline Stages:
1. Data Preparation
   - Download and preprocess MNIST
   - Create balanced training set
   - Generate exemplars

2. Model Initialization
   - Load GNN specifications
   - Generate connectivity matrices
   - Initialize model parameters

3. Training
   - Forward/backward passes
   - Parameter updates
   - State tracking

4. Evaluation
   - Generation quality
   - Reconstruction accuracy
   - State analysis
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Add parent directory to Python path for imports
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[1]
if str(top) not in sys.path:
    sys.path.insert(0, str(top))

from rgm.utils import RGMLogging, RGMExperimentState
from rgm.mnist_download import MNISTPreprocessor
from rgm.rgm_render import RGMRenderer
from rgm.rgm_execute import RGMExecutor

class RGMPipeline:
    """Main RGM pipeline runner."""
    
    def __init__(self, exp_dir: Optional[Path] = None):
        """Initialize pipeline."""
        # Setup logging
        self.logger = RGMLogging.get_logger("rgm.runner")
        
        # Create experiment state
        exp_name = "rgm_mnist_pipeline"
        if exp_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = Path("experiments") / f"{exp_name}_{timestamp}"
        
        self.logger.info(f"Initializing RGM pipeline in: {str(exp_dir)}")
        self.exp_state = RGMExperimentState(exp_dir)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def run(self) -> None:
        """Run the complete RGM pipeline."""
        try:
            self.logger.info("\n" + "="*50)
            self.logger.info("Starting RGM MNIST Pipeline")
            self.logger.info("="*50)
            
            # Stage 1: Data Preparation
            self.logger.info("\nStage 1: Data Preparation")
            self.logger.info("-"*30)
            mnist_data = self.prepare_mnist_data()
            
            # Stage 2: Model Initialization
            self.logger.info("\nStage 2: Model Initialization")
            self.logger.info("-"*30)
            matrices = self.initialize_model()
            
            # Stage 3: Training
            self.logger.info("\nStage 3: Training")
            self.logger.info("-"*30)
            self.train_model(mnist_data, matrices)
            
            # Stage 4: Evaluation
            self.logger.info("\nStage 4: Evaluation")
            self.logger.info("-"*30)
            self.evaluate_model(mnist_data)
            
            # Stage 5: Inference
            self.logger.info("\nStage 5: Inference")
            self.logger.info("-"*30)
            executor = RGMExecutor(matrices)
            executor.run_inference(mnist_data)
            
            self.logger.info("\n" + "="*50)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise

    def prepare_mnist_data(self) -> Dict:
        """Prepare MNIST dataset."""
        try:
            self.logger.info("Preparing MNIST dataset...")
            
            # Initialize preprocessor with experiment state
            data_dir = self.exp_state.get_dir("data")
            preprocessor = MNISTPreprocessor(
                data_dir=data_dir,
                exp_state=self.exp_state
            )
            
            # Prepare datasets with default settings
            mnist_data = preprocessor.prepare_datasets()
            
            self.logger.info("MNIST data preparation complete")
            return mnist_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare MNIST dataset: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise

    def initialize_model(self) -> Dict:
        """Initialize RGM model."""
        try:
            self.logger.info("Initializing RGM model...")
            
            # Initialize renderer
            renderer = RGMRenderer(self.exp_state)
            
            # Generate matrices
            matrices_dir = renderer.render_matrices()
            self.logger.info(f"Generated matrices in: {str(matrices_dir)}")
            
            # Load and validate matrices
            matrices = self._load_matrices(matrices_dir)
            self._validate_matrices(matrices)
            
            self.logger.info("Model initialization complete")
            return matrices
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise

    def train_model(self, mnist_data: Dict, matrices: Dict) -> None:
        """Train RGM model."""
        try:
            self.logger.info("Training RGM model...")
            # TODO: Implement training loop
            self.logger.info("Training complete")
            
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