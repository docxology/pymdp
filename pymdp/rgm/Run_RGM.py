"""
RGM Pipeline Runner
=================

Main script for running the Renormalization Generative Model pipeline on MNIST.
"""

import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json
import numpy as np
import psutil

# Add parent directory to Python path for imports
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[1]
if str(top) not in sys.path:
    sys.path.insert(0, str(top))

# Core RGM components
from rgm.utils import (
    RGMLogging,
    RGMExperimentState,
    RGMExperimentUtils
)
from rgm.utils.custom_json_encoder import CustomJSONEncoder
from rgm.utils.rgm_model_state import RGMModelState
from rgm.utils.rgm_model_initializer import RGMModelInitializer

# Data handling
from rgm.mnist_download import MNISTPreprocessor, MNISTDataset
from rgm.utils.rgm_data_loader import RGMDataLoader

# Pipeline stages
from rgm.rgm_render import RGMRenderer
from rgm.training.rgm_trainer import RGMTrainer
from rgm.evaluation.rgm_evaluator import RGMEvaluator

class RGMPipeline:
    """
    Main RGM pipeline runner.
    
    Handles:
    - MNIST data preparation
    - GNN specification verification
    - Matrix generation and visualization
    - Model initialization and training
    - Model evaluation and analysis
    """
    
    def __init__(self, exp_dir: Optional[Path] = None):
        """
        Initialize pipeline.
        
        Args:
            exp_dir: Optional experiment directory path
        """
        # Setup logging and experiment state
        self.logger = RGMLogging.get_logger("rgm.runner")
        self.experiment = RGMExperimentState("rgm_mnist_pipeline", exp_dir)
        
        # Log initialization
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 Initializing Renormalization Generative Model Pipeline")
        self.logger.info(f"📂 Experiment directory: {self.experiment.exp_dir}")
        self.logger.info("="*80 + "\n")
        
        # Copy GNN specifications
        gnn_source_dir = RGMExperimentUtils.get_gnn_dir()
        gnn_target_dir = self.experiment.get_dir('gnn_specs')
        RGMExperimentUtils.copy_gnn_files(gnn_source_dir, gnn_target_dir)
        
        # Setup device and model state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🖥️  Using device: {self.device}")
        self.model_state = None
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load pipeline configuration."""
        config_path = Path(__file__).parent / "configs" / "mnist_config.json"
        with open(config_path) as f:
            return json.load(f)

    def train_model(self, mnist_data: MNISTDataset):
        """
        Train the RGM model.
        
        Args:
            mnist_data: MNIST dataset container with data loaders
        """
        try:
            self.logger.info("🎯 Starting model training...")
            
            # Log training configuration
            train_config = self.config["training"]
            self.logger.info("\n📋 Training Configuration:")
            for key, value in train_config.items():
                self.logger.info(f"   - {key}: {value}")
            
            # Log dataset sizes
            self.logger.info("\n📊 Dataset Information:")
            self.logger.info(f"   - Training samples:   {len(mnist_data.train_loader.dataset):,}")
            self.logger.info(f"   - Validation samples: {len(mnist_data.val_loader.dataset):,}")
            self.logger.info(f"   - Test samples:       {len(mnist_data.test_loader.dataset):,}")
            self.logger.info(f"   - Batch size:         {train_config['batch_size']}")
            
            # Initialize trainer
            trainer = RGMTrainer(
                model_state=self.model_state,
                train_loader=mnist_data.train_loader,
                val_loader=mnist_data.val_loader,
                config=train_config,
                exp_dir=self.experiment.exp_dir,
                device=self.device
            )
            
            # Train model
            trainer.train()
            
            self.logger.info("✅ Model training complete")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise

    def run(self):
        """Run complete pipeline."""
        try:
            # Stage 1: Data Preparation
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 Stage 1: Data Preparation")
            self.logger.info("-"*80)
            mnist_data = self.prepare_data()
            self.logger.info("✅ MNIST data downloaded and verified")
            
            # Stage 2: GNN Specification Verification
            self.logger.info("\n" + "="*80)
            self.logger.info("🔍 Stage 2: GNN Specification Verification")
            self.logger.info("-"*80)
            gnn_dir = self.experiment.get_dir('gnn_specs')
            self.logger.info(f"GNN files found in: {gnn_dir}")
            self.logger.info("✅ GNN files verified")
            
            # Stage 3: Matrix Generation
            self.logger.info("\n" + "="*80)
            self.logger.info("🔢 Stage 3: Matrix Generation")
            self.logger.info("-"*80)
            renderer = RGMRenderer(self.experiment.exp_dir)
            matrices = renderer.render_matrices()
            
            # Stage 4: Model Initialization
            self.logger.info("\n" + "="*80)
            self.logger.info("🔧 Stage 4: Model Initialization")
            self.logger.info("-"*80)
            initializer = RGMModelInitializer(self.experiment.exp_dir, device=self.device)
            matrices = initializer.load_matrices()
            self.model_state = RGMModelState(matrices, device=self.device)
            self.logger.info(f"✅ Model initialized with {len(matrices)} matrices")
            
            # Stage 5: Training
            self.logger.info("\n" + "="*80)
            self.logger.info("🏃 Stage 5: Model Training")
            self.logger.info("-"*80)
            self.train_model(mnist_data)
            
            # Stage 6: Evaluation
            self.logger.info("\n" + "="*80)
            self.logger.info("📈 Stage 6: Model Evaluation")
            self.logger.info("-"*80)
            evaluator = RGMEvaluator(
                model=self.model_state,
                test_loader=mnist_data.test_loader,
                exp_dir=self.experiment.exp_dir,
                device=self.device
            )
            evaluator.evaluate()
            
            # Pipeline Complete
            self.logger.info("\n" + "="*80)
            self.logger.info("🎉 Pipeline Execution Complete!")
            self.logger.info("="*80 + "\n")
            
        except Exception as e:
            self.logger.error("\n" + "="*80)
            self.logger.error("❌ Pipeline Execution Failed!")
            self.logger.error(f"Error: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            self._save_error_state()
            self.logger.error("="*80 + "\n")
            raise
            
    def _save_error_state(self):
        """Save error state for debugging."""
        try:
            error_dir = self.experiment.get_dir('simulation')
            error_state = {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'memory_usage': psutil.Process().memory_info().rss / 1024**2,
                'model_state': self.model_state.state_dict() if self.model_state else None
            }
            
            error_path = error_dir / 'error_state.json'
            with open(error_path, 'w') as f:
                json.dump(error_state, f, cls=CustomJSONEncoder, indent=2)
            self.logger.info(f"💾 Error state saved to: {error_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save error state: {str(e)}")

    def prepare_data(self) -> MNISTDataset:
        """
        Prepare MNIST dataset for training.
        
        This method:
        1. Downloads MNIST if not present
        2. Creates train/val/test splits
        3. Applies normalization
        4. Creates data loaders
        
        Returns:
            MNISTDataset: Container with data loaders and datasets
        """
        try:
            self.logger.info("Preparing MNIST data...")
            
            # Get data configuration
            data_config = self.config["data"]
            
            # Initialize preprocessor with config
            data_dir = self.experiment.get_dir('data')
            preprocessor = MNISTPreprocessor(
                data_dir=data_dir,
                batch_size=data_config["batch_size"]
            )
            
            # Prepare datasets and loaders
            mnist_data = preprocessor.prepare_data()
            
            # Log dataset information
            self.logger.info("\n📊 Dataset Statistics:")
            self.logger.info(f"   • Training samples:   {len(mnist_data.train_loader.dataset):,}")
            self.logger.info(f"   • Validation samples: {len(mnist_data.val_loader.dataset):,}")
            self.logger.info(f"   • Test samples:       {len(mnist_data.test_loader.dataset):,}")
            self.logger.info(f"   • Batch size:         {data_config['batch_size']}")
            self.logger.info(f"   • Workers:            {data_config['num_workers']}")
            self.logger.info(f"   • Pin memory:         {data_config['pin_memory']}")
            
            # Log data directory
            self.logger.info(f"\n💾 Data stored in: {data_dir}")
            
            return mnist_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare data: {str(e)}")
            raise

# Main execution
if __name__ == "__main__":
    try:
        pipeline = RGMPipeline()
        pipeline.run()
    except Exception as e:
        logging.error(f"❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)
