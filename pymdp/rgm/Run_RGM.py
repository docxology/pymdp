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

from rgm.utils import (
    RGMLogging,
    RGMExperimentState,
    RGMExperimentUtils
)
from rgm.utils.custom_json_encoder import CustomJSONEncoder
from rgm.mnist_download import MNISTPreprocessor
from rgm.rgm_render import RGMRenderer
from rgm.training.rgm_trainer import RGMTrainer
from rgm.evaluation.rgm_evaluator import RGMEvaluator
from rgm.utils.rgm_model_state import RGMModelState
from rgm.utils.rgm_data_loader import RGMDataLoader
from rgm.utils.rgm_model_initializer import RGMModelInitializer

class RGMPipeline:
    """Main RGM pipeline runner."""
    
    def __init__(self, exp_dir: Optional[Path] = None):
        """Initialize pipeline."""
        self.logger = RGMLogging.get_logger("rgm.runner")
        self.experiment = RGMExperimentState("rgm_mnist_pipeline", exp_dir)
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 Initializing Renormalization Generative Model Pipeline")
        self.logger.info(f"📂 Experiment directory: {self.experiment.exp_dir}")
        self.logger.info("="*80 + "\n")
        
        # Copy GNN files to experiment directory
        gnn_source_dir = RGMExperimentUtils.get_gnn_dir()
        gnn_target_dir = self.experiment.get_dir('gnn_specs')
        RGMExperimentUtils.copy_gnn_files(gnn_source_dir, gnn_target_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🖥️  Using device: {self.device}")
        self.model_state = None
        
    def run(self):
        """Run complete pipeline."""
        try:
            # Stage 1: Data Preparation
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 Stage 1: Data Preparation")
            self.logger.info("-"*80)
            mnist_data = self.prepare_data()
            self.logger.info("✅ MNIST data downloaded and verified")
            
            # Stage 2: GNN Verification
            self.logger.info("\n" + "="*80)
            self.logger.info("🔍 Stage 2: GNN Specification Verification")
            self.logger.info("-"*80)
            self.verify_gnn_files()
            self.logger.info("✅ GNN files verified")
            
            # Stage 3: Matrix Generation
            self.logger.info("\n" + "="*80)
            self.logger.info("🔢 Stage 3: Matrix Generation")
            self.logger.info("-"*80)
            matrices_dir = self.render_matrices()
            self.logger.info(f"✅ Matrices generated and saved in: {matrices_dir}")
            
            # Stage 4: Model Initialization
            self.logger.info("\n" + "="*80)
            self.logger.info("🔧 Stage 4: Model Initialization")
            self.logger.info("-" * 80)
            
            # Initialize model
            initializer = RGMModelInitializer(self.experiment.exp_dir, device=self.device)
            matrices = initializer.load_matrices()
            
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
            self.evaluate_model(mnist_data)
            
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
        
    def render_matrices(self) -> Dict[str, torch.Tensor]:
        """Render matrices from GNN specifications."""
        try:
            self.logger.info("Rendering matrices from GNN specs...")
            
            # Get GNN directory from experiment
            gnn_dir = self.experiment.get_dir('gnn_specs')
            if not gnn_dir.exists():
                raise FileNotFoundError(f"GNN specification directory not found: {gnn_dir}")
            
            # Initialize renderer with experiment directory
            renderer = RGMRenderer(self.experiment.exp_dir)
            
            # Generate matrices using new method name
            matrices = renderer.render_matrices()
            
            # Save matrices
            matrix_dir = self.experiment.get_dir('matrices')
            matrix_dir.mkdir(exist_ok=True)
            
            for name, matrix in matrices.items():
                torch.save(matrix, matrix_dir / f"{name}.pt")
                self.logger.debug(f"Saved matrix {name} with shape {matrix.shape}")
                
            self.logger.info(f"✅ Matrices generated and saved in: {matrix_dir}")
            return matrices
            
        except Exception as e:
            self.logger.error(f"Failed to render matrices: {str(e)}")
            raise
        
    def initialize_model(self, matrices: Dict[str, torch.Tensor]) -> RGMModelState:
        """Initialize RGM model."""
        try:
            self.logger.info("🔧 Initializing model components...")
            
            # Model configuration
            model_config = {
                'input_size': matrices['A0'].size(0),
                'hierarchy_levels': sum(1 for k in matrices if k.startswith('A')),
                'latent_dims': [64, 32, 16],  # Example latent dimensions
                'activation': 'relu',
                'normalization': 'batch_norm',
                'dropout_rate': 0.1
            }
            
            # Initialize model state
            model_state = RGMModelState(matrices, model_config)
            
            # Log model architecture
            self.logger.info("\n📐 Model Architecture:")
            self.logger.info("-"*40)
            self.logger.info(f"   - Input Size: {model_config['input_size']}")
            self.logger.info(f"   - Hierarchy Levels: {model_config['hierarchy_levels']}")
            self.logger.info(f"   - Latent Dimensions: {model_config['latent_dims']}")
            
            # Log parameter counts
            total_params = sum(p.numel() for p in model_state.get_parameters())
            self.logger.info(f"\n📊 Model Statistics:")
            self.logger.info("-"*40)
            self.logger.info(f"   - Total Parameters: {total_params:,}")
            self.logger.info(f"   - Device: {model_state.device}")
            
            return model_state
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {str(e)}")
            raise
        
    def train_model(self, mnist_data: Dict) -> None:
        """Train RGM model."""
        try:
            self.logger.info("🎯 Starting model training...")
            
            # Training configuration
            train_config = {
                'n_epochs': 50,
                'batch_size': 128,
                'learning_rate': 0.001,
                'log_interval': 10000,
                'checkpoint_interval': 5,
                'latent_dim': 64,
                'validation_interval': 5,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.001
                },
                'optimizer': {
                    'type': 'adam',
                    'betas': (0.9, 0.999),
                    'weight_decay': 1e-5
                },
                'scheduler': {
                    'type': 'plateau',
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-6
                }
            }
            
            # Log training configuration
            self.logger.info(f"📋 Training Configuration:")
            for key, value in train_config.items():
                self.logger.info(f"   - {key}: {value}")
            
            # Prepare data loaders
            data_loader = RGMDataLoader(train_config)
            data_loaders = data_loader.prepare_data_loaders(mnist_data)
            
            # Initialize trainer
            trainer = RGMTrainer(
                model_state=self.model_state,
                config=train_config,
                data_loaders=data_loaders
            )
            
            # Training loop
            for epoch in range(train_config['n_epochs']):
                self.logger.info(f"\n📈 Epoch [{epoch+1}/{train_config['n_epochs']}]")
                
                # Training phase
                train_metrics = trainer.train_epoch(data_loaders['train'])
                
                # Validation phase
                if (epoch + 1) % train_config['validation_interval'] == 0:
                    val_metrics = trainer.evaluate(data_loaders['val'])
                    self.logger.info(f"   ↳ Validation Loss: {val_metrics['val_loss']:.4f}")
                
                # Log metrics
                self.logger.info(f"   ↳ Train Loss: {train_metrics['avg_loss']:.4f}")
                self.logger.info(f"   ↳ Time: {train_metrics['time']:.2f}s")
                
                # Save checkpoint
                if (epoch + 1) % train_config['checkpoint_interval'] == 0:
                    checkpoint_path = self.experiment.get_dir('checkpoints') / f"checkpoint_epoch_{epoch+1}.pt"
                    self.model_state.save_checkpoint(checkpoint_path)
                    self.logger.info(f"   ↳ Checkpoint saved: {checkpoint_path}")
                
                # Check for early stopping
                if train_metrics.get('early_stop', False):
                    self.logger.info("⚠️ Early stopping triggered!")
                    break
            
            self.logger.info("✅ Model training complete")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
        
    def evaluate_model(self, mnist_data: Dict) -> None:
        """Evaluate RGM model."""
        try:
            self.logger.info("🔍 Starting model evaluation...")
            
            # Evaluation configuration
            eval_config = {
                'batch_size': 128,
                'n_samples': 1000,
                'latent_dims': [64, 32, 16]
            }
            
            # Initialize evaluator
            evaluator = RGMEvaluator(self.model_state, eval_config)
            
            # Compute metrics
            metrics = evaluator.compute_metrics(mnist_data['test'])
            
            # Log results
            self.logger.info("\n📊 Evaluation Results:")
            self.logger.info("-"*40)
            for metric, value in metrics.items():
                self.logger.info(f"   - {metric.replace('_', ' ').title()}: {value}")
            
            # Generate visualizations
            vis_dir = self.experiment.get_dir('analysis')
            evaluator.generate_visualizations(vis_dir)
            self.logger.info(f"\n🎨 Visualizations saved to: {vis_dir}")
            
            # Performance analysis
            self.logger.info("\n📈 Performance Analysis:")
            self.logger.info("-"*40)
            self.logger.info(f"   - GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            self.logger.info(f"   - CPU Usage: {psutil.cpu_percent()}%")
            self.logger.info(f"   - Memory Usage: {psutil.Process().memory_info().rss/1e9:.2f}GB")
            
            self.logger.info("✅ Evaluation complete")
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {str(e)}")
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
                'model_state': {
                    'config': self.model_state.state['config'],
                    'hierarchy_levels': self.model_state.state['hierarchy_levels'],
                    'parameter_shapes': {
                        name: list(param.shape) 
                        for name, param in self.model_state.state['parameters'].items()
                    }
                } if self.model_state else None,
                'last_metrics': getattr(self, '_last_metrics', None),
                'device': str(getattr(self, 'device', 'unknown')),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_memory': {
                    'allocated': torch.cuda.memory_allocated() / 1e9,
                    'cached': torch.cuda.memory_reserved() / 1e9
                } if torch.cuda.is_available() else None,
                'system_info': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available': psutil.virtual_memory().available / 1e9
                }
            }
            
            error_path = self.experiment.get_dir('simulation') / "error_state.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2, cls=CustomJSONEncoder)
            
            self.logger.info(f"💾 Error state saved to: {error_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save error state: {str(e)}")

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
        logging.error(f"❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
