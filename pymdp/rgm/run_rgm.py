#!/usr/bin/env python3
"""RGM Pipeline Runner

Main entry point for training and evaluating RGM models.
Handles configuration, data loading, training, and visualization.

The pipeline consists of 5 stages:
1. MNIST Dataset Preparation
   - Download and validate MNIST dataset
   - Generate sample visualizations
   - Confirm data integrity

2. GNN Matrix Construction
   - Initialize matrices based on configuration
   - Analyze prior states and matrix properties
   - Generate matrix visualizations

3. Model Training
   - Initialize model with matrices
   - Train on MNIST dataset
   - Track and visualize progress

4. Model Evaluation
   - Evaluate on test set
   - Generate performance metrics
   - Create reconstruction samples

5. Post-Test Analysis
   - Analyze matrix evolution
   - Generate final visualizations
   - Compile comprehensive report
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union, Any
import torch
from torch.utils.data import DataLoader

try:
    # Try importing as installed package
    from rgm.utils.rgm_logging import RGMLogging
    from rgm.utils.rgm_config_validator import RGMConfigValidator
    from rgm.utils.rgm_data_loader import RGMDataLoader
    from rgm.models.rgm_mnist import RGMMNISTModel
    from rgm.visualization import RGMRenderer
    from rgm.training.rgm_trainer import RGMTrainer
except ImportError:
    # Try relative imports for development
    from .utils.rgm_logging import RGMLogging
    from .utils.rgm_config_validator import RGMConfigValidator
    from .utils.rgm_data_loader import RGMDataLoader
    from .models.rgm_mnist import RGMMNISTModel
    from .visualization import RGMRenderer
    from .training.rgm_trainer import RGMTrainer

# Default configuration
DEFAULT_CONFIG = "configs/mnist_config.json"
DEFAULT_EXP_DIR = "experiments/mnist"
DEFAULT_MODE = "train"

# Add parent directory to Python path for development mode
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if os.path.exists(os.path.join(PROJECT_ROOT, ".env")):
    sys.path.insert(0, PROJECT_ROOT)

import argparse

def ensure_package_installed():
    """Ensure the RGM package is installed."""
    try:
        import torch
        from rgm.utils.rgm_logging import RGMLogging
        print("✓ RGM package is installed")
        return True
    except ImportError:
        print("ℹ️ RGM package not installed, running setup...")
        setup_script = os.path.join(PROJECT_ROOT, "setup.py")
        try:
            subprocess.check_call([sys.executable, setup_script])
            print("✓ Setup completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Setup failed: {e}")
            return False

def check_environment():
    """Check and setup the environment."""
    print("🔍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        sys.exit("❌ Python 3.8 or higher is required")
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ CUDA not available, using CPU")
    
    # Check dependencies
    try:
        import numpy
        import matplotlib
        import seaborn
        import pandas
        import scipy
        import tqdm
        import yaml
        import jsonschema
        import psutil
        print("✓ All dependencies are installed")
    except ImportError as e:
        module = str(e).split("'")[1]
        print(f"❌ Missing dependency: {module}")
        print("ℹ️ Running setup to install dependencies...")
        if not ensure_package_installed():
            sys.exit(1)

def setup_environment(exp_dir: Union[str, Path]) -> None:
    """Setup experiment environment.
    
    Args:
        exp_dir: Path to experiment directory
    """
    exp_dir = Path(exp_dir)
    
    # Create experiment directories
    dirs = [
        exp_dir,
        exp_dir / "checkpoints",
        exp_dir / "logs",
        exp_dir / "visualizations"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {d}")

# Only import RGM modules after ensuring installation
if ensure_package_installed():
    from rgm.utils.rgm_logging import RGMLogging
    from rgm.utils.rgm_config_validator import RGMConfigValidator
    from rgm.utils.rgm_data_loader import RGMDataLoader
    from rgm.models.rgm_mnist import RGMMNISTModel
    from rgm.visualization import RGMRenderer
    from rgm.training.rgm_trainer import RGMTrainer
else:
    sys.exit(1)


class RGMPipeline:
    """Pipeline for training and evaluating RGM models.
    
    Features:
    - Configuration validation and management
    - Data loading and preprocessing
    - Model initialization and training
    - Matrix visualization and analysis
    - Comprehensive logging and error handling
    
    Args:
        config_path: Path to configuration JSON file
        exp_dir: Experiment directory for saving outputs
        device: Optional torch device (defaults to CUDA if available)
    """
    
    def __init__(self,
                 config_path: str,
                 exp_dir: str,
                 device: Optional[torch.device] = None):
        """Initialize the RGM pipeline with configuration and resources."""
        self.exp_dir = Path(exp_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Set up logging
        self.logger = RGMLogging.get_logger("rgm.pipeline")
        self.logger.info("🚀 Initializing RGM Pipeline")
        self.logger.info(f"📂 Experiment directory: {self.exp_dir}")
        self.logger.info(f"💻 Device: {self.device}")
        
        # Load and validate configuration
        self.config = RGMConfigValidator.load_and_validate(config_path)
        
        # Initialize components
        self.data_loader = RGMDataLoader(self.config)
        self.renderer = RGMRenderer(self.exp_dir, self.config, self.device)
        
    def train(self) -> None:
        """Train the RGM model using configured parameters.
        
        This method handles the complete training pipeline:
        1. Matrix generation
        2. Model initialization
        3. Data preparation
        4. Training loop execution
        5. Model checkpointing
        6. Final visualization
        
        Raises:
            Exception: If any stage of training fails
        """
        self.logger.info("Starting RGM training...")
        
        try:
            # Generate matrices
            matrices = self.renderer.generate_matrices()
            self.logger.info("✓ Generated initial matrices")
            
            # Initialize model
            model = RGMMNISTModel(
                matrices=matrices,
                config=self.config,
                device=self.device
            )
            self.logger.info("✓ Initialized model")
            
            # Prepare data
            train_loader = self.data_loader.get_train_loader()
            val_loader = self.data_loader.get_val_loader()
            self.logger.info("✓ Prepared data loaders")
            
            # Initialize trainer
            trainer = RGMTrainer(
                model=model,
                config=self.config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                exp_dir=self.exp_dir
            )
            
            # Train model
            trainer.train()
            self.logger.info("✓ Training complete")
            
            # Save final model state
            trainer.save_checkpoint("final")
            self.logger.info("✓ Saved final model state")
            
            # Visualize final matrices
            final_matrices = model.get_matrices()
            self.renderer.visualize_matrices(final_matrices)
            self.logger.info("✓ Generated final visualizations")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the trained model on test data.
        
        Args:
            test_loader: DataLoader containing test dataset
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            Exception: If evaluation fails
        """
        self.logger.info("Starting model evaluation...")
        
        try:
            # Load best model
            model = RGMMNISTModel.load_from_checkpoint(
                self.exp_dir / "checkpoints" / "best.pt",
                config=self.config,
                device=self.device
            )
            self.logger.info("✓ Loaded best model checkpoint")
            
            # Run evaluation
            metrics = model.evaluate(test_loader)
            self.logger.info("✓ Completed evaluation")
            
            # Log metrics
            for name, value in metrics.items():
                self.logger.info(f"{name}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def analyze(self) -> Dict[str, Any]:
        """Perform post-training analysis.
        
        Returns:
            Dictionary containing analysis results
            
        Raises:
            Exception: If analysis fails
        """
        self.logger.info("Starting post-training analysis...")
        
        try:
            # Load final model state
            model = RGMMNISTModel.load_from_checkpoint(
                self.exp_dir / "checkpoints" / "final.pt",
                config=self.config,
                device=self.device
            )
            
            # Analyze matrix evolution
            matrix_analysis = self.renderer.analyze_matrix_evolution(
                model.get_matrices()
            )
            
            # Generate visualizations
            self.renderer.generate_analysis_plots(matrix_analysis)
            
            return matrix_analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for RGM pipeline.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate RGM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run in evaluation mode (default: training mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=DEFAULT_EXP_DIR,
        help="Experiment directory for saving outputs"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Create experiment directory if it doesn't exist
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config directory if it doesn't exist
    config_dir = Path(args.config).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # If config file doesn't exist, create a default one
    if not Path(args.config).exists():
        create_default_config(args.config)
    
    return args

def create_default_config(config_path: str) -> None:
    """Create a default configuration file if none exists.
    
    Args:
        config_path: Path where to create the config file
    """
    default_config = {
        "data": {
            "dataset": "mnist",
            "batch_size": 64,
            "num_workers": 4
        },
        "model": {
            "hidden_size": 256,
            "num_layers": 4,
            "dropout": 0.1
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 10
        }
    }
    
    import json
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"Created default config at {config_path}")

def main() -> None:
    """Main entry point for command line execution."""
    args = parse_args()
    
    # Initialize device
    device = torch.device(args.device)
    
    try:
        # Initialize pipeline
        pipeline = RGMPipeline(
            config_path=args.config,
            exp_dir=args.exp_dir,
            device=device
        )
        
        # Run requested operation
        if not args.evaluate:
            pipeline.train()
        else:
            metrics = pipeline.evaluate()
            print("\nEvaluation Metrics:")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
                
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
