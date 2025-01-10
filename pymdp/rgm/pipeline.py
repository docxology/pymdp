"""RGM Pipeline Module

This module provides the core pipeline functionality for the RGM (Relational Graph Machine) model.
It handles the complete lifecycle of RGM experiments including setup, training, evaluation, and analysis.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
from torch.utils.data import DataLoader

from .utils.rgm_logging import RGMLogging
from .utils.rgm_config_validator import RGMConfigValidator
from .utils.rgm_data_loader import RGMDataLoader
from .models.rgm_mnist import RGMMNISTModel
from .visualization import RGMRenderer
from .training.rgm_trainer import RGMTrainer
from .utils.matrix_init import initialize_matrices

class RGMPipeline:
    """Pipeline for training and evaluating RGM models.
    
    This class orchestrates the complete RGM experiment pipeline including:
    - Environment setup and validation
    - Data preparation and loading
    - Model initialization and training
    - Evaluation and analysis
    - Visualization and reporting
    
    Args:
        config_path: Path to configuration JSON file, or None to use default
        exp_dir: Experiment directory for saving outputs
        device: Optional device specification ("cpu" or None for auto-detect)
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "rgm_default_config.json"
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        exp_dir: str = "experiments/rgm",
        device: Optional[str] = None
    ):
        """Initialize the RGM pipeline with configuration and resources."""
        self.exp_dir = Path(exp_dir)
        
        # Create experiment directories first
        self._create_directories()
        
        # Set up logging
        self._setup_logging()
        
        # Load and validate configuration
        config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_and_validate_config(config_path)
        
        # Setup device
        self.device = self._setup_device(device)
        self.logger.info(f"💻 Using device: {self.device}")
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✓ Pipeline initialized successfully")
    
    def _setup_logging(self) -> None:
        """Set up logging for the pipeline.
        
        Configures logging with both console and file output, using the RGMLogging utility.
        Console output includes color formatting for different log levels.
        File output includes detailed information with timestamps and line numbers.
        """
        log_dir = self.exp_dir / "logs"
        self.logger = RGMLogging.setup_logging(
            log_dir=log_dir,
            level="INFO",
            log_to_file=True,
            name="rgm.pipeline"
        )
        self.logger.info("✓ Logging initialized")
    
    def _create_directories(self) -> None:
        """Create necessary experiment directories."""
        dirs = [
            self.exp_dir,
            self.exp_dir / "checkpoints",
            self.exp_dir / "logs",
            self.exp_dir / "visualizations",
            self.exp_dir / "analysis",
            self.exp_dir / "data"
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _load_and_validate_config(self, config_path: str) -> Dict:
        """Load and validate configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
            JSONDecodeError: If config file is not valid JSON
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse config file: {e}")
            raise
            
        # Validate configuration using RGMConfigValidator
        validator = RGMConfigValidator()
        try:
            validator._validate_config(config)
            self.logger.info("✓ Configuration validated successfully")
        except ValueError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
            
        return config
        
    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """Set up computation device (CPU/GPU).
        
        Args:
            device: Optional device specification
            
        Returns:
            torch.device: Selected computation device
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        try:
            # Initialize data loader
            data_dir = self.exp_dir / "data"
            data_dir.mkdir(exist_ok=True)
            self.data_loader = RGMDataLoader(
                config=self.config,
                data_dir=str(data_dir)
            )
            self.logger.info("✓ Data loader initialized")
            
            # Initialize matrices before model
            self.matrices = initialize_matrices(self.config, self.device)
            self.logger.info("✓ Matrices initialized")
            
            # Initialize model
            self.model = RGMMNISTModel(
                config=self.config,
                device=self.device,
                matrices=self.matrices
            )
            self.model.to(self.device)
            self.logger.info("✓ Model initialized")
            
            # Initialize renderer
            self.renderer = RGMRenderer(
                exp_dir=self.exp_dir,
                config=self.config,
                device=self.device
            )
            self.logger.info("✓ Renderer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def train(self) -> None:
        """Train the RGM model using configured parameters."""
        self.logger.info("Starting RGM training...")
        
        try:
            # Get data loaders
            train_loader = self.data_loader.get_train_loader()
            val_loader = self.data_loader.get_val_loader()
            self.logger.info("✓ Prepared data loaders")
            
            # Initialize trainer
            trainer = RGMTrainer(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config["training"],
                exp_dir=self.exp_dir,
                device=self.device
            )
            
            # Train model
            trainer.train()
            self.logger.info("✓ Training complete")
            
            # Save final model state
            trainer.save_checkpoint("final")
            self.logger.info("✓ Saved final model state")
            
            # Visualize final matrices
            final_matrices = self.model.get_matrices()
            self.renderer.visualize_matrices(final_matrices)
            self.logger.info("✓ Generated final visualizations")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        self.logger.info("Starting model evaluation...")
        
        try:
            # Get test loader
            test_loader = self.data_loader.get_test_loader()
            
            # Run evaluation
            metrics = self.model.evaluate(test_loader)
            self.logger.info("✓ Completed evaluation")
            
            # Log metrics
            for name, value in metrics.items():
                self.logger.info(f"{name}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def analyze(self) -> Dict[str, Any]:
        """Perform post-training analysis."""
        self.logger.info("Starting post-training analysis...")
        
        try:
            # Analyze matrix evolution
            matrix_analysis = self.renderer.analyze_matrix_evolution(
                self.model.get_matrices()
            )
            
            # Generate visualizations
            self.renderer.generate_analysis_plots(matrix_analysis)
            
            return matrix_analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise 