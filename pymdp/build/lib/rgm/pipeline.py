"""
RGM Pipeline Implementation

This module provides the main pipeline for training and evaluating RGM models.
It coordinates data loading, model training, evaluation, and visualization.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .utils.rgm_logging import RGMLogging
from .utils.rgm_config_validator import RGMConfigValidator
from .utils.rgm_data_loader import RGMDataLoader
from .models.rgm_mnist import RGMMNISTModel
from .visualization import RGMRenderer
from .training.rgm_trainer import RGMTrainer


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
            
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model on test data.
        
        This method:
        1. Loads the best checkpoint
        2. Initializes model with saved state
        3. Runs evaluation on test set
        4. Computes and returns metrics
        
        Returns:
            Dictionary containing evaluation metrics:
            - test_loss: Overall test set loss
            - test_recon_loss: Reconstruction loss component
            - test_state_loss: State prediction loss component
            
        Raises:
            FileNotFoundError: If no checkpoint is found
            Exception: If evaluation fails
        """
        self.logger.info("Starting evaluation...")
        
        try:
            # Load best model checkpoint
            checkpoint_path = self.exp_dir / "checkpoints" / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"No checkpoint found at {checkpoint_path}"
                )
            
            # Load model state
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            matrices = checkpoint["matrices"]
            
            # Initialize model with loaded matrices
            model = RGMMNISTModel(
                matrices=matrices,
                config=self.config,
                device=self.device
            )
            model.load_state_dict(checkpoint["model_state"])
            self.logger.info("✓ Loaded model from checkpoint")
            
            # Prepare test data
            test_loader = self.data_loader.get_test_loader()
            self.logger.info("✓ Prepared test data loader")
            
            # Evaluate model
            metrics = self._evaluate_model(model, test_loader)
            self.logger.info("✓ Evaluation complete")
            
            # Log metrics
            for name, value in metrics.items():
                self.logger.info(f"   • {name}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def _evaluate_model(self,
                       model: RGMMNISTModel,
                       test_loader: DataLoader) -> Dict[str, float]:
        """Run evaluation loop on provided model and data.
        
        Args:
            model: RGM model to evaluate
            test_loader: DataLoader for test dataset
            
        Returns:
            Dictionary of averaged evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_state_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Forward pass
                loss, recon_loss, state_loss = model(batch)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_state_loss += state_loss.item()
                n_batches += 1
        
        # Compute averages
        metrics = {
            "test_loss": total_loss / n_batches,
            "test_recon_loss": total_recon_loss / n_batches,
            "test_state_loss": total_state_loss / n_batches
        }
        
        return metrics 