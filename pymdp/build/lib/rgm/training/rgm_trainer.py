"""
RGM Model Trainer
=================

Handles training loop and optimization for the RGM model.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Union
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

from ..utils.rgm_logging import RGMLogging
from ..utils.rgm_model_state import RGMModelState

class RGMTrainer:
    """Manages training of the RGM model."""
    
    def __init__(
        self,
        model_state: RGMModelState,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        exp_dir: Path,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_state: RGM model state
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            exp_dir: Experiment directory
            device: Computation device
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.logger = RGMLogging.get_logger("rgm.trainer")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and data
        self.model_state = model_state
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Validate config
        self._validate_config(config)
        self.config = config
        
        # Setup directories
        self.exp_dir = exp_dir
        self.checkpoint_dir = exp_dir / "checkpoints"
        self.vis_dir = exp_dir / "visualizations" / "training"
        self.log_dir = exp_dir / "logs"
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set visualization directory in model
        self.model_state.set_visualization_dir(self.vis_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Initialize optimizer and scheduler
        self._setup_training()
        
        self.logger.info(f"🔧 Trainer initialized on device: {self.device}")
        
    def _validate_config(self, config: Dict):
        """Validate training configuration."""
        required_fields = [
            'n_epochs', 'batch_size', 'learning_rate', 'log_interval',
            'checkpoint_interval', 'validation_interval', 'early_stopping',
            'optimizer', 'scheduler'
        ]
        
        missing = [f for f in required_fields if f not in config]
        if missing:
            raise ValueError(f"Missing required configuration fields: {missing}")
            
        if config['early_stopping']['patience'] < 1:
            raise ValueError("Early stopping patience must be >= 1")
            
    def _setup_training(self):
        """Setup optimizer and learning rate scheduler."""
        opt_config = self.config["optimizer"]
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model_state.parameters(),
            lr=self.config["learning_rate"],
            betas=tuple(opt_config["betas"]),
            weight_decay=opt_config["weight_decay"],
            amsgrad=opt_config.get("amsgrad", False)
        )
        
        # Setup scheduler
        sched_config = self.config["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=sched_config["factor"],
            patience=sched_config["patience"],
            min_lr=sched_config["min_lr"],
            threshold=sched_config.get("threshold", 1e-4),
            cooldown=sched_config.get("cooldown", 0)
        )
        
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file
            is_best: Whether this is the best model so far
            
        Raises:
            IOError: If saving checkpoint fails
        """
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model_state.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'training_history': self.training_history
            }
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / filename
            torch.save(checkpoint, checkpoint_path)
            
            # If best model, create a copy
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                
            self.logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise IOError(f"Failed to save checkpoint: {str(e)}")
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If loading checkpoint fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model_state.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint.get('training_history', [])
            
            self.logger.info(f"📂 Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
            
    def _log_training_state(self, epoch: int, batch_idx: int, loss: float, outputs: Dict):
        """Log training progress."""
        # Calculate progress
        progress = batch_idx / len(self.train_loader) * 100
        
        # Log to console
        self.logger.info(
            f"Epoch [{epoch+1}/{self.config['n_epochs']}] "
            f"[{batch_idx:>4d}/{len(self.train_loader):<4d} ({progress:>3.0f}%)] "
            f"Loss: {loss:.6f}"
        )
        
        # Save training state
        state = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'loss_components': outputs['loss_components']
        }
        
        # Append to history
        self.training_history.append(state)
        
        # Save to log file
        with open(self.log_dir / 'training_log.jsonl', 'a') as f:
            f.write(json.dumps(state) + '\n')
            
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
            
        Raises:
            RuntimeError: If training fails
        """
        self.model_state.train()
        epoch_loss = 0.0
        
        try:
            with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config['n_epochs']}") as pbar:
                for batch_idx, (data, _) in enumerate(pbar):
                    data = data.to(self.device)
                    
                    # Forward pass
                    outputs = self.model_state(data)
                    loss = outputs['loss']
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if "gradient_clip_val" in self.config:
                        torch.nn.utils.clip_grad_norm_(
                            self.model_state.parameters(),
                            self.config["gradient_clip_val"]
                        )
                    
                    self.optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    self.global_step += 1
                    
                    # Log progress
                    if batch_idx % self.config["log_interval"] == 0:
                        self._log_training_state(
                            self.current_epoch,
                            batch_idx,
                            loss.item(),
                            outputs
                        )
                        
                        # Visualize batch
                        self.model_state.visualize_batch(data, outputs, batch_idx)
                        
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'recon': f"{outputs['loss_components']['reconstruction']:.4f}",
                        'pred': f"{outputs['loss_components']['prediction']:.4f}",
                        'sparse': f"{outputs['loss_components']['sparsity']:.4f}"
                    })
                    
            return epoch_loss / len(self.train_loader)
            
        except Exception as e:
            self.logger.error(f"Error during training epoch: {str(e)}")
            raise RuntimeError(f"Training epoch failed: {str(e)}")
            
    def validate(self) -> float:
        """
        Run validation loop.
        
        Returns:
            Average validation loss
            
        Raises:
            RuntimeError: If validation fails
        """
        self.model_state.eval()
        val_loss = 0.0
        
        try:
            with torch.no_grad():
                for data, _ in self.val_loader:
                    data = data.to(self.device)
                    outputs = self.model_state(data)
                    val_loss += outputs['loss'].item()
                    
            return val_loss / len(self.val_loader)
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise RuntimeError(f"Validation failed: {str(e)}")
            
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check early stopping conditions.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            Whether to stop training
        """
        if val_loss < self.best_val_loss - self.config["early_stopping"]["min_delta"]:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
            
        self.patience_counter += 1
        return (self.patience_counter >= self.config["early_stopping"]["patience"] and
                self.current_epoch >= self.config["early_stopping"]["min_epochs"])
                
    def _plot_training_curves(self):
        """Generate training visualization plots."""
        if not self.training_history:
            return
            
        # Extract metrics
        epochs = [entry['epoch'] for entry in self.training_history]
        losses = [entry['loss'] for entry in self.training_history]
        lrs = [entry['learning_rate'] for entry in self.training_history]
        
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, label='Training Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.vis_dir / 'training_curves.png')
        plt.close()
        
        # Plot learning rate
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.savefig(self.vis_dir / 'learning_rate.png')
        plt.close()
        
    def train(self):
        """
        Execute complete training loop.
        
        This method:
        1. Runs training epochs
        2. Performs validation
        3. Handles checkpointing
        4. Manages early stopping
        5. Updates learning rate
        
        Raises:
            RuntimeError: If training fails
        """
        self.logger.info("\n🏃 Starting training loop...")
        
        try:
            for epoch in range(self.current_epoch, self.config["n_epochs"]):
                self.current_epoch = epoch
                
                # Training phase
                train_loss = self.train_epoch()
                
                # Validation phase
                if (epoch + 1) % self.config["validation_interval"] == 0:
                    val_loss = self.validate()
                    
                    # Update learning rate
                    self.scheduler.step(val_loss)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.save_checkpoint('best_model.pt', is_best=True)
                    
                    # Check for early stopping
                    if self._check_early_stopping(val_loss):
                        self.logger.info("⚠️ Early stopping triggered!")
                        break
                    
                    # Log validation results
                    self.logger.info(
                        f"\n📊 Epoch {epoch+1} Results:\n"
                        f"   • Train Loss: {train_loss:.6f}\n"
                        f"   • Val Loss:   {val_loss:.6f}\n"
                        f"   • LR:         {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                
                # Regular checkpoint
                if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}.pt')
                    
                # Plot training curves
                if (epoch + 1) % self.config.get("plot_frequency", 5) == 0:
                    self._plot_training_curves()
                    
            # Final checkpoint
            self.save_checkpoint('final_model.pt')
            self._plot_training_curves()
            
            self.logger.info("✅ Training complete!")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            # Save error state
            try:
                self.save_checkpoint('error_state.pt')
            except:
                pass
            raise RuntimeError(f"Training failed: {str(e)}") 