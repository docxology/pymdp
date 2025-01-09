"""
RGM Model Trainer
=================

Handles training loop and optimization for the RGM model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Optional
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

from ..utils.rgm_logging import RGMLogging
from ..utils.rgm_model_state import RGMModelState
from ..utils.visualization_utils import RGMVisualizationUtils

class RGMTrainer:
    """Manages training of the RGM model."""
    
    def __init__(
        self,
        model_state: RGMModelState,  # Changed from 'model' to 'model_state'
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
        """
        self.logger = RGMLogging.get_logger("rgm.trainer")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and data
        self.model_state = model_state
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training config
        self.config = config
        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        
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
        
        # Initialize optimizer and scheduler
        self._setup_training()
        
        self.logger.info(f"🔧 Trainer initialized on device: {self.device}")
        
    def _setup_training(self):
        """Setup optimizer and learning rate scheduler."""
        # Get optimizer config
        opt_config = self.config["optimizer"]
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model_state.parameters(),
            lr=self.learning_rate,
            betas=tuple(opt_config["betas"]),
            weight_decay=opt_config["weight_decay"]
        )
        
        # Setup scheduler
        sched_config = self.config["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=sched_config["factor"],
            patience=sched_config["patience"],
            min_lr=sched_config["min_lr"]
        )
        
    def _log_training_state(self, epoch: int, batch_idx: int, loss: float, outputs: Dict):
        """Log training progress."""
        # Calculate progress
        progress = batch_idx / len(self.train_loader) * 100
        
        # Log to console
        self.logger.info(
            f"Epoch [{epoch+1}/{self.n_epochs}] "
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
        
        with open(self.log_dir / 'training_log.jsonl', 'a') as f:
            f.write(json.dumps(state) + '\n')
            
    def train_epoch(self):
        """Train for one epoch."""
        self.model_state.train()
        epoch_loss = 0.0
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.n_epochs}") as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                
                # Forward pass
                outputs = self.model_state(data)
                loss = outputs['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
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
        
    def train(self):
        """
        Execute complete training loop.
        
        This method:
        1. Runs training epochs
        2. Performs validation
        3. Handles checkpointing
        4. Manages early stopping
        5. Updates learning rate
        """
        self.logger.info("\n🏃 Starting training loop...")
        
        try:
            for epoch in range(self.n_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_loss = self.train_epoch()
                
                # Validation phase
                if (epoch + 1) % self.config["validation_interval"] == 0:
                    val_loss = self.validate()
                    
                    # Update learning rate
                    self.scheduler.step(val_loss)
                    
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
                
                # Save checkpoint
                if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                    self._save_checkpoint(f'epoch_{epoch+1}.pt')
                    self.logger.info(f"💾 Checkpoint saved at epoch {epoch+1}")
                    
                # Plot training curves
                if (epoch + 1) % self.config.get("plot_frequency", 5) == 0:
                    self._plot_training_curves()
                    
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise
            
        self.logger.info("✅ Training complete!")
        
    def validate(self):
        """Run validation loop."""
        self.model_state.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                outputs = self.model_state(data)
                val_loss += outputs['loss'].item()
                
        return val_loss / len(self.val_loader)
        
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping conditions."""
        if val_loss < self.best_val_loss - self.config["early_stopping"]["min_delta"]:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
            
        self.patience_counter += 1
        return self.patience_counter >= self.config["early_stopping"]["patience"]
        
    def _plot_training_curves(self):
        """Generate training visualization plots."""
        # Load training log
        log_file = self.log_dir / 'training_log.jsonl'
        if not log_file.exists():
            return
            
        # Read log data
        losses = {'train': [], 'val': []}
        lrs = []
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                losses['train'].append(entry['loss'])
                if 'val_loss' in entry:
                    losses['val'].append(entry['val_loss'])
                lrs.append(entry['learning_rate'])
                
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(losses['train'], label='Train')
        if losses['val']:
            plt.plot(losses['val'], label='Validation')
        plt.title('Training Progress')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.vis_dir / 'training_curves.png')
        plt.close()
        
        # Plot learning rate
        plt.figure(figsize=(10, 5))
        plt.plot(lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.savefig(self.vis_dir / 'learning_rate.png')
        plt.close() 