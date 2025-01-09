"""
RGM Model Trainer
===============

Handles training loop and optimization for the RGM model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Optional
from tqdm import tqdm

from .rgm_logging import RGMLogging
from .rgm_model_state import RGMModelState

class RGMTrainer:
    """Manages training of the RGM model."""
    
    def __init__(
        self,
        model: RGMModelState,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        exp_dir: Path,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: RGM model state
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            exp_dir: Experiment directory
            device: Computation device
        """
        self.logger = RGMLogging.get_logger("rgm.trainer")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🔧 Trainer initialized on device: {self.device}")
        
        # Model and data
        self.model = model
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
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Set visualization directory in model
        self.model.set_visualization_dir(self.vis_dir)
        
        # Optimizer setup
        opt_config = config["optimizer"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=opt_config["betas"],
            weight_decay=opt_config["weight_decay"]
        )
        
        # Learning rate scheduler
        sched_config = config["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=sched_config["factor"],
            patience=sched_config["patience"],
            min_lr=sched_config["min_lr"]
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.n_epochs}") as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = outputs['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{outputs['loss_components']['reconstruction']:.4f}",
                    'pred': f"{outputs['loss_components']['prediction']:.4f}",
                    'sparse': f"{outputs['loss_components']['sparsity']:.4f}"
                })
                
                # Visualize batch
                if batch_idx % self.config["log_interval"] == 0:
                    self.model.visualize_batch(data, outputs, batch_idx)
                    
        return epoch_loss / len(self.train_loader)
        
    def validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                val_loss += outputs['loss'].item()
                
        val_loss /= len(self.val_loader)
        return val_loss
        
    def train(self):
        """Run complete training loop."""
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Validation
            if (epoch + 1) % self.config["validation_interval"] == 0:
                val_loss = self.validate()
                self.logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint('best.pt')
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config["early_stopping"]["patience"]:
                    self.logger.info("Early stopping triggered")
                    break
                    
            # Regular checkpointing
            if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pt')
                
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename) 