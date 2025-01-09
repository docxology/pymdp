"""
RGM Model Trainer
==============

Training implementation for the Renormalization Generative Model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import time
import psutil
import numpy as np
from tqdm import tqdm

from rgm.utils import RGMLogging
from rgm.utils.rgm_model_state import RGMModelState
from rgm.utils.rgm_progress_tracker import RGMProgressTracker

class RGMTrainer:
    """Handles training for the Renormalization Generative Model."""
    
    def __init__(self, model_state: RGMModelState, config: Dict, data_loaders: Dict[str, DataLoader]):
        """Initialize trainer."""
        self.logger = RGMLogging.get_logger("rgm.trainer")
        self.model_state = model_state
        self.config = config
        self.device = model_state.device
        self.data_loaders = data_loaders
        
        # Initialize optimizer and scheduler
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Initialize progress tracker
        self.progress_tracker = RGMProgressTracker(
            self.config.get('output_dir', Path('training_progress'))
        )
        
        self.logger.info(f"🔧 Trainer initialized on device: {self.device}")
        
    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        
        if optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model_state.get_parameters(),
                lr=self.config['learning_rate'],
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        # Add other optimizer types as needed
        
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
    def _initialize_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Initialize learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'plateau').lower()
        
        if scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        # Add other scheduler types as needed
        
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch."""
        self.current_epoch += 1
        epoch_metrics = {
            'loss': 0.0,
            'reconstruction_error': 0.0,
            'samples_processed': 0,
            'batch_losses': []
        }
        
        start_time = time.time()
        n_batches = len(train_loader)
        log_interval = max(n_batches // 10, 1)  # Log roughly 10 times per epoch
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config['n_epochs']}", 
                   leave=False)
        
        for batch_idx, (data, labels) in enumerate(pbar):
            try:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self._forward_pass(data)
                loss = self._compute_loss(outputs, data, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_state.get_parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                epoch_metrics['loss'] += batch_loss
                epoch_metrics['batch_losses'].append(batch_loss)
                epoch_metrics['samples_processed'] += len(data)
                
                # Update progress bar
                avg_loss = epoch_metrics['loss'] / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'avg_loss': f"{avg_loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Periodic logging
                if batch_idx % self.config['log_interval'] == 0:
                    self.logger.info(
                        f"   ↳ Batch [{batch_idx:>5}/{n_batches}] "
                        f"Loss: {batch_loss:.4f} "
                        f"Avg Loss: {avg_loss:.4f}"
                    )
                    
            except torch.cuda.OutOfMemoryError:
                self.logger.error("❌ GPU out of memory!")
                torch.cuda.empty_cache()
                raise
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_idx}: {str(e)}")
                raise
        
        # Compute epoch statistics
        epoch_metrics.update({
            'time': time.time() - start_time,
            'avg_loss': epoch_metrics['loss'] / n_batches,
            'loss_std': np.std(epoch_metrics['batch_losses']),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Update learning rate scheduler
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(epoch_metrics['avg_loss'])
        new_lr = self.optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            self.logger.info(f"📉 Learning rate adjusted: {old_lr:.6f} → {new_lr:.6f}")
        
        # Early stopping check
        if epoch_metrics['avg_loss'] < self.best_loss - self.config['early_stopping']['min_delta']:
            self.best_loss = epoch_metrics['avg_loss']
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        if self.epochs_without_improvement >= self.config['early_stopping']['patience']:
            self.logger.info("⚠️ Early stopping triggered!")
            epoch_metrics['early_stop'] = True
        
        # Update progress tracker
        self.progress_tracker.update(self.current_epoch, epoch_metrics)
        
        return epoch_metrics
    
    def _forward_pass(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform forward pass through the model."""
        return self.model_state.model(data)
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the total loss based on the Free Energy Principle.
        
        The loss consists of:
        1. Reconstruction error (accuracy term)
        2. KL divergence of latent states (complexity term)
        3. Hierarchical consistency (message passing term)
        4. Precision weighting of each term
        """
        # Flatten input data
        batch_size = data.size(0)
        data_flat = data.reshape(batch_size, -1)
        
        # 1. Reconstruction error (accuracy)
        recon_loss = nn.MSELoss()(outputs['reconstructed'], data_flat)
        
        # 2. KL divergence (complexity)
        kl_loss = sum(
            -0.5 * torch.sum(1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp())
            for latent_mean, latent_log_var in outputs['latent_params']
        ) / batch_size
        
        # 3. Hierarchical consistency
        consistency_loss = sum(
            nn.MSELoss()(
                outputs['top_down'][i],
                outputs['bottom_up'][i]
            )
            for i in range(len(outputs['top_down']))
        )
        
        # Combine losses with learned precision weights
        total_loss = (
            self.model_state.state['parameters']['precision_recon'] * recon_loss +
            self.model_state.state['parameters']['precision_kl'] * kl_loss +
            self.model_state.state['parameters']['precision_consistency'] * consistency_loss
        )
        
        return total_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate model performance."""
        self.model_state['model'].eval()
        eval_metrics = {
            'val_loss': 0.0,
            'reconstruction_error': 0.0,
            'generation_quality': 0.0
        }
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self._forward_pass(data)
                loss = self._compute_loss(outputs, data, labels)
                
                # Update metrics
                eval_metrics['val_loss'] += loss.item()
                
        # Average metrics
        eval_metrics['val_loss'] /= len(val_loader)
        
        return eval_metrics 