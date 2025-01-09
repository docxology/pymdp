"""
RGM Execution Engine
===================

Runs RGM inference and learning on MNIST dataset.

This module:
1. Loads rendered matrices and configurations
2. Performs hierarchical message passing
3. Implements active learning for MNIST
4. Tracks learning progress and performance
5. Handles checkpointing and state management

Components:
----------
- RGMExecutor: Main execution engine
- Message Passing: Hierarchical belief propagation
- Active Learning: Online parameter updates
- Metrics: Performance tracking and analysis
- Checkpointing: State persistence and recovery
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm

# Local imports using relative imports
from .utils.rgm_experiment_utils import RGMExperimentUtils
from .utils.rgm_model_initializer import RGMModelInitializer
from .utils.rgm_data_manager import RGMDataManager
from .utils.rgm_message_passing import RGMMessagePassing
from .utils.rgm_matrix_normalizer import RGMMatrixNormalizer
from .utils.rgm_metrics_utils import RGMMetricsUtils

class RGMExecutor:
    """Executes RGM inference and learning"""
    
    def __init__(self, matrices: Dict[str, np.ndarray]):
        """
        Initialize executor with normalized matrices.
        
        Args:
            matrices: Dictionary of normalized matrices
        """
        try:
            # Get experiment state
            self.experiment = RGMExperimentUtils.get_experiment()
            self.logger = RGMExperimentUtils.get_logger('executor')
            self.logger.info("Initializing RGM executor...")
            
            # Store matrices
            self.matrices = matrices
            
            # Load configuration
            self.config = self._load_config()
            
            # Initialize model state
            self.model_initializer = RGMModelInitializer()
            self.model_state = self.model_initializer.initialize_model_state(self.config)
            
            # Initialize components
            self.message_passer = RGMMessagePassing(self.config)
            self.matrix_normalizer = RGMMatrixNormalizer()
            self.metrics = RGMMetricsUtils()
            
            # Initialize data manager
            self.data_manager = RGMDataManager(self.config)
            
            # Create checkpoint directory
            self.checkpoint_dir = self.experiment['dirs']['simulation'] / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
            
            self.logger.info("RGM executor initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize executor: {str(e)}")
            raise
            
    def _load_config(self) -> Dict:
        """Load experiment configuration"""
        config_path = self.experiment['dirs']['config'] / "merged_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
            
    def execute_inference(self) -> Path:
        """
        Execute RGM inference and learning on MNIST.
        
        Returns:
            Path to results directory
        """
        try:
            self.logger.info("Starting RGM execution...")
            start_time = time.time()
            
            # Load MNIST data
            train_loader, test_loader = self.data_manager.get_data_loaders()
            self.logger.info("Loaded MNIST datasets")
            
            # Training phase
            self.logger.info("Starting training phase...")
            for epoch in range(self.config['training']['n_epochs']):
                self._train_epoch(epoch, train_loader)
                
                # Validate on test set
                if (epoch + 1) % self.config['training']['eval_frequency'] == 0:
                    self._evaluate(epoch, test_loader)
                    
                # Save checkpoint
                if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                    self._save_checkpoint(epoch)
                    
            # Final evaluation
            self.logger.info("Running final evaluation...")
            final_metrics = self._evaluate(epoch, test_loader, final=True)
            
            # Save results
            results_dir = self._save_results(final_metrics)
            
            duration = time.time() - start_time
            self.logger.info(f"Execution completed in {duration:.2f}s")
            
            return results_dir
            
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            self._save_error_state()
            raise
            
    def _train_epoch(self, epoch: int, train_loader):
        """Train one epoch"""
        self.logger.info(f"\nEpoch {epoch+1}")
        
        # Progress tracking
        n_batches = len(train_loader)
        progress = tqdm(train_loader, desc=f"Training epoch {epoch+1}")
        
        # Training metrics
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'elbo': 0.0
        }
        
        # Train on batches
        for batch_idx, (data, targets) in enumerate(progress):
            batch_metrics = self._train_batch(data, targets)
            
            # Update metrics
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
                
            # Update progress
            progress.set_postfix({
                'loss': f"{batch_metrics['loss']:.4f}",
                'acc': f"{batch_metrics['accuracy']:.2%}"
            })
            
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        # Log epoch metrics
        self.logger.info(
            f"Epoch {epoch+1} - "
            f"Loss: {epoch_metrics['loss']:.4f}, "
            f"Accuracy: {epoch_metrics['accuracy']:.2%}, "
            f"ELBO: {epoch_metrics['elbo']:.4f}"
        )
        
        return epoch_metrics
        
    def _train_batch(self, data: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Process batch through RGM hierarchy"""
        try:
            batch_size = data.size(0)
            metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'elbo': 0.0
            }
            
            # Process each sample
            for idx in range(batch_size):
                image = data[idx].squeeze().numpy()
                label = targets[idx].item()
                
                # Run inference
                sample_metrics = self._process_sample(image, label, training=True)
                
                # Update batch metrics
                for key in metrics:
                    metrics[key] += sample_metrics[key]
                    
            # Average batch metrics
            for key in metrics:
                metrics[key] /= batch_size
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in batch training: {str(e)}")
            raise
            
    def _process_sample(self, image: np.ndarray, label: int, training: bool = True) -> Dict:
        """Process single sample through hierarchy"""
        try:
            # Initialize metrics
            metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'elbo': 0.0
            }
            
            # Reset beliefs
            self.model_state['beliefs'] = self.model_initializer._initialize_beliefs(
                self.config['model']['hierarchy']
            )
            
            # Run message passing
            beliefs, converged = self.message_passer.run_message_passing(
                self.model_state['beliefs'],
                self.matrices,
                self.model_state['learning']['precision']
            )
            
            if not converged:
                self.logger.warning("Message passing did not converge")
                
            # Update model state
            self.model_state['beliefs'] = beliefs
            
            # Compute metrics
            predicted = np.argmax(beliefs['states'][-1])
            metrics['accuracy'] = float(predicted == label)
            metrics['elbo'] = self._compute_elbo(beliefs)
            metrics['loss'] = -metrics['elbo']
            
            # Active learning update if training
            if training:
                self._active_learning_update(label)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error processing sample: {str(e)}")
            raise
            
    def _compute_elbo(self, beliefs: Dict[str, List[np.ndarray]]) -> float:
        """Compute Evidence Lower BOund"""
        try:
            elbo = 0.0
            
            # Add state terms
            for level in range(len(beliefs['states'])):
                state = beliefs['states'][level]
                elbo += np.sum(state * np.log(state + 1e-12))  # Entropy
                
                # Add likelihood if not top level
                if level < len(beliefs['states']) - 1:
                    A = self.matrices[f'A{level}']
                    higher_state = beliefs['states'][level + 1]
                    likelihood = np.dot(A.T, higher_state)
                    elbo += np.sum(state * np.log(likelihood + 1e-12))
                    
            # Add factor terms
            for level in range(len(beliefs['factors'])):
                factor = beliefs['factors'][level]
                elbo += np.sum(factor * np.log(factor + 1e-12))  # Entropy
                
                B = self.matrices[f'B{level}']
                D = self.matrices[f'D{level}']
                likelihood = np.dot(B.T, factor) * D
                elbo += np.sum(factor * np.log(likelihood + 1e-12))
                
            return float(elbo)
            
        except Exception as e:
            self.logger.error(f"Error computing ELBO: {str(e)}")
            raise
            
    def _active_learning_update(self, label: int):
        """Update model based on prediction error"""
        try:
            # Get current prediction
            predicted = np.argmax(self.model_state['beliefs']['states'][-1])
            
            # Only update if prediction is wrong
            if predicted != label:
                # Compute prediction error
                target = np.zeros_like(self.model_state['beliefs']['states'][-1])
                target[label] = 1.0
                error = target - self.model_state['beliefs']['states'][-1]
                
                # Update precision based on error
                self.model_state['learning']['precision'] *= (
                    1.0 + self.model_state['learning']['beta'] * np.sum(error ** 2)
                )
                
                # Update confusion matrix
                self.model_state['metrics']['confusion_matrix'][label, predicted] += 1
                
        except Exception as e:
            self.logger.error(f"Error in active learning update: {str(e)}")
            raise
            
    def _evaluate(self, epoch: int, test_loader, final: bool = False) -> Dict:
        """Evaluate model on test set"""
        try:
            self.logger.info(f"\nEvaluating model at epoch {epoch+1}")
            
            metrics = {
                'accuracy': 0.0,
                'elbo': 0.0,
                'confusion_matrix': np.zeros((10, 10)),
                'class_accuracies': np.zeros(10),
                'class_counts': np.zeros(10)
            }
            
            # Process test set
            n_samples = 0
            for data, targets in tqdm(test_loader, desc="Evaluating"):
                batch_size = data.size(0)
                n_samples += batch_size
                
                for idx in range(batch_size):
                    image = data[idx].squeeze().numpy()
                    label = targets[idx].item()
                    
                    sample_metrics = self._process_sample(
                        image, 
                        label,
                        training=False
                    )
                    
                    metrics['accuracy'] += sample_metrics['accuracy']
                    metrics['elbo'] += sample_metrics['elbo']
                    
                    predicted = np.argmax(self.model_state['beliefs']['states'][-1])
                    metrics['confusion_matrix'][label, predicted] += 1
                    metrics['class_counts'][label] += 1
                    if predicted == label:
                        metrics['class_accuracies'][label] += 1
                        
            # Average metrics
            metrics['accuracy'] /= n_samples
            metrics['elbo'] /= n_samples
            metrics['class_accuracies'] /= metrics['class_counts']
            
            # Log evaluation results
            self._log_evaluation_metrics(metrics, final)
            
            # Save if final
            if final:
                self._save_evaluation_results(metrics)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state': self.model_state,
                'matrices': self.matrices,
                'config': self.config
            }
            
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def _save_error_state(self):
        """Save error state information"""
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'model_state': self.model_state,
                'last_metrics': getattr(self, '_last_metrics', None)
            }
            
            error_path = self.experiment['dirs']['simulation'] / "error_state.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving error state: {str(e)}")
