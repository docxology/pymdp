"""
RGM Metrics Utilities
===================

Handles metrics computation and tracking for RGM.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_core_utils import RGMCoreUtils

class RGMMetricsUtils:
    """Handles metrics computation and tracking"""
    
    def __init__(self):
        """Initialize metrics utilities"""
        self.logger = RGMExperimentUtils.get_logger('metrics')
        self.experiment = RGMExperimentUtils.get_experiment()
        self.core = RGMCoreUtils()
        
        # Initialize metrics storage
        self.metrics = {
            'training': self._initialize_training_metrics(),
            'validation': self._initialize_validation_metrics(),
            'test': self._initialize_test_metrics()
        }
        
    def _initialize_training_metrics(self) -> Dict:
        """Initialize training metrics"""
        return {
            'epoch_metrics': [],
            'batch_metrics': [],
            'learning_curves': {
                'elbo': [],
                'accuracy': [],
                'precision': [],
                'loss': []
            },
            'convergence': {
                'converged': False,
                'epoch': None,
                'metric': None,
                'value': None
            }
        }
        
    def _initialize_validation_metrics(self) -> Dict:
        """Initialize validation metrics"""
        return {
            'best_metrics': {
                'accuracy': 0.0,
                'elbo': float('-inf'),
                'epoch': None
            },
            'history': [],
            'early_stopping': {
                'best_value': float('-inf'),
                'patience_counter': 0,
                'should_stop': False
            }
        }
        
    def _initialize_test_metrics(self) -> Dict:
        """Initialize test metrics"""
        return {
            'confusion_matrix': np.zeros((10, 10)),
            'class_metrics': {
                'precision': np.zeros(10),
                'recall': np.zeros(10),
                'f1_score': np.zeros(10)
            },
            'overall_metrics': {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0
            }
        }
        
    def update_training_metrics(self, epoch_metrics: Dict, batch_metrics: Dict):
        """Update training metrics"""
        try:
            # Update epoch metrics
            self.metrics['training']['epoch_metrics'].append(epoch_metrics)
            
            # Update learning curves
            for key in ['elbo', 'accuracy', 'precision', 'loss']:
                if key in epoch_metrics:
                    self.metrics['training']['learning_curves'][key].append(
                        epoch_metrics[key]
                    )
                    
            # Check convergence
            if self._check_convergence(epoch_metrics):
                self.metrics['training']['convergence']['converged'] = True
                self.metrics['training']['convergence']['epoch'] = len(
                    self.metrics['training']['epoch_metrics']
                )
                
            # Save metrics
            self._save_training_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating training metrics: {str(e)}")
            raise
            
    def update_validation_metrics(self, val_metrics: Dict, epoch: int):
        """Update validation metrics"""
        try:
            # Add to history
            self.metrics['validation']['history'].append(val_metrics)
            
            # Update best metrics
            if val_metrics['accuracy'] > self.metrics['validation']['best_metrics']['accuracy']:
                self.metrics['validation']['best_metrics'].update({
                    'accuracy': val_metrics['accuracy'],
                    'elbo': val_metrics['elbo'],
                    'epoch': epoch
                })
                
            # Check early stopping
            self._check_early_stopping(val_metrics['accuracy'])
            
            # Save metrics
            self._save_validation_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating validation metrics: {str(e)}")
            raise
            
    def update_test_metrics(self, predictions: np.ndarray, targets: np.ndarray):
        """Update test metrics"""
        try:
            # Update confusion matrix
            self.metrics['test']['confusion_matrix'] += self._compute_confusion_matrix(
                predictions, targets
            )
            
            # Update class metrics
            precision, recall, f1 = self._compute_class_metrics(
                self.metrics['test']['confusion_matrix']
            )
            
            self.metrics['test']['class_metrics'].update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Update overall metrics
            self.metrics['test']['overall_metrics'].update({
                'accuracy': np.trace(self.metrics['test']['confusion_matrix']) / 
                           np.sum(self.metrics['test']['confusion_matrix']),
                'macro_f1': np.mean(f1),
                'weighted_f1': np.average(
                    f1,
                    weights=np.sum(self.metrics['test']['confusion_matrix'], axis=1)
                )
            })
            
            # Save metrics
            self._save_test_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating test metrics: {str(e)}")
            raise
            
    def compute_elbo(self, beliefs: Dict[str, List[np.ndarray]], 
                    matrices: Dict[str, np.ndarray]) -> float:
        """Compute Evidence Lower BOund"""
        try:
            elbo = 0.0
            
            # Add state terms
            for level in range(len(beliefs['states'])):
                state = beliefs['states'][level]
                elbo += self.compute_entropy(state)  # Entropy
                
                # Add likelihood if not top level
                if level < len(beliefs['states']) - 1:
                    A = matrices[f'A{level}']
                    higher_state = beliefs['states'][level + 1]
                    likelihood = np.dot(A.T, higher_state)
                    elbo += np.sum(state * np.log(likelihood + 1e-12))
                    
            # Add factor terms
            for level in range(len(beliefs['factors'])):
                factor = beliefs['factors'][level]
                elbo += self.compute_entropy(factor)  # Entropy
                
                B = matrices[f'B{level}']
                D = matrices[f'D{level}']
                likelihood = np.dot(B.T, factor) * D
                elbo += np.sum(factor * np.log(likelihood + 1e-12))
                
            return float(elbo)
            
        except Exception as e:
            self.logger.error(f"Error computing ELBO: {str(e)}")
            raise
            
    def compute_entropy(self, distribution: np.ndarray, eps: float = 1e-12) -> float:
        """Compute entropy of distribution"""
        try:
            return -np.sum(distribution * np.log(distribution + eps))
        except Exception as e:
            self.logger.error(f"Error computing entropy: {str(e)}")
            raise
            
    def _compute_confusion_matrix(self, predictions: np.ndarray, 
                                targets: np.ndarray) -> np.ndarray:
        """Compute confusion matrix"""
        try:
            n_classes = 10
            cm = np.zeros((n_classes, n_classes))
            for pred, target in zip(predictions, targets):
                cm[target, pred] += 1
            return cm
        except Exception as e:
            self.logger.error(f"Error computing confusion matrix: {str(e)}")
            raise
            
    def _compute_class_metrics(self, confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-class precision, recall, and F1 score"""
        try:
            # Compute true positives, false positives, false negatives
            tp = np.diag(confusion_matrix)
            fp = np.sum(confusion_matrix, axis=0) - tp
            fn = np.sum(confusion_matrix, axis=1) - tp
            
            # Compute metrics
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
            
            return precision, recall, f1
            
        except Exception as e:
            self.logger.error(f"Error computing class metrics: {str(e)}")
            raise
            
    def _check_convergence(self, metrics: Dict) -> bool:
        """Check convergence criteria"""
        try:
            # Get convergence history
            elbo_history = self.metrics['training']['learning_curves']['elbo']
            if len(elbo_history) < 5:
                return False
                
            # Check if ELBO has stabilized
            recent_elbos = elbo_history[-5:]
            elbo_std = np.std(recent_elbos)
            
            return elbo_std < 1e-4
            
        except Exception as e:
            self.logger.error(f"Error checking convergence: {str(e)}")
            raise
            
    def _check_early_stopping(self, current_value: float):
        """Check early stopping criteria"""
        try:
            early_stopping = self.metrics['validation']['early_stopping']
            
            if current_value > early_stopping['best_value']:
                early_stopping['best_value'] = current_value
                early_stopping['patience_counter'] = 0
            else:
                early_stopping['patience_counter'] += 1
                
            if early_stopping['patience_counter'] >= 5:  # Patience threshold
                early_stopping['should_stop'] = True
                
        except Exception as e:
            self.logger.error(f"Error checking early stopping: {str(e)}")
            raise
            
    def _save_training_metrics(self):
        """Save training metrics"""
        try:
            metrics_dir = self.experiment['dirs']['results'] / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            with open(metrics_dir / "training_metrics.json", 'w') as f:
                json.dump(self.metrics['training'], f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            self.logger.error(f"Error saving training metrics: {str(e)}")
            raise
            
    def _save_validation_metrics(self):
        """Save validation metrics"""
        try:
            metrics_dir = self.experiment['dirs']['results'] / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            with open(metrics_dir / "validation_metrics.json", 'w') as f:
                json.dump(self.metrics['validation'], f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            self.logger.error(f"Error saving validation metrics: {str(e)}")
            raise
            
    def _save_test_metrics(self):
        """Save test metrics"""
        try:
            metrics_dir = self.experiment['dirs']['results'] / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            with open(metrics_dir / "test_metrics.json", 'w') as f:
                json.dump(self.metrics['test'], f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            self.logger.error(f"Error saving test metrics: {str(e)}")
            raise

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)