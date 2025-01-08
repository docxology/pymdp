"""
RGM Model Initializer
====================

Handles model state initialization and validation.
Ensures proper setup of beliefs, learning parameters, and metrics.
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from rgm_experiment_utils import RGMExperimentUtils
from rgm_validation_utils import RGMValidationUtils

class RGMModelInitializer:
    """Initializes and validates RGM model state"""
    
    def __init__(self):
        """Initialize model initializer"""
        self.logger = RGMExperimentUtils.get_logger('model_init')
        self.validator = RGMValidationUtils()
        
    def initialize_model_state(self, config: Dict) -> Dict:
        """
        Initialize complete model state.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Dictionary containing initialized model state
        """
        try:
            self.logger.info("Initializing model state...")
            
            # Get hierarchy configuration
            hierarchy = config['model']['hierarchy']
            n_levels = hierarchy['n_levels']
            
            # Initialize state components
            state = {
                'current_level': 0,
                'iteration': 0,
                'beliefs': self._initialize_beliefs(hierarchy),
                'learning': self._initialize_learning_params(config['learning']),
                'metrics': self._initialize_metrics(),
                'validation': self._initialize_validation_state()
            }
            
            # Validate initial state
            if not self._validate_initial_state(state):
                raise ValueError("Invalid initial model state")
                
            self.logger.info(f"Initialized model state with {n_levels} levels")
            return state
            
        except Exception as e:
            self.logger.error(f"Error initializing model state: {str(e)}")
            raise
            
    def _initialize_beliefs(self, hierarchy: Dict) -> Dict:
        """Initialize belief states for all levels"""
        try:
            beliefs = {
                'states': [],
                'factors': [],
                'policies': []
            }
            
            # Initialize each level
            for level in range(hierarchy['n_levels']):
                level_key = f'level{level}'
                if level_key not in hierarchy['dimensions']:
                    raise ValueError(f"Missing configuration for {level_key}")
                    
                level_config = hierarchy['dimensions'][level_key]
                if 'state' not in level_config or 'factor' not in level_config:
                    raise ValueError(f"Missing state/factor dimensions for {level_key}")
                    
                # Get dimensions
                state_dim = level_config['state']
                factor_dim = level_config['factor']
                policy_dim = level_config.get('policy', 1)  # Default to 1 if not specified
                
                # Initialize with uniform distributions
                beliefs['states'].append(np.ones(state_dim) / state_dim)
                beliefs['factors'].append(np.ones(factor_dim) / factor_dim)
                beliefs['policies'].append(np.ones(policy_dim) / policy_dim)
                
                self.logger.info(
                    f"Initialized level {level} beliefs: "
                    f"states({state_dim}), factors({factor_dim}), policies({policy_dim})"
                )
                
            return beliefs
            
        except Exception as e:
            self.logger.error(f"Error initializing beliefs: {str(e)}")
            raise
            
    def _initialize_learning_params(self, learning_config: Dict) -> Dict:
        """Initialize learning parameters"""
        try:
            params = {
                'precision': learning_config['precision_init'],
                'learning_rate': learning_config['learning_rate'],
                'beta': learning_config['active_learning']['beta'],
                'max_precision': learning_config['active_learning']['max_precision'],
                'convergence_threshold': learning_config['message_passing']['convergence_threshold'],
                'histories': {
                    'elbo': [],
                    'accuracy': [],
                    'precision': []
                }
            }
            
            # Add regularization parameters if specified
            if 'regularization' in learning_config:
                reg_config = learning_config['regularization']
                params.update({
                    'l1_weight': reg_config.get('l1_weight', 0.0),
                    'l2_weight': reg_config.get('l2_weight', 0.0),
                    'dropout_rate': reg_config.get('dropout_rate', 0.0)
                })
                
            return params
            
        except Exception as e:
            self.logger.error(f"Error initializing learning parameters: {str(e)}")
            raise
            
    def _initialize_metrics(self) -> Dict:
        """Initialize tracking metrics"""
        try:
            return {
                'confusion_matrix': np.zeros((10, 10)),
                'class_accuracies': np.zeros(10),
                'class_counts': np.zeros(10),
                'total_accuracy': 0.0,
                'elbo': 0.0,
                'training_iterations': 0,
                'convergence_point': None,
                'best_accuracy': 0.0,
                'best_elbo': float('-inf'),
                'validation_history': []
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {str(e)}")
            raise
            
    def _initialize_validation_state(self) -> Dict:
        """Initialize validation state tracking"""
        try:
            return {
                'last_validated': None,
                'validation_frequency': 100,
                'validation_metrics': [],
                'warnings': [],
                'errors': [],
                'checkpoints': []
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing validation state: {str(e)}")
            raise
            
    def _validate_initial_state(self, state: Dict) -> bool:
        """Validate initial model state"""
        try:
            # Check belief normalization
            for level in range(len(state['beliefs']['states'])):
                states = state['beliefs']['states'][level]
                factors = state['beliefs']['factors'][level]
                policies = state['beliefs']['policies'][level]
                
                if not np.allclose(np.sum(states), 1.0):
                    self.logger.error(f"State beliefs not normalized at level {level}")
                    return False
                    
                if not np.allclose(np.sum(factors), 1.0):
                    self.logger.error(f"Factor beliefs not normalized at level {level}")
                    return False
                    
                if not np.allclose(np.sum(policies), 1.0):
                    self.logger.error(f"Policy beliefs not normalized at level {level}")
                    return False
                    
            # Check learning parameters
            learning = state['learning']
            if learning['precision'] <= 0:
                self.logger.error("Invalid precision value")
                return False
                
            if learning['learning_rate'] <= 0:
                self.logger.error("Invalid learning rate")
                return False
                
            # Check metrics initialization
            metrics = state['metrics']
            if not all(key in metrics for key in [
                'confusion_matrix', 'class_accuracies', 'total_accuracy'
            ]):
                self.logger.error("Missing required metrics")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating initial state: {str(e)}")
            return False 