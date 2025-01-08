"""
RGM Validation Utilities
=======================

Provides validation utilities for RGM pipeline components.
Ensures proper configuration and initialization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

class RGMValidationUtils:
    """Validation utilities for RGM pipeline"""
    
    @staticmethod
    def validate_experiment_structure(experiment: Dict) -> bool:
        """Validate experiment directory structure"""
        required_dirs = [
            'root',
            'logs',
            'render',
            'matrices',
            'config',
            'gnn_specs',
            'visualizations',
            'simulation',
            'results',
            'analysis'
        ]
        
        for dir_name in required_dirs:
            if dir_name not in experiment['dirs']:
                logging.error(f"Missing required directory: {dir_name}")
                return False
                
            if not experiment['dirs'][dir_name].exists():
                logging.error(f"Directory does not exist: {experiment['dirs'][dir_name]}")
                return False
                
        return True
        
    @staticmethod
    def validate_matrices(matrices: Dict, config: Dict) -> bool:
        """Validate matrix shapes and properties"""
        try:
            # Check required matrices
            required_matrices = [
                'A0', 'B0', 'D0',
                'A1', 'B1', 'D1',
                'A2', 'B2', 'D2',
                'A3', 'B3', 'D3',
                'E'
            ]
            
            for name in required_matrices:
                if name not in matrices:
                    logging.error(f"Missing required matrix: {name}")
                    return False
                    
            # Check shapes
            shapes = config['model']['matrix_shapes']
            for name, matrix in matrices.items():
                if name not in shapes:
                    logging.warning(f"No shape specification for matrix: {name}")
                    continue
                    
                expected = tuple(shapes[name])
                actual = matrix.shape
                
                if expected != actual:
                    logging.error(
                        f"Shape mismatch for {name}: expected {expected}, got {actual}"
                    )
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error validating matrices: {str(e)}")
            return False
            
    @staticmethod
    def validate_model_state(state: Dict) -> bool:
        """Validate model state"""
        try:
            # Check required components
            required = ['current_level', 'iteration', 'beliefs', 'learning', 'metrics']
            for key in required:
                if key not in state:
                    logging.error(f"Missing required state component: {key}")
                    return False
                    
            # Check beliefs
            beliefs = state['beliefs']
            if not all(k in beliefs for k in ['states', 'factors', 'policies']):
                logging.error("Missing belief components")
                return False
                
            # Check learning parameters
            learning = state['learning']
            required_learning = ['precision', 'learning_rate', 'beta']
            if not all(k in learning for k in required_learning):
                logging.error("Missing learning parameters")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating model state: {str(e)}")
            return False
            
    @staticmethod
    def validate_checkpoint(checkpoint: Dict) -> bool:
        """Validate checkpoint contents"""
        try:
            # Check required components
            required = ['epoch', 'model_state', 'matrices', 'config']
            if not all(k in checkpoint for k in required):
                logging.error("Invalid checkpoint format")
                return False
                
            # Validate model state
            if not RGMValidationUtils.validate_model_state(checkpoint['model_state']):
                return False
                
            # Validate matrices
            if not RGMValidationUtils.validate_matrices(
                checkpoint['matrices'],
                checkpoint['config']
            ):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating checkpoint: {str(e)}")
            return False