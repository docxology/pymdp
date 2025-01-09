"""
Base RGM Model Implementation
===========================

This module provides the base implementation of the Recursive Generative Model (RGM).
It defines the core functionality for hierarchical message passing, matrix operations,
and state management.

The base model implements:
1. Hierarchical message passing
2. Matrix factorization
3. State management
4. Basic inference
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

class RGMBase:
    """Base class for Recursive Generative Models"""
    
    def __init__(self, config: Dict):
        """
        Initialize RGM base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.logger = logging.getLogger('rgm.model')
        self.config = config
        
        # Extract model parameters
        self.n_levels = config['model']['hierarchy']['n_levels']
        self.dimensions = config['model']['hierarchy']['dimensions']
        self.activation = config['model']['hierarchy']['activation']
        
        # Initialize matrices
        self.matrices = self._initialize_matrices()
        
        # Initialize states
        self.states = self._initialize_states()
        
    def _initialize_matrices(self) -> Dict[str, np.ndarray]:
        """
        Initialize model matrices.
        
        Returns:
            Dictionary of initialized matrices
        """
        matrices = {}
        
        try:
            # Initialize hierarchy matrices
            for level in range(self.n_levels - 1):
                dim_lower = self.dimensions[level]
                dim_upper = self.dimensions[level + 1]
                
                # Forward matrices (bottom-up)
                matrices[f'A_{level}'] = np.random.randn(dim_upper, dim_lower) * 0.01
                
                # Backward matrices (top-down)
                matrices[f'B_{level}'] = np.random.randn(dim_lower, dim_upper) * 0.01
                
                # State matrices
                matrices[f'D_{level}'] = np.eye(dim_lower)
                matrices[f'E_{level}'] = np.eye(dim_upper)
                
            self.logger.info(f"Initialized {len(matrices)} matrices")
            return matrices
            
        except Exception as e:
            self.logger.error(f"Failed to initialize matrices: {str(e)}")
            raise
            
    def _initialize_states(self) -> Dict[str, np.ndarray]:
        """
        Initialize model states.
        
        Returns:
            Dictionary of initialized states
        """
        states = {}
        
        try:
            # Initialize states for each level
            for level in range(self.n_levels):
                dim = self.dimensions[level]
                states[f'mu_{level}'] = np.zeros(dim)
                states[f'sigma_{level}'] = np.ones(dim)
                
            self.logger.info(f"Initialized states for {self.n_levels} levels")
            return states
            
        except Exception as e:
            self.logger.error(f"Failed to initialize states: {str(e)}")
            raise
            
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            return x
            
    def forward_pass(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Perform forward pass through the hierarchy.
        
        Args:
            x: Input data
            
        Returns:
            List of activations at each level
        """
        try:
            activations = [x]
            current = x
            
            # Propagate up through hierarchy
            for level in range(self.n_levels - 1):
                # Get forward matrix
                A = self.matrices[f'A_{level}']
                
                # Compute activation
                current = self._apply_activation(A @ current)
                activations.append(current)
                
            return activations
            
        except Exception as e:
            self.logger.error(f"Forward pass failed: {str(e)}")
            raise
            
    def backward_pass(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Perform backward pass through the hierarchy.
        
        Args:
            y: Top-level state
            
        Returns:
            List of reconstructions at each level
        """
        try:
            reconstructions = [y]
            current = y
            
            # Propagate down through hierarchy
            for level in range(self.n_levels - 2, -1, -1):
                # Get backward matrix
                B = self.matrices[f'B_{level}']
                
                # Compute reconstruction
                current = self._apply_activation(B @ current)
                reconstructions.append(current)
                
            return reconstructions[::-1]  # Reverse to match forward order
            
        except Exception as e:
            self.logger.error(f"Backward pass failed: {str(e)}")
            raise
            
    def update_states(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update model states based on input.
        
        Args:
            x: Input data
            
        Returns:
            Updated states dictionary
        """
        try:
            # Forward pass
            activations = self.forward_pass(x)
            
            # Update means
            for level, activation in enumerate(activations):
                self.states[f'mu_{level}'] = activation
                
            # Backward pass
            reconstructions = self.backward_pass(activations[-1])
            
            # Update variances based on reconstruction error
            for level in range(self.n_levels):
                error = activations[level] - reconstructions[level]
                self.states[f'sigma_{level}'] = np.abs(error) + 1e-6
                
            return self.states
            
        except Exception as e:
            self.logger.error(f"State update failed: {str(e)}")
            raise
            
    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'matrices': self.matrices,
                'states': self.states,
                'config': self.config
            }
            np.save(path, checkpoint)
            self.logger.info(f"Saved checkpoint to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        try:
            checkpoint = np.load(path, allow_pickle=True).item()
            self.matrices = checkpoint['matrices']
            self.states = checkpoint['states']
            self.config = checkpoint['config']
            self.logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise 