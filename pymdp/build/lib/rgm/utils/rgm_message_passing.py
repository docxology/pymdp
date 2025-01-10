"""
RGM Message Passing
==================

Implements message passing algorithms for RGM inference.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_core_utils import RGMCoreUtils
from .rgm_matrix_normalizer import RGMMatrixNormalizer

class RGMMessagePassing:
    """Handles message passing for RGM inference"""
    
    def __init__(self, config: Dict):
        """
        Initialize message passing.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = RGMExperimentUtils.get_logger('message_passing')
        self.config = config['learning']['message_passing']
        self.core = RGMCoreUtils()
        self.normalizer = RGMMatrixNormalizer()
        
        # Get parameters
        self.max_iterations = self.config['max_iterations']
        self.convergence_threshold = self.config['convergence_threshold']
        self.damping_factor = self.config.get('damping_factor', 0.9)
        
    def run_message_passing(self,
                          beliefs: Dict[str, List[np.ndarray]],
                          matrices: Dict[str, np.ndarray],
                          precision: float) -> Tuple[Dict[str, List[np.ndarray]], bool]:
        """
        Run complete message passing iteration.
        
        Args:
            beliefs: Initial belief state dictionary
            matrices: Dictionary of transition matrices
            precision: Current precision value
            
        Returns:
            Tuple of (updated beliefs, convergence flag)
        """
        try:
            current_beliefs = {
                'states': [s.copy() for s in beliefs['states']],
                'factors': [f.copy() for f in beliefs['factors']]
            }
            
            for iteration in range(self.max_iterations):
                # Store previous beliefs
                previous_beliefs = {
                    'states': [s.copy() for s in current_beliefs['states']],
                    'factors': [f.copy() for f in current_beliefs['factors']]
                }
                
                # Bottom-up pass
                current_beliefs = self._bottom_up_pass(
                    current_beliefs,
                    matrices,
                    precision
                )
                
                # Top-down pass
                current_beliefs = self._top_down_pass(
                    current_beliefs,
                    matrices,
                    precision
                )
                
                # Check convergence
                if self._check_convergence(previous_beliefs, current_beliefs):
                    self.logger.info(f"Converged after {iteration+1} iterations")
                    return current_beliefs, True
                    
            self.logger.warning("Message passing did not converge")
            return current_beliefs, False
            
        except Exception as e:
            self.logger.error(f"Error in message passing: {str(e)}")
            raise
            
    def _bottom_up_pass(self,
                       beliefs: Dict[str, List[np.ndarray]],
                       matrices: Dict[str, np.ndarray],
                       precision: float) -> Dict[str, List[np.ndarray]]:
        """Perform bottom-up message passing"""
        try:
            updated_beliefs = {
                'states': [s.copy() for s in beliefs['states']],
                'factors': [f.copy() for f in beliefs['factors']]
            }
            
            # Process each level
            for level in range(len(beliefs['states'])):
                # Update state beliefs
                if level == 0:
                    # Process input at lowest level
                    state_msg = self._compute_input_message(
                        matrices['input'],
                        matrices[f'A{level}'],
                        precision
                    )
                else:
                    # Process state message from level below
                    state_msg = self._compute_state_message(
                        updated_beliefs['states'][level-1],
                        matrices[f'A{level}'],
                        precision
                    )
                    
                updated_beliefs['states'][level] = self._update_belief(
                    updated_beliefs['states'][level],
                    state_msg
                )
                
                # Update factor beliefs
                factor_msg = self._compute_factor_message(
                    updated_beliefs['states'][level],
                    matrices[f'B{level}'],
                    matrices[f'D{level}']
                )
                
                updated_beliefs['factors'][level] = self._update_belief(
                    updated_beliefs['factors'][level],
                    factor_msg
                )
                
            return updated_beliefs
            
        except Exception as e:
            self.logger.error(f"Error in bottom-up pass: {str(e)}")
            raise
            
    def _top_down_pass(self,
                      beliefs: Dict[str, List[np.ndarray]],
                      matrices: Dict[str, np.ndarray],
                      precision: float) -> Dict[str, List[np.ndarray]]:
        """Perform top-down message passing"""
        try:
            updated_beliefs = {
                'states': [s.copy() for s in beliefs['states']],
                'factors': [f.copy() for f in beliefs['factors']]
            }
            
            # Process each level from top down
            for level in reversed(range(len(beliefs['states']) - 1)):
                # Update state beliefs
                state_pred = self._compute_state_prediction(
                    updated_beliefs['states'][level + 1],
                    matrices[f'A{level}'].T,
                    precision
                )
                
                updated_beliefs['states'][level] = self._update_belief(
                    updated_beliefs['states'][level],
                    state_pred
                )
                
                # Update factor beliefs
                factor_pred = self._compute_factor_prediction(
                    updated_beliefs['factors'][level + 1],
                    matrices[f'B{level}'].T,
                    matrices[f'D{level}']
                )
                
                updated_beliefs['factors'][level] = self._update_belief(
                    updated_beliefs['factors'][level],
                    factor_pred
                )
                
            return updated_beliefs
            
        except Exception as e:
            self.logger.error(f"Error in top-down pass: {str(e)}")
            raise
            
    def _compute_input_message(self,
                             input_data: np.ndarray,
                             A: np.ndarray,
                             precision: float) -> np.ndarray:
        """Compute message from input data"""
        try:
            # Compute likelihood
            likelihood = np.dot(A.T, input_data)
            
            # Apply precision
            likelihood = np.power(likelihood, precision)
            
            # Normalize
            return self._normalize(likelihood)
            
        except Exception as e:
            self.logger.error(f"Error computing input message: {str(e)}")
            raise
            
    def _compute_state_message(self,
                             lower_state: np.ndarray,
                             A: np.ndarray,
                             precision: float) -> np.ndarray:
        """Compute state belief message"""
        try:
            # Compute likelihood with normalized matrix
            likelihood = np.dot(self.normalizer.normalize_matrix(A.T), lower_state)
            
            # Apply precision
            likelihood = np.power(likelihood, precision)
            
            # Normalize
            return self._normalize(likelihood)
            
        except Exception as e:
            self.logger.error(f"Error computing state message: {str(e)}")
            raise
            
    def _compute_factor_message(self,
                              state: np.ndarray,
                              B: np.ndarray,
                              D: np.ndarray) -> np.ndarray:
        """Compute factor belief message"""
        try:
            # Compute likelihood
            likelihood = np.dot(B.T, state)
            
            # Apply prior
            likelihood *= D
            
            # Normalize
            return self._normalize(likelihood)
            
        except Exception as e:
            self.logger.error(f"Error computing factor message: {str(e)}")
            raise
            
    def _compute_state_prediction(self,
                                higher_state: np.ndarray,
                                A_transpose: np.ndarray,
                                precision: float) -> np.ndarray:
        """Compute top-down state prediction"""
        try:
            # Compute prediction
            prediction = np.dot(A_transpose, higher_state)
            
            # Apply precision
            prediction = np.power(prediction, precision)
            
            # Normalize
            return self._normalize(prediction)
            
        except Exception as e:
            self.logger.error(f"Error computing state prediction: {str(e)}")
            raise
            
    def _compute_factor_prediction(self,
                                 higher_factor: np.ndarray,
                                 B_transpose: np.ndarray,
                                 D: np.ndarray) -> np.ndarray:
        """Compute top-down factor prediction"""
        try:
            # Compute prediction
            prediction = np.dot(B_transpose, higher_factor)
            
            # Apply prior
            prediction *= D
            
            # Normalize
            return self._normalize(prediction)
            
        except Exception as e:
            self.logger.error(f"Error computing factor prediction: {str(e)}")
            raise
            
    def _update_belief(self,
                      current: np.ndarray,
                      message: np.ndarray) -> np.ndarray:
        """Update belief with new message"""
        try:
            # Apply damping
            updated = (1 - self.damping_factor) * current + self.damping_factor * message
            
            # Normalize
            return self._normalize(updated)
            
        except Exception as e:
            self.logger.error(f"Error updating belief: {str(e)}")
            raise
            
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to sum to 1"""
        try:
            # Add small constant for numerical stability
            x = x + 1e-12
            
            # Normalize
            return x / np.sum(x)
            
        except Exception as e:
            self.logger.error(f"Error normalizing array: {str(e)}")
            raise
            
    def _check_convergence(self,
                         previous: Dict[str, List[np.ndarray]],
                         current: Dict[str, List[np.ndarray]]) -> bool:
        """Check if beliefs have converged"""
        try:
            # Check state convergence
            for prev_state, curr_state in zip(previous['states'], current['states']):
                if np.max(np.abs(prev_state - curr_state)) > self.convergence_threshold:
                    return False
                    
            # Check factor convergence
            for prev_factor, curr_factor in zip(previous['factors'], current['factors']):
                if np.max(np.abs(prev_factor - curr_factor)) > self.convergence_threshold:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking convergence: {str(e)}")
            raise