import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class InferenceUtils:
    """Utilities for active inference calculations"""
    
    @staticmethod
    def calculate_expected_states(qs: np.ndarray, B: np.ndarray, action: int) -> np.ndarray:
        """Calculate expected states given current state and specific action (slice of B)"""
        return B[:, :, action] @ qs
        
    @staticmethod
    def calculate_expected_free_energy(
        qs: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        action: int,
        use_states_info_gain: bool = True,
        gamma: float = 16.0
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate expected free energy for a given action
        
        Returns:
            Tuple[float, np.ndarray]: (policy_value, predicted_states)
        """
        # Get predicted states
        qs_predicted = InferenceUtils.calculate_expected_states(qs, B, action)
        
        # Calculate expected observations
        qo_predicted = A @ qs_predicted
        
        # Calculate utility (preference satisfaction)
        utility = qo_predicted @ C
        
        # Calculate policy value
        policy_value = utility
        if use_states_info_gain:
            info_gain = InferenceUtils.calculate_state_info_gain(
                posterior=qs_predicted,
                prior=qs
            )
            policy_value += info_gain
        
        return policy_value, qs_predicted
        
    @staticmethod
    def calculate_state_info_gain(posterior: np.ndarray, prior: np.ndarray) -> float:
        """Calculate KL divergence between posterior and prior"""
        eps = 1e-16  # Small constant for numerical stability
        posterior = np.clip(posterior, eps, 1.0)
        prior = np.clip(prior, eps, 1.0)
        return np.sum(posterior * np.log(posterior / prior))
        
    @staticmethod
    def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply softmax with temperature"""
        x = x / temperature
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum() 