import numpy as np
import logging
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path

# Import utilities
from .utils.matrix_utils import MatrixUtils
from .utils.inference_utils import InferenceUtils
from .utils.logging_utils import setup_experiment_logging

logger = logging.getLogger(__name__)

class ActiveInferenceGNN:
    """Active inference agent for GNN-based models"""
    
    def __init__(self, 
                 A: np.ndarray,
                 B: np.ndarray,
                 C: np.ndarray,
                 D: np.ndarray,
                 control_fac_idx: List[int],
                 policy_len: int = 1,
                 num_states: Optional[List[int]] = None,
                 num_controls: Optional[List[int]] = None,
                 gamma: float = 16.0,
                 use_states_info_gain: bool = True):
        """Initialize active inference agent"""
        try:
            # Store raw matrices
            self.A_raw = A.copy()
            self.B_raw = B.copy()
            self.C_raw = C.copy()
            self.D_raw = D.copy()
            
            # Log matrix shapes
            logger.info("Initializing active inference agent with matrices:")
            logger.info(f"A shape: {A.shape}")
            logger.info(f"B shape: {B.shape}")
            logger.info(f"C shape: {C.shape}")
            logger.info(f"D shape: {D.shape}")
            
            # Normalize matrices using utility
            matrices = {
                'A': A,
                'B': B,
                'C': C,
                'D': D
            }
            normalized = MatrixUtils.normalize_matrices(matrices)
            
            self.A = normalized['A']
            self.B = normalized['B']
            self.C = normalized['C']
            self.D = normalized['D']
            
            # Store parameters
            self.control_fac_idx = control_fac_idx
            self.policy_len = policy_len
            self.num_states = num_states or [A.shape[1]]
            self.num_controls = num_controls or [B.shape[-1]]
            self.gamma = gamma
            self.use_states_info_gain = use_states_info_gain
            
            # Initialize beliefs
            self.qs = self.D.copy()  # Current beliefs about states
            self.qpi = None  # Current policy beliefs
            
            # Initialize history tracking
            self.history = {
                'beliefs': [],
                'observations': [],
                'actions': [],
                'policy_probs': []
            }
            
            logger.info("Active inference agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing active inference agent: {str(e)}")
            raise
            
    def infer_states(self, obs: int) -> np.ndarray:
        """Update beliefs about states using observation"""
        try:
            # Get likelihood for observation
            likelihood = self.A[obs, :]
            
            # Multiply by current beliefs and normalize
            posterior = likelihood * self.qs
            posterior = posterior / posterior.sum() if posterior.sum() > 0 else self.D.copy()
            
            # Update beliefs and track history
            self.qs = posterior
            self.history['beliefs'].append(posterior.copy())
            self.history['observations'].append(obs)
            
            logger.debug(f"Updated beliefs: {posterior}")
            return posterior
            
        except Exception as e:
            logger.error(f"Error in state inference: {str(e)}")
            raise
            
    def infer_policies(self) -> np.ndarray:
        """Evaluate policies using expected free energy"""
        try:
            num_policies = self.num_controls[0]
            policy_values = np.zeros(num_policies)
            
            # Calculate expected free energy for each policy
            for policy_idx in range(num_policies):
                policy_value, _ = InferenceUtils.calculate_expected_free_energy(
                    qs=self.qs,
                    A=self.A,
                    B=self.B,
                    C=self.C,
                    action=policy_idx,
                    use_states_info_gain=self.use_states_info_gain,
                    gamma=self.gamma
                )
                policy_values[policy_idx] = policy_value
                    
            # Convert to policy distribution using softmax
            policy_probs = InferenceUtils.softmax(policy_values, temperature=self.gamma)
            
            # Store and track
            self.qpi = policy_probs
            self.history['policy_probs'].append(policy_probs.copy())
            
            logger.debug(f"Policy probabilities: {policy_probs}")
            return policy_probs
            
        except Exception as e:
            logger.error(f"Error in policy inference: {str(e)}")
            raise
            
    def sample_action(self) -> int:
        """Sample action from policy distribution"""
        try:
            # Get policy probabilities if not already computed
            if self.qpi is None:
                self.qpi = self.infer_policies()
                
            # Sample action
            action = np.random.choice(len(self.qpi), p=self.qpi)
            
            # Track history
            self.history['actions'].append(action)
            
            logger.debug(f"Selected action: {action}")
            return action
            
        except Exception as e:
            logger.error(f"Error sampling action: {str(e)}")
            raise
            
    def get_history(self) -> Dict:
        """Get agent history"""
        return {
            'beliefs': np.array(self.history['beliefs']),
            'observations': np.array(self.history['observations']),
            'actions': np.array(self.history['actions']),
            'policy_probs': np.array(self.history['policy_probs'])
        }
        
    def reset(self):
        """Reset agent to initial state"""
        self.qs = self.D.copy()
        self.qpi = None
        self.history = {
            'beliefs': [],
            'observations': [],
            'actions': [],
            'policy_probs': []
        }

