"""
RGM Core Utilities
=================

Core utility functions for RGM implementation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils

class RGMCoreUtils:
    """Core utility functions for RGM"""
    
    def __init__(self):
        """Initialize core utilities"""
        self.logger = RGMExperimentUtils.get_logger('core_utils')
        
    @staticmethod
    def ensure_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
        
    @staticmethod
    def ensure_tensor(x: Union[np.ndarray, torch.Tensor], 
                     device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert input to torch tensor"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if device is not None:
            x = x.to(device)
        return x
        
    @staticmethod
    def normalize_array(x: np.ndarray, axis: Optional[int] = None, 
                       eps: float = 1e-12) -> np.ndarray:
        """Normalize array along specified axis"""
        x = x + eps  # Add small constant for stability
        if axis is None:
            return x / np.sum(x)
        return x / np.sum(x, axis=axis, keepdims=True)
        
    @staticmethod
    def compute_entropy(distribution: np.ndarray, eps: float = 1e-12) -> float:
        """Compute entropy of distribution"""
        return -np.sum(distribution * np.log(distribution + eps))
        
    @staticmethod
    def compute_kl_divergence(p: np.ndarray, q: np.ndarray, 
                            eps: float = 1e-12) -> float:
        """Compute KL divergence between distributions"""
        p = p + eps
        q = q + eps
        return np.sum(p * np.log(p / q))
        
    @staticmethod
    def compute_cross_entropy(p: np.ndarray, q: np.ndarray, 
                            eps: float = 1e-12) -> float:
        """Compute cross entropy between distributions"""
        q = q + eps
        return -np.sum(p * np.log(q))
        
    @staticmethod
    def compute_mutual_information(joint: np.ndarray, eps: float = 1e-12) -> float:
        """Compute mutual information from joint distribution"""
        # Add small constant for stability
        joint = joint + eps
        
        # Compute marginals
        p_x = np.sum(joint, axis=1)
        p_y = np.sum(joint, axis=0)
        
        # Compute entropies
        h_x = -np.sum(p_x * np.log(p_x))
        h_y = -np.sum(p_y * np.log(p_y))
        h_xy = -np.sum(joint * np.log(joint))
        
        return h_x + h_y - h_xy
        
    @staticmethod
    def compute_precision_weighted_error(prediction: np.ndarray,
                                       target: np.ndarray,
                                       precision: float) -> float:
        """Compute precision-weighted prediction error"""
        error = target - prediction
        return float(precision * np.sum(error ** 2))
        
    @staticmethod
    def compute_expected_free_energy(beliefs: Dict[str, List[np.ndarray]],
                                   matrices: Dict[str, np.ndarray]) -> float:
        """Compute expected free energy"""
        efe = 0.0
        
        # Add complexity terms (KL from prior)
        for level in range(len(beliefs['states'])):
            state = beliefs['states'][level]
            prior = matrices[f'D{level}']
            efe += RGMCoreUtils.compute_kl_divergence(state, prior)
            
        # Add accuracy terms (expected log likelihood)
        for level in range(len(beliefs['states']) - 1):
            curr_state = beliefs['states'][level]
            next_state = beliefs['states'][level + 1]
            A = matrices[f'A{level}']
            
            expected_obs = np.dot(A, curr_state)
            efe -= np.sum(next_state * np.log(expected_obs + 1e-12))
            
        return float(efe)
        
    @staticmethod
    def compute_belief_update(current: np.ndarray,
                            message: np.ndarray,
                            learning_rate: float) -> np.ndarray:
        """Compute belief update with learning rate"""
        updated = (1 - learning_rate) * current + learning_rate * message
        return RGMCoreUtils.normalize_array(updated)
        
    @staticmethod
    def compute_precision_update(current: float,
                               error: float,
                               beta: float,
                               max_precision: float) -> float:
        """Update precision based on prediction error"""
        updated = current * (1.0 + beta * error)
        return float(np.clip(updated, 0.0, max_precision))
        
    @staticmethod
    def check_array_properties(x: np.ndarray,
                             name: str = "array") -> Tuple[bool, List[str]]:
        """Check basic array properties"""
        messages = []
        
        # Check for NaN/Inf values
        if not np.all(np.isfinite(x)):
            messages.append(f"{name} contains NaN/Inf values")
            
        # Check non-negativity
        if np.any(x < 0):
            messages.append(f"{name} contains negative values")
            
        # Check normalization for probability distributions
        if x.ndim == 1 or x.ndim == 2:
            if not np.allclose(np.sum(x), 1.0, rtol=1e-5):
                messages.append(f"{name} not normalized")
                
        return len(messages) == 0, messages
        
    @staticmethod
    def save_array_info(array: np.ndarray,
                       name: str,
                       save_dir: Path) -> Dict:
        """Save array information"""
        info = {
            'name': name,
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'min': float(np.min(array)),
            'max': float(np.max(array)),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'sparsity': float(np.mean(array == 0)),
            'properties': {
                'finite': bool(np.all(np.isfinite(array))),
                'non_negative': bool(np.all(array >= 0)),
                'normalized': bool(np.allclose(np.sum(array), 1.0, rtol=1e-5))
            }
        }
        
        # Save info
        info_path = save_dir / f"{name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        return info