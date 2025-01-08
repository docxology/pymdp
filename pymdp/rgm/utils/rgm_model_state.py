"""
RGM Model State
==============

Manages RGM model state and belief updates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .rgm_experiment_utils import RGMExperimentUtils
from .rgm_core_utils import RGMCoreUtils
from .rgm_matrix_normalizer import RGMMatrixNormalizer

class RGMModelState:
    """Manages RGM model state"""
    
    def __init__(self, config: Dict):
        """
        Initialize model state.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = RGMExperimentUtils.get_logger('model_state')
        self.config = config
        self.core = RGMCoreUtils()
        self.normalizer = RGMMatrixNormalizer()
        
        # Initialize state components
        self.beliefs = self._initialize_beliefs()
        self.learning = self._initialize_learning()
        self.metrics = self._initialize_metrics()
        
    def _initialize_beliefs(self) -> Dict[str, List[np.ndarray]]:
        """Initialize belief states"""
        try:
            hierarchy = self.config['model']['hierarchy']
            beliefs = {
                'states': [],
                'factors': []
            }
            
            # Initialize each level
            for level in range(hierarchy['n_levels']):
                level_config = hierarchy['dimensions'][f'level{level}']
                
                # Initialize with uniform distributions
                state_dim = level_config['state']
                factor_dim = level_config['factor']
                
                beliefs['states'].append(np.ones(state_dim) / state_dim)
                beliefs['factors'].append(np.ones(factor_dim) / factor_dim)
                
            return beliefs
            
        except Exception as e:
            self.logger.error(f"Error initializing beliefs: {str(e)}")
            raise