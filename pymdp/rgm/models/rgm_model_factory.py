"""RGM Model Factory"""

import torch
import torch.nn as nn
from typing import Dict, List
from pathlib import Path

from rgm.utils.rgm_logging import RGMLogging

class RGMModelFactory:
    """Factory for creating RGM models."""
    
    @staticmethod
    def create_model(config: Dict, matrices: Dict[str, torch.Tensor]) -> nn.Module:
        """Create RGM model based on configuration."""
        hierarchy_levels = sum(1 for k in matrices if k.startswith('A'))
        
        model_config = {
            'input_size': matrices['A0'].size(0),
            'hierarchy_levels': hierarchy_levels,
            'latent_dims': config.get('latent_dims', [64, 32, 16]),
            'activation': config.get('activation', 'relu'),
            'normalization': config.get('normalization', 'batch_norm'),
            'dropout_rate': config.get('dropout_rate', 0.1)
        }
        
        return RGMModel(model_config, matrices) 