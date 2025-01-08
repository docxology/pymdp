"""
RGM Configuration Manager
========================

Manages RGM model configuration including:
1. Default configuration generation
2. Configuration validation
3. Hierarchy setup
4. Learning parameters
5. Data configuration
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from rgm_experiment_utils import RGMExperimentUtils

class RGMConfigManager:
    """Manages RGM configuration and validation"""
    
    def __init__(self):
        """Initialize configuration manager"""
        self.logger = RGMExperimentUtils.get_logger('config')
        self.default_config_path = Path(__file__).parent.parent / "configs" / "default_config.json"
        
    def get_default_config(self) -> Dict:
        """Load default configuration"""
        try:
            if not self.default_config_path.exists():
                self.logger.error(f"Default config not found at {self.default_config_path}")
                raise FileNotFoundError(f"Default config not found")
                
            with open(self.default_config_path) as f:
                config = json.load(f)
                
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading default config: {str(e)}")
            raise