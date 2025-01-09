"""
RGM Configuration Loader
=======================

Loads and validates RGM configuration files.
Handles default configuration and user overrides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .rgm_experiment_utils import RGMExperimentUtils

class RGMConfigLoader:
    """Loads and validates RGM configuration"""
    
    def __init__(self):
        """Initialize configuration loader"""
        self.logger = RGMExperimentUtils.get_logger('config')
        
    def load_config(self) -> Dict:
        """
        Load and validate configuration.
        
        Returns:
            Dictionary containing validated configuration
        """
        try:
            # Get experiment state
            experiment = RGMExperimentUtils.get_experiment()
            
            # Load base configuration
            base_config = self._load_base_config()
            
            # Load user configuration if it exists
            user_config = self._load_user_config()
            
            # Merge configurations
            config = self._merge_configs(base_config, user_config)
            
            # Validate configuration
            self._validate_config(config)
            
            # Save merged configuration
            self._save_config(config, experiment['dirs']['root'])
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _load_base_config(self) -> Dict:
        """Load base configuration"""
        try:
            # Get base config path
            base_path = Path(__file__).parent.parent / "configs" / "rgm_base.json"
            
            if not base_path.exists():
                raise FileNotFoundError("Base configuration not found")
                
            # Load base config
            with open(base_path) as f:
                config = json.load(f)
                
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading base configuration: {str(e)}")
            raise
            
    def _load_user_config(self) -> Optional[Dict]:
        """Load user configuration if it exists"""
        try:
            # Get user config path
            user_path = Path(__file__).parent.parent / "configs" / "rgm_user.json"
            
            if not user_path.exists():
                return None
                
            # Load user config
            with open(user_path) as f:
                config = json.load(f)
                
            return config
            
        except Exception as e:
            self.logger.warning(f"Error loading user configuration: {str(e)}")
            return None
            
    def _merge_configs(self, base_config: Dict, user_config: Optional[Dict]) -> Dict:
        """Merge base and user configurations"""
        try:
            # Start with base configuration
            config = base_config.copy()
            
            # Merge user configuration if it exists
            if user_config:
                def deep_merge(d1: Dict, d2: Dict):
                    for k, v in d2.items():
                        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                            deep_merge(d1[k], v)
                        else:
                            d1[k] = v
                            
                deep_merge(config, user_config)
                
            return config
            
        except Exception as e:
            self.logger.error(f"Error merging configurations: {str(e)}")
            raise
            
    def _validate_config(self, config: Dict):
        """Validate configuration"""
        try:
            # Check required sections
            required_sections = [
                'data',
                'model',
                'training',
                'inference',
                'logging'
            ]
            
            missing = [s for s in required_sections if s not in config]
            if missing:
                raise ValueError(f"Missing required sections: {', '.join(missing)}")
                
            # Validate data configuration
            data_config = config['data']
            if not all(k in data_config for k in ['dataset', 'batch_size']):
                raise ValueError("Invalid data configuration")
                
            # Validate model configuration
            model_config = config['model']
            if not all(k in model_config for k in ['architecture', 'parameters']):
                raise ValueError("Invalid model configuration")
                
            # Validate training configuration
            training_config = config['training']
            if not all(k in training_config for k in ['epochs', 'learning_rate']):
                raise ValueError("Invalid training configuration")
                
            # Validate inference configuration
            inference_config = config['inference']
            if not all(k in inference_config for k in ['iterations', 'precision']):
                raise ValueError("Invalid inference configuration")
                
            # Validate logging configuration
            logging_config = config['logging']
            if not all(k in logging_config for k in ['level', 'format']):
                raise ValueError("Invalid logging configuration")
                
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            raise
            
    def _save_config(self, config: Dict, save_dir: Path):
        """Save merged configuration"""
        try:
            # Create config directory if it doesn't exist
            config_dir = save_dir / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Save merged configuration
            save_path = config_dir / "rgm_config.json"
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Saved merged configuration to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise