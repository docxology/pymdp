"""
RGM Configuration Loader
=======================

Loads and validates RGM configuration files.
"""

import os
import json
import jsonschema
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

from pymdp.rgm.utils.rgm_experiment_utils import RGMExperimentUtils

class RGMConfigLoader:
    """Loads and validates RGM configurations"""
    
    def __init__(self):
        """Initialize configuration loader"""
        self.logger = RGMExperimentUtils.get_logger('config')
        self.experiment = RGMExperimentUtils.get_experiment()
        
        # Load schema
        self.schema = self._load_schema()
        
    def load_config(self) -> Dict:
        """Load and validate configuration"""
        try:
            # Load base config
            base_config = self._load_base_config()
            
            # Load experiment config
            exp_config = self._load_experiment_config()
            
            # Merge configurations
            config = self._merge_configs(base_config, exp_config)
            
            # Validate configuration
            self._validate_config(config)
            
            # Save merged config
            self._save_config(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _load_schema(self) -> Dict:
        """Load configuration schema"""
        try:
            # First try experiment config directory
            schema_path = self.experiment['dirs']['config'] / "config_schema.json"
            
            # Fall back to package config if not found
            if not schema_path.exists():
                schema_path = Path(__file__).parent.parent / "configs" / "config_schema.json"
                
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema not found at {schema_path}")
                
            with open(schema_path) as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading schema: {str(e)}")
            raise
            
    def _load_base_config(self) -> Dict:
        """Load base configuration"""
        try:
            # First try experiment config directory
            base_path = self.experiment['dirs']['config'] / "default_config.json"
            
            # Fall back to package config if not found
            if not base_path.exists():
                base_path = Path(__file__).parent.parent / "configs" / "default_config.json"
                
            if not base_path.exists():
                raise FileNotFoundError("Default config not found")
                
            with open(base_path) as f:
                config = json.load(f)
                
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading base config: {str(e)}")
            raise
            
    def _load_experiment_config(self) -> Dict:
        """Load experiment-specific configuration"""
        exp_path = self.experiment['dirs']['config'] / "experiment_config.json"
        
        try:
            with open(exp_path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.info("No experiment config found, using defaults")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading experiment config: {str(e)}")
            raise
            
    def _merge_configs(self, base: Dict, exp: Dict) -> Dict:
        """Merge base and experiment configurations"""
        try:
            config = base.copy()
            
            def deep_merge(d1: Dict, d2: Dict):
                """Recursively merge dictionaries"""
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                        
            deep_merge(config, exp)
            return config
            
        except Exception as e:
            self.logger.error(f"Error merging configurations: {str(e)}")
            raise
            
    def _validate_config(self, config: Dict):
        """Validate configuration against schema"""
        try:
            # Validate against JSON schema
            jsonschema.validate(instance=config, schema=self.schema)
            
            # Additional validation
            self._validate_hierarchy(config)
            self._validate_matrices(config)
            self._validate_learning(config)
            
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise ValueError(f"Invalid configuration: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            raise
            
    def _validate_hierarchy(self, config: Dict):
        """Validate hierarchy configuration"""
        try:
            hierarchy = config['model']['hierarchy']
            
            # Check levels
            if 'n_levels' not in hierarchy:
                raise ValueError("Missing n_levels in hierarchy")
                
            n_levels = hierarchy['n_levels']
            
            # Check dimensions for each level
            for level in range(n_levels):
                level_key = f'level{level}'
                if level_key not in hierarchy['dimensions']:
                    raise ValueError(f"Missing configuration for {level_key}")
                    
                level_config = hierarchy['dimensions'][level_key]
                required = ['state', 'factor']
                missing = [dim for dim in required if dim not in level_config]
                if missing:
                    raise ValueError(f"Missing dimensions for {level_key}: {missing}")
                    
        except KeyError as e:
            raise ValueError(f"Missing required configuration: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid hierarchy configuration: {str(e)}")
            
    def _validate_matrices(self, config: Dict):
        """Validate matrix configurations"""
        try:
            matrices = config['model']['matrices']
            required = ['A', 'B', 'D']
            
            # Check required matrices
            missing = [m for m in required if m not in matrices]
            if missing:
                raise ValueError(f"Missing matrix configurations: {missing}")
                
            # Check matrix properties
            for name, matrix_config in matrices.items():
                if 'initialization' not in matrix_config:
                    raise ValueError(f"Missing initialization for matrix {name}")
                if 'constraints' not in matrix_config:
                    raise ValueError(f"Missing constraints for matrix {name}")
                    
        except KeyError as e:
            raise ValueError(f"Missing required matrix configuration: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid matrix configuration: {str(e)}")
            
    def _validate_learning(self, config: Dict):
        """Validate learning configuration"""
        try:
            learning = config['learning']
            required = ['active', 'message_passing', 'regularization']
            
            # Check required sections
            missing = [s for s in required if s not in learning]
            if missing:
                raise ValueError(f"Missing learning sections: {missing}")
                
            # Validate active learning
            active = learning['active']
            if not isinstance(active.get('enabled'), bool):
                raise ValueError("active.enabled must be boolean")
                
            # Validate message passing
            mp = learning['message_passing']
            if mp['max_iterations'] < 1:
                raise ValueError("max_iterations must be positive")
                
        except KeyError as e:
            raise ValueError(f"Missing required learning configuration: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid learning configuration: {str(e)}")
            
    def _save_config(self, config: Dict):
        """Save merged configuration"""
        try:
            # Save to experiment directory
            config_path = self.experiment['dirs']['config'] / "merged_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Saved merged config to: {config_path}")
            
            # Save validation report
            report = {
                'timestamp': datetime.now().isoformat(),
                'validation': {
                    'schema_valid': True,
                    'hierarchy_valid': True,
                    'matrices_valid': True,
                    'learning_valid': True
                }
            }
            
            report_path = self.experiment['dirs']['config'] / "validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise