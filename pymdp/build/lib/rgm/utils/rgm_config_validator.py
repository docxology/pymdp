"""
RGM Configuration Validator
========================

Validates configuration for the RGM pipeline:
- Required fields and types
- Value ranges and constraints
- Consistency between related fields
"""

import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger('rgm')


class RGMConfigValidator:
    """Configuration validator for RGM models.
    
    This class provides methods to load and validate RGM configuration files,
    ensuring all required fields are present and valid.
    
    Example:
        >>> config = RGMConfigValidator.load_and_validate("config.json")
        >>> print(config['architecture']['layer_dims'])
        [784, 500, 200]
    """
    
    @staticmethod
    def load_and_validate(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from a JSON file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
            json.JSONDecodeError: If config file is not valid JSON
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config = json.load(f)
            
        RGMConfigValidator.validate_config(config)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate RGM configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['architecture', 'initialization', 'training', 'data']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Missing required config section: {section}\n"
                    f"Available sections: {list(config.keys())}"
                )
        
        # Validate each section
        RGMConfigValidator._validate_architecture(config['architecture'])
        RGMConfigValidator._validate_initialization(config['initialization'])
        RGMConfigValidator._validate_training(config['training'])
        RGMConfigValidator._validate_data(config['data'])
        
        # Validate consistency between sections
        RGMConfigValidator._validate_cross_section_consistency(config)
        
        logger.info("✓ Configuration validated successfully")

    @staticmethod
    def _validate_architecture(config: Dict[str, Any]) -> None:
        """Validate architecture configuration."""
        required_fields = ['layer_dims']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required architecture field: {field}")
        
        # Validate layer dimensions
        layer_dims = config['layer_dims']
        if not isinstance(layer_dims, list):
            raise ValueError("layer_dims must be a list")
        
        if len(layer_dims) < 2:
            raise ValueError("Must have at least 2 layers")
        
        if layer_dims[0] != 784:  # MNIST input dimension
            raise ValueError("First layer dimension must be 784 (MNIST input)")
        
        for dim in layer_dims:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("Layer dimensions must be positive integers")

    @staticmethod
    def _validate_initialization(config: Dict[str, Any]) -> None:
        """Validate initialization configuration."""
        required_sections = ['recognition', 'generative', 'lateral']
        required_fields = ['method']
        valid_methods = ['xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                        'kaiming_normal', 'identity_with_noise']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing initialization section: {section}")
            
            section_config = config[section]
            for field in required_fields:
                if field not in section_config:
                    raise ValueError(f"Missing field in {section}: {field}")
            
            if section_config['method'] not in valid_methods:
                raise ValueError(
                    f"Invalid initialization method for {section}: "
                    f"{section_config['method']}"
                )
            
            if section_config['method'] == 'identity_with_noise':
                if 'noise_std' not in section_config:
                    raise ValueError("noise_std required for identity_with_noise")
                if not 0 <= section_config['noise_std'] <= 1:
                    raise ValueError("noise_std must be between 0 and 1")
            
            if 'gain' in section_config:
                if not isinstance(section_config['gain'], (int, float)):
                    raise ValueError("gain must be a number")
                if section_config['gain'] <= 0:
                    raise ValueError("gain must be positive")

    @staticmethod
    def _validate_training(config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        required_fields = {
            'batch_size': (int, lambda x: x > 0),
            'num_epochs': (int, lambda x: x > 0),
            'learning_rate': (float, lambda x: x > 0),
            'weight_decay': (float, lambda x: x >= 0),
            'num_workers': (int, lambda x: x >= 0),
            'pin_memory': (bool, None),
            'checkpoint_interval': (int, lambda x: x > 0),
            'validation_interval': (int, lambda x: x > 0)
        }
        
        for field, (expected_type, validator) in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required training field: {field}")
            
            value = config[field]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"{field} must be of type {expected_type.__name__}"
                )
            
            if validator and not validator(value):
                raise ValueError(f"Invalid value for {field}: {value}")

    @staticmethod
    def _validate_data(config: Dict[str, Any]) -> None:
        """Validate data configuration."""
        required_fields = ['dataset', 'validation_split', 'normalization']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required data field: {field}")
        
        if config['dataset'] != 'MNIST':
            raise ValueError(f"Unsupported dataset: {config['dataset']}")
        
        if not 0 <= config['validation_split'] <= 1:
            raise ValueError("validation_split must be between 0 and 1")
        
        # Validate normalization
        norm_config = config['normalization']
        if 'mean' not in norm_config or 'std' not in norm_config:
            raise ValueError("Normalization must specify mean and std")
        
        if not isinstance(norm_config['mean'], list):
            raise ValueError("mean must be a list")
        if not isinstance(norm_config['std'], list):
            raise ValueError("std must be a list")
        
        if len(norm_config['mean']) != 1 or len(norm_config['std']) != 1:
            raise ValueError("mean and std must have length 1 for MNIST")
        
        # Validate augmentation if present
        if 'augmentation' in config:
            aug_config = config['augmentation']
            if not isinstance(aug_config.get('enabled', False), bool):
                raise ValueError("augmentation.enabled must be boolean")
            
            if aug_config.get('enabled', False):
                if 'random_rotation' in aug_config:
                    if not isinstance(aug_config['random_rotation'], (int, float)):
                        raise ValueError("random_rotation must be numeric")
                    if aug_config['random_rotation'] < 0:
                        raise ValueError("random_rotation must be non-negative")
                
                if 'random_translation' in aug_config:
                    if not isinstance(aug_config['random_translation'], (int, float)):
                        raise ValueError("random_translation must be numeric")
                    if not 0 <= aug_config['random_translation'] <= 1:
                        raise ValueError(
                            "random_translation must be between 0 and 1"
                        )

    @staticmethod
    def _validate_cross_section_consistency(config: Dict[str, Any]) -> None:
        """Validate consistency between different configuration sections."""
        # Validate batch size against GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            batch_size = config['training']['batch_size']
            input_dim = config['architecture']['layer_dims'][0]
            
            # Rough estimate of memory needed per sample
            mem_per_sample = input_dim * 4  # 4 bytes per float32
            total_mem_needed = batch_size * mem_per_sample * 3  # Forward + backward + gradients
            
            if total_mem_needed > gpu_mem * 0.8:  # 80% of GPU memory
                logger.warning(
                    f"Batch size {batch_size} may be too large for GPU memory"
                )
        
        # Validate learning rate against batch size
        if config['training']['learning_rate'] * config['training']['batch_size'] > 1:
            logger.warning(
                "Learning rate might be too high for the given batch size"
            )
        
        # Validate checkpoint interval against number of epochs
        if (config['training']['checkpoint_interval'] >
                config['training']['num_epochs']):
            logger.warning(
                "Checkpoint interval is larger than total number of epochs"
            ) 