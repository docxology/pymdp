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
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger('rgm')

class RGMConfigValidator:
    """Configuration validator for RGM models."""
    
    REQUIRED_SECTIONS = {
        'model', 'training', 'learning', 'data', 'architecture'
    }
    
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
            
        RGMConfigValidator._validate_config(config)
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate RGM configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        missing_sections = RGMConfigValidator.REQUIRED_SECTIONS - set(config.keys())
        if missing_sections:
            raise ValueError(
                f"Missing required config sections: {', '.join(missing_sections)}\n"
                f"Available sections: {list(config.keys())}"
            )
        
        # Validate each section
        RGMConfigValidator._validate_data_config(config['data'])
        RGMConfigValidator._validate_model_config(config['model'])
        RGMConfigValidator._validate_training_config(config['training'])
        RGMConfigValidator._validate_learning_config(config['learning'])
        RGMConfigValidator._validate_architecture_config(config['architecture'])
        
        logger.info("✓ Configuration validated successfully")
    
    @staticmethod
    def _validate_data_config(config: Dict[str, Any]) -> None:
        """Validate data configuration section."""
        # Check for mnist subsection
        if 'mnist' not in config:
            raise ValueError("Missing 'mnist' subsection in data config")
            
        mnist_config = config['mnist']
        required_fields = {
            'training_samples': int,
            'test_samples': int,
            'batch_size': int,
            'augmentation': dict,
            'preprocessing': dict,
            'image_format': dict
        }
        
        RGMConfigValidator._validate_section(mnist_config, 'data.mnist', required_fields)
        
        # Validate image format
        image_format = mnist_config['image_format']
        if not all(isinstance(dim, int) and dim > 0 for dim in image_format['input_size']):
            raise ValueError("input_size must be a list of positive integers")
    
    @staticmethod
    def _validate_architecture_config(config: Dict[str, Any]) -> None:
        """Validate architecture configuration section."""
        required_fields = {
            'type': str,
            'layers': list,
            'optimizer': dict,
            'loss_function': str
        }
        
        RGMConfigValidator._validate_section(config, 'architecture', required_fields)
        
        # Validate optimizer section
        optimizer = config['optimizer']
        required_optimizer_fields = {
            'type': str,
            'learning_rate': (int, float)
        }
        RGMConfigValidator._validate_section(optimizer, 'architecture.optimizer', required_optimizer_fields)
        
        # Validate each layer
        for layer in config['layers']:
            required_layer_fields = {
                'name': str,
                'type': str
            }
            RGMConfigValidator._validate_section(layer, f"architecture.layers.{layer.get('name', 'unnamed')}", required_layer_fields)
            
            # Additional validations based on layer type
            layer_type = layer['type']
            if layer_type == 'convolutional':
                if 'filters' not in layer or 'kernel_size' not in layer or 'activation' not in layer:
                    raise ValueError(f"Missing fields in convolutional layer: {layer}")
            elif layer_type == 'pooling':
                if 'pool_size' not in layer:
                    raise ValueError(f"Missing 'pool_size' in pooling layer: {layer}")
            elif layer_type == 'fully_connected':
                if 'units' not in layer or 'activation' not in layer:
                    raise ValueError(f"Missing fields in fully_connected layer: {layer}")
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
    
    @staticmethod
    def _validate_model_config(config: Dict[str, Any]) -> None:
        """Validate model configuration section."""
        required_fields = {
            'hierarchy': dict,
            'matrix_shapes': dict
        }
        
        RGMConfigValidator._validate_section(config, 'model', required_fields)
        
        # Validate hierarchy
        hierarchy = config['hierarchy']
        if 'n_levels' not in hierarchy:
            raise ValueError("Missing n_levels in model hierarchy")
        if 'dimensions' not in hierarchy:
            raise ValueError("Missing dimensions in model hierarchy")
            
        # Validate dimensions for each level
        dimensions = hierarchy['dimensions']
        for level in range(hierarchy['n_levels']):
            level_key = f'level{level}'
            if level_key not in dimensions:
                raise ValueError(f"Missing {level_key} in hierarchy dimensions")
            level_config = dimensions[level_key]
            if 'state' not in level_config or 'factor' not in level_config:
                raise ValueError(f"Missing state or factor dimensions for {level_key}")
    
    @staticmethod
    def _validate_training_config(config: Dict[str, Any]) -> None:
        """Validate training configuration section."""
        required_fields = {
            'epochs': int,
            'learning_rate': float,
            'weight_decay': float,
            'patience': int,
            'save_interval': int,
            'grad_clip': float,
            'scheduler': dict
        }
        
        RGMConfigValidator._validate_section(config, 'training', required_fields)
        
        # Additional validation
        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
    
    @staticmethod
    def _validate_visualization_config(config: Dict[str, Any]) -> None:
        """Validate visualization configuration section."""
        required_fields = {
            'plot_interval': int,
            'sample_size': int,
            'save_format': str,
            'dpi': int
        }
        
        RGMConfigValidator._validate_section(config, 'visualization', required_fields)
    
    @staticmethod
    def _validate_logging_config(config: Dict[str, Any]) -> None:
        """Validate logging configuration section."""
        required_fields = {
            'level': str,
            'save_dir': str,
            'filename': str,
            'console_format': str,
            'file_format': str
        }
        
        RGMConfigValidator._validate_section(config, 'logging', required_fields)
    
    @staticmethod
    def _validate_section(
        config: Dict[str, Any],
        section_name: str,
        required_fields: Dict[str, type]
    ) -> None:
        """Validate a configuration section.
        
        Args:
            config: Section configuration to validate
            section_name: Name of the section being validated
            required_fields: Dictionary of field names and their expected types
            
        Raises:
            ValueError: If section validation fails
        """
        for field, expected_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {section_name} section")
            
            value = config[field]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Field '{field}' in {section_name} section must be of type {expected_type.__name__}"
                ) 
    
    @staticmethod
    def _validate_learning_config(config: Dict[str, Any]) -> None:
        """Validate learning configuration section."""
        required_fields = {
            'precision_init': float,
            'learning_rate': float,
            'active_learning': dict,
            'message_passing': dict
        }
        
        RGMConfigValidator._validate_section(config, 'learning', required_fields)
        
        # Additional validation
        if config['precision_init'] <= 0:
            raise ValueError("precision_init must be positive")
        if config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive") 