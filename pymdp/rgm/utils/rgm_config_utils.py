"""
Configuration Utilities
=====================

Utilities for managing RGM configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

class RGMConfigUtils:
    """Utilities for configuration management"""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path) as f:
                config = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
        return config
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations"""
        merged = {}
        
        for config in configs:
            RGMConfigUtils._deep_update(merged, config)
            
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any],
                       schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against schema"""
        valid = True
        messages = []
        
        def _validate_field(field_path: str,
                          field_value: Any,
                          field_schema: Dict[str, Any]) -> None:
            nonlocal valid
            
            # Check required fields
            if field_schema.get('required', False) and field_value is None:
                valid = False
                messages.append(f"Missing required field: {field_path}")
                return
                
            # Check type
            if 'type' in field_schema:
                expected_type = field_schema['type']
                if not isinstance(field_value, expected_type):
                    valid = False
                    messages.append(
                        f"Invalid type for {field_path}: "
                        f"expected {expected_type.__name__}, got {type(field_value).__name__}"
                    )
                    return
                    
                # Check list items if type is list
                if expected_type == list and 'items' in field_schema:
                    item_schema = field_schema['items']
                    for i, item in enumerate(field_value):
                        item_path = f"{field_path}[{i}]"
                        if not isinstance(item, item_schema['type']):
                            valid = False
                            messages.append(
                                f"Invalid type for {item_path}: "
                                f"expected {item_schema['type'].__name__}, "
                                f"got {type(item).__name__}"
                            )
                            
            # Check range
            if 'range' in field_schema:
                min_val, max_val = field_schema['range']
                if not (min_val <= field_value <= max_val):
                    valid = False
                    messages.append(
                        f"Value out of range for {field_path}: "
                        f"expected [{min_val}, {max_val}], got {field_value}"
                    )
                    
            # Check enum
            if 'enum' in field_schema and field_value not in field_schema['enum']:
                valid = False
                messages.append(
                    f"Invalid value for {field_path}: "
                    f"expected one of {field_schema['enum']}, got {field_value}"
                )
        
        def _validate_recursive(path: str,
                              value: Any,
                              schema_part: Dict[str, Any]) -> None:
            if isinstance(schema_part, dict) and 'properties' in schema_part:
                # Validate nested structure
                for key, subschema in schema_part['properties'].items():
                    if key in value:
                        new_path = f"{path}.{key}" if path else key
                        _validate_recursive(new_path, value[key], subschema)
            else:
                # Validate leaf field
                _validate_field(path, value, schema_part)
                
        _validate_recursive("", config, schema)
        return valid, messages
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default RGM configuration"""
        return {
            "gnn_files": [
                "rgm_base.gnn",
                "rgm_svd_block.gnn", 
                "rgm_hierarchical_level.gnn",
                "rgm_mnist.gnn",
                "rgm_message_passing.gnn",
                "rgm_active_learning.gnn"
            ],
            "data": {
                "mnist": {
                    "training_samples": 10000,
                    "exemplars_per_class": 13,
                    "image_format": {
                        "input_size": [28, 28],
                        "output_size": [32, 32],
                        "channels": 3,
                        "dtype": "float32",
                        "range": [0, 1]
                    }
                }
            },
            "preprocessing": {
                "histogram_equalization": True,
                "clip_limit": 0.03,
                "gaussian_smoothing": True,
                "kernel_size": [2, 2],
                "sigma": 1.0
            },
            "model": {
                "hierarchy": {
                    "n_levels": 4,
                    "level0": {
                        "block_size": [4, 4],
                        "n_components": 32,
                        "quantization_levels": 7
                    }
                },
                "matrix_shapes": {
                    "A": [1024, 256],  # 32x32 -> 16x16
                    "B": [256, 256],   # State transitions
                    "D": [256],        # Initial states
                    "E": [10]          # Path states (digit classes)
                }
            },
            "learning": {
                "fast_structure": {
                    "exemplars_per_class": 13,
                    "concentration": 0.0625
                },
                "active_learning": {
                    "beta": 512.0,
                    "max_samples": 10000,
                    "elbo_threshold": -13.85
                }
            }
        }
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any],
                    update_dict: Dict[str, Any]) -> None:
        """Recursively update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                RGMConfigUtils._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value