"""
GNN Specification Loader

This module provides utilities for loading and validating GNN specifications
from configuration files.
"""

from pathlib import Path
from typing import Dict, Any
import yaml

class RGMGNNLoader:
    """Loader for GNN specifications and configurations."""
    
    def __init__(self):
        """Initialize the GNN loader."""
        pass
        
    def load_specifications(self, config_dir: Path) -> Dict[str, Any]:
        """Load GNN specifications from configuration directory.
        
        Args:
            config_dir: Directory containing GNN configuration files
            
        Returns:
            Dictionary containing validated GNN specifications
            
        Raises:
            FileNotFoundError: If no valid configuration files found
            ValueError: If specifications are invalid
        """
        # Find all .gnn files
        config_files = list(config_dir.glob("*.gnn"))
        if not config_files:
            raise FileNotFoundError(
                f"No .gnn configuration files found in {config_dir}"
            )
            
        # Load and merge configurations
        specs = {}
        for config_file in config_files:
            with open(config_file) as f:
                config = yaml.safe_load(f)
                specs.update(self._validate_config(config))
                
        return specs
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GNN configuration structure.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = [
            'architecture',
            'model',
            'training',
            'matrices'
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        # Validate architecture section
        arch = config['architecture']
        if 'hierarchy' not in arch:
            raise ValueError("Missing 'hierarchy' in architecture config")
            
        hierarchy = arch['hierarchy']
        if 'levels' not in hierarchy or 'dimensions' not in hierarchy:
            raise ValueError(
                "Architecture hierarchy must specify 'levels' and 'dimensions'"
            )
            
        # Validate dimensions for each level
        levels = hierarchy['levels']
        dimensions = hierarchy['dimensions']
        for level in range(levels):
            level_key = f'level{level}'
            if level_key not in dimensions:
                raise ValueError(f"Missing dimensions for {level_key}")
                
            level_dims = dimensions[level_key]
            required_dims = ['input', 'state', 'factor']
            for dim in required_dims:
                if dim not in level_dims:
                    raise ValueError(
                        f"Missing {dim} dimension for {level_key}"
                    )
                    
        # Validate matrices section
        matrices = config['matrices']
        required_matrix_types = ['recognition', 'generative', 'lateral']
        for matrix_type in required_matrix_types:
            if matrix_type not in matrices:
                raise ValueError(
                    f"Missing {matrix_type} matrices in config"
                )
                
            # Validate matrix dimensions
            matrix_group = matrices[matrix_type]
            for level in range(levels):
                matrix_key = f'{matrix_type[0].upper()}{level}'
                if matrix_key not in matrix_group:
                    raise ValueError(
                        f"Missing {matrix_key} in {matrix_type} matrices"
                    )
                    
                dims = matrix_group[matrix_key]
                if not isinstance(dims, list) or len(dims) != 2:
                    raise ValueError(
                        f"Invalid dimensions for {matrix_key}: {dims}"
                    )
                    
        return config