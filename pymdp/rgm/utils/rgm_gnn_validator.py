"""
RGM GNN Validator
================

Validates GNN specifications for RGM.
Ensures proper structure and relationships between specifications.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from rgm_experiment_utils import RGMExperimentUtils

class RGMGNNValidator:
    """Validates GNN specifications"""
    
    def __init__(self):
        """Initialize GNN validator"""
        self.logger = RGMExperimentUtils.get_logger('gnn_validator')
        
    def validate_gnn_spec(self, spec: Dict, filename: str) -> Tuple[bool, List[str]]:
        """
        Validate GNN specification.
        
        Args:
            spec: GNN specification dictionary
            filename: Name of GNN file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            messages = []
            
            # Check required sections first
            if not self._check_required_sections(spec, filename, messages):
                return False, messages
                
            # Validate based on file type
            if filename == 'rgm_base.gnn':
                messages.extend(self._validate_base_spec(spec))
            elif filename == 'rgm_mnist.gnn':
                messages.extend(self._validate_mnist_spec(spec))
            elif filename == 'rgm_hierarchical_level.gnn':
                messages.extend(self._validate_hierarchical_spec(spec))
            else:
                messages.extend(self._validate_additional_spec(spec))
                
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error validating GNN spec: {str(e)}")
            raise
            
    def _check_required_sections(self, spec: Dict, filename: str, messages: List[str]) -> bool:
        """Check required sections based on file type"""
        required_sections = {
            'rgm_base.gnn': ['hierarchy', 'learning'],
            'rgm_mnist.gnn': ['data', 'hierarchy', 'learning'],
            'rgm_hierarchical_level.gnn': ['hierarchy'],
            'rgm_message_passing.gnn': ['messagePassing'],
            'rgm_active_learning.gnn': ['activeLearning']
        }
        
        if filename not in required_sections:
            return True  # No specific requirements for additional specs
            
        missing = [s for s in required_sections[filename] if s not in spec]
        if missing:
            messages.append(f"Missing required sections: {', '.join(missing)}")
            return False
            
        return True
        
    def _validate_base_spec(self, spec: Dict) -> List[str]:
        """Validate base GNN specification"""
        messages = []
        
        # Validate hierarchy
        if 'hierarchy' in spec:
            hierarchy = spec['hierarchy']
            
            # Check dimensions
            if 'dimensions' not in hierarchy:
                messages.append("Missing dimensions in hierarchy")
            else:
                dims = hierarchy['dimensions']
                for level in range(4):  # Assuming 4-level hierarchy
                    level_key = f'level{level}'
                    if level_key not in dims:
                        messages.append(f"Missing {level_key} in dimensions")
                    else:
                        level_dims = dims[level_key]
                        if 'state' not in level_dims or 'factor' not in level_dims:
                            messages.append(f"Missing state/factor dimensions for {level_key}")
                            
            # Check matrices
            if 'matrices' not in hierarchy:
                messages.append("Missing matrices section in hierarchy")
            else:
                matrices = hierarchy['matrices']
                required_matrices = ['A', 'B', 'D', 'E']
                for matrix in required_matrices:
                    if matrix not in matrices:
                        messages.append(f"Missing matrix specification: {matrix}")
                    else:
                        matrix_spec = matrices[matrix]
                        if not self._validate_matrix_spec(matrix_spec):
                            messages.append(f"Invalid specification for matrix {matrix}")
                            
        # Validate learning parameters
        if 'learning' in spec:
            learning = spec['learning']
            required_params = ['active', 'message_passing', 'regularization']
            for param in required_params:
                if param not in learning:
                    messages.append(f"Missing learning parameter: {param}")
                    
        return messages
        
    def _validate_mnist_spec(self, spec: Dict) -> List[str]:
        """Validate MNIST-specific specification"""
        messages = []
        
        # Validate data configuration
        if 'data' in spec:
            data = spec['data']
            if data.get('type') != 'mnist':
                messages.append("Invalid data type - must be 'mnist'")
                
            # Check input configuration
            if 'input' in data:
                input_config = data['input']
                required_input = ['size', 'channels', 'preprocessing']
                for param in required_input:
                    if param not in input_config:
                        messages.append(f"Missing input parameter: {param}")
                        
            # Check output configuration
            if 'output' in data:
                output_config = data['output']
                if output_config.get('classes') != 10:
                    messages.append("Invalid number of output classes - must be 10 for MNIST")
                    
        # Validate hierarchy
        if 'hierarchy' in spec:
            hierarchy = spec['hierarchy']
            required_levels = ['level0', 'level1', 'level2', 'level3']
            for level in required_levels:
                if level not in hierarchy:
                    messages.append(f"Missing hierarchy level: {level}")
                else:
                    level_config = hierarchy[level]
                    if not self._validate_level_config(level_config, level):
                        messages.append(f"Invalid configuration for {level}")
                        
        return messages
        
    def _validate_hierarchical_spec(self, spec: Dict) -> List[str]:
        """Validate hierarchical level specification"""
        messages = []
        
        if 'hierarchy' not in spec:
            messages.append("Missing hierarchy section")
            return messages
            
        hierarchy = spec['hierarchy']
        
        # Check level structure
        if 'level_structure' not in hierarchy:
            messages.append("Missing level_structure in hierarchy")
        else:
            structure = hierarchy['level_structure']
            for level in range(4):
                level_key = f'level{level}'
                if level_key not in structure:
                    messages.append(f"Missing {level_key} in level_structure")
                else:
                    level_config = structure[level_key]
                    if not self._validate_level_structure(level_config, level):
                        messages.append(f"Invalid structure for {level_key}")
                        
        # Check connections
        if 'connections' not in hierarchy:
            messages.append("Missing connections configuration")
        else:
            connections = hierarchy['connections']
            required_connections = ['bottom_up', 'top_down', 'lateral']
            for conn in required_connections:
                if conn not in connections:
                    messages.append(f"Missing connection type: {conn}")
                    
        return messages
        
    def _validate_additional_spec(self, spec: Dict) -> List[str]:
        """Validate additional specification"""
        messages = []
        
        # Check for modelType
        if 'modelType' not in spec or spec['modelType'] != 'RGM':
            messages.append("Missing or invalid modelType - must be 'RGM'")
            
        # Validate dimensions if present
        if 'dimensions' in spec:
            dims = spec['dimensions']
            if not isinstance(dims, dict):
                messages.append("Invalid dimensions format")
                
        return messages
        
    def _validate_matrix_spec(self, spec: Dict) -> bool:
        """Validate matrix specification"""
        required = ['type', 'initialization', 'constraints']
        if not all(key in spec for key in required):
            return False
            
        valid_inits = ['random', 'uniform', 'zeros', 'ones']
        if spec['initialization']['method'] not in valid_inits:
            return False
            
        if not isinstance(spec['constraints'], list):
            return False
            
        return True
        
    def _validate_level_config(self, config: Dict, level: str) -> bool:
        """Validate level configuration"""
        required = ['type', 'activation']
        if not all(key in config for key in required):
            return False
            
        # All levels except top need receptive field and stride
        if level != 'level3':
            if 'receptive_field' not in config or 'stride' not in config:
                return False
                
        return True
        
    def _validate_level_structure(self, config: Dict, level: int) -> bool:
        """Validate level structure"""
        required = ['type', 'dimensions']
        if not all(key in config for key in required):
            return False
            
        dims = config['dimensions']
        if 'state' not in dims or 'factor' not in dims:
            return False
            
        return True