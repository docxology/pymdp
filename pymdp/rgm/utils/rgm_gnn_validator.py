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

from .rgm_experiment_utils import RGMExperimentUtils

class RGMGNNValidator:
    """Validates GNN specifications"""
    
    def __init__(self):
        """Initialize GNN validator"""
        self.logger = logging.getLogger('rgm.gnn_validator')
        
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
        
        # Check hierarchy structure
        if 'hierarchy' in spec:
            hierarchy = spec['hierarchy']
            
            # Check required hierarchy fields
            required_fields = ['n_levels', 'dimensions']
            missing = [f for f in required_fields if f not in hierarchy]
            if missing:
                messages.append(f"Missing required hierarchy fields: {', '.join(missing)}")
                
            # Check dimensions for each level
            if 'dimensions' in hierarchy:
                dimensions = hierarchy['dimensions']
                for level in range(hierarchy.get('n_levels', 0)):
                    level_key = f'level{level}'
                    if level_key not in dimensions:
                        messages.append(f"Missing dimensions for {level_key}")
                        continue
                        
                    level_dims = dimensions[level_key]
                    if not all(k in level_dims for k in ['state', 'factor']):
                        messages.append(f"Missing state/factor dimensions for {level_key}")
                        
        # Check learning parameters
        if 'learning' in spec:
            learning = spec['learning']
            
            # Check required learning fields
            required_fields = ['precision_init', 'learning_rate']
            missing = [f for f in required_fields if f not in learning]
            if missing:
                messages.append(f"Missing required learning fields: {', '.join(missing)}")
                
        return messages
        
    def _validate_mnist_spec(self, spec: Dict) -> List[str]:
        """Validate MNIST-specific GNN specification"""
        messages = []
        
        # Check data configuration
        if 'data' in spec:
            data = spec['data']
            
            # Check required data fields
            required_fields = ['input', 'output']
            missing = [f for f in required_fields if f not in data]
            if missing:
                messages.append(f"Missing required data fields: {', '.join(missing)}")
                
            # Check input configuration
            if 'input' in data:
                input_config = data['input']
                if not all(k in input_config for k in ['size', 'channels']):
                    messages.append("Missing input size/channels configuration")
                    
            # Check output configuration
            if 'output' in data:
                output_config = data['output']
                if not all(k in output_config for k in ['classes', 'encoding']):
                    messages.append("Missing output classes/encoding configuration")
                    
        return messages
        
    def _validate_hierarchical_spec(self, spec: Dict) -> List[str]:
        """Validate hierarchical level GNN specification"""
        messages = []
        
        # Check hierarchy structure
        if 'hierarchy' in spec:
            hierarchy = spec['hierarchy']
            
            # Check level structure
            if 'level_structure' in hierarchy:
                structure = hierarchy['level_structure']
                
                # Check each level
                for level_key, level_config in structure.items():
                    if not self._validate_level_config(level_config, level_key):
                        messages.append(f"Invalid level configuration: {level_key}")
                        
            # Check connections
            if 'connections' in hierarchy:
                connections = hierarchy['connections']
                if not all(k in connections for k in ['bottom_up', 'top_down']):
                    messages.append("Missing required connection types")
                    
        return messages
        
    def _validate_additional_spec(self, spec: Dict) -> List[str]:
        """Validate additional GNN specification"""
        messages = []
        
        # Check basic structure
        if not isinstance(spec, dict):
            messages.append("Invalid specification format")
            return messages
            
        # Check model type
        if 'modelType' not in spec or spec['modelType'] != 'RGM':
            messages.append("Invalid or missing model type")
            
        return messages
        
    def _validate_level_config(self, config: Dict, level: str) -> bool:
        """Validate level configuration"""
        required_fields = ['type', 'dimensions', 'processing']
        return all(k in config for k in required_fields)