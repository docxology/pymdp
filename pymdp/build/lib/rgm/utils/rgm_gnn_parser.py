"""
RGM GNN Parser
=============

Parses and validates GNN specifications for RGM.
Handles specification parsing, validation, and normalization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from rgm_experiment_utils import RGMExperimentUtils
from rgm_validation_utils import RGMValidationUtils

class RGMGNNParser:
    """Parses and validates GNN specifications"""
    
    def __init__(self):
        """Initialize GNN parser"""
        self.logger = RGMExperimentUtils.get_logger('gnn_parser')
        self.validator = RGMValidationUtils()
        
    def parse_gnn_spec(self, spec: Dict) -> Dict:
        """
        Parse and validate GNN specification.
        
        Args:
            spec: Raw GNN specification dictionary
            
        Returns:
            Parsed and validated specification
        """
        try:
            # Validate basic structure
            if not self._validate_basic_structure(spec):
                raise ValueError("Invalid GNN specification structure")
                
            # Parse hierarchy
            if 'hierarchy' in spec:
                spec['hierarchy'] = self._parse_hierarchy(spec['hierarchy'])
                
            # Parse learning parameters
            if 'learning' in spec:
                spec['learning'] = self._parse_learning_params(spec['learning'])
                
            # Parse data configuration
            if 'data' in spec:
                spec['data'] = self._parse_data_config(spec['data'])
                
            # Parse validation settings
            if 'validation' in spec:
                spec['validation'] = self._parse_validation_config(spec['validation'])
                
            # Add computed fields
            spec = self._add_computed_fields(spec)
            
            return spec
            
        except Exception as e:
            self.logger.error(f"Error parsing GNN spec: {str(e)}")
            raise
            
    def _validate_basic_structure(self, spec: Dict) -> bool:
        """Validate basic specification structure"""
        try:
            # Check required fields
            required = ['modelType']
            if not all(key in spec for key in required):
                self.logger.error("Missing required fields in specification")
                return False
                
            # Check model type
            if spec['modelType'] != 'RGM':
                self.logger.error(f"Invalid model type: {spec['modelType']}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating structure: {str(e)}")
            raise
            
    def _parse_hierarchy(self, hierarchy: Dict) -> Dict:
        """Parse hierarchy configuration"""
        try:
            parsed = hierarchy.copy()
            
            # Parse dimensions
            if 'dimensions' in parsed:
                for level, config in parsed['dimensions'].items():
                    # Validate level format
                    if not level.startswith('level'):
                        raise ValueError(f"Invalid level format: {level}")
                        
                    # Parse level configuration
                    config = self._parse_level_config(config, level)
                    parsed['dimensions'][level] = config
                    
            # Parse matrix configurations
            if 'matrices' in parsed:
                for matrix_type, config in parsed['matrices'].items():
                    config = self._parse_matrix_config(config, matrix_type)
                    parsed['matrices'][matrix_type] = config
                    
            # Parse connections
            if 'connections' in parsed:
                parsed['connections'] = self._parse_connections(parsed['connections'])
                
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing hierarchy: {str(e)}")
            raise
            
    def _parse_level_config(self, config: Dict, level: str) -> Dict:
        """Parse level configuration"""
        try:
            parsed = config.copy()
            
            # Validate dimensions
            required_dims = ['state', 'factor']
            for dim in required_dims:
                if dim not in parsed:
                    raise ValueError(f"Missing {dim} dimension in {level}")
                    
            # Parse processing configuration
            if 'processing' in parsed:
                parsed['processing'] = self._parse_processing_config(
                    parsed['processing'],
                    level
                )
                
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing level config: {str(e)}")
            raise
            
    def _parse_matrix_config(self, config: Dict, matrix_type: str) -> Dict:
        """Parse matrix configuration"""
        try:
            parsed = config.copy()
            
            # Validate initialization
            if 'initialization' not in parsed:
                raise ValueError(f"Missing initialization for matrix {matrix_type}")
                
            init = parsed['initialization']
            valid_methods = ['random', 'uniform', 'zeros', 'ones']
            if init['method'] not in valid_methods:
                raise ValueError(f"Invalid initialization method: {init['method']}")
                
            # Validate constraints
            if 'constraints' not in parsed:
                raise ValueError(f"Missing constraints for matrix {matrix_type}")
                
            valid_constraints = [
                'column_normalized',
                'non_negative',
                'symmetric',
                'normalized'
            ]
            for constraint in parsed['constraints']:
                if constraint not in valid_constraints:
                    raise ValueError(f"Invalid constraint: {constraint}")
                    
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing matrix config: {str(e)}")
            raise
            
    def _parse_processing_config(self, config: Dict, level: str) -> Dict:
        """Parse processing configuration"""
        try:
            parsed = config.copy()
            
            # Validate activation
            if 'activation' in parsed:
                valid_activations = ['relu', 'softmax', 'tanh', 'sigmoid']
                if parsed['activation']['type'] not in valid_activations:
                    raise ValueError(
                        f"Invalid activation type in {level}: "
                        f"{parsed['activation']['type']}"
                    )
                    
            # Validate pooling
            if 'pooling' in parsed:
                valid_pooling = ['max', 'average']
                if parsed['pooling']['type'] not in valid_pooling:
                    raise ValueError(
                        f"Invalid pooling type in {level}: "
                        f"{parsed['pooling']['type']}"
                    )
                    
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing processing config: {str(e)}")
            raise
            
    def _parse_connections(self, connections: Dict) -> Dict:
        """Parse connection configuration"""
        try:
            parsed = connections.copy()
            
            # Validate connection types
            valid_types = ['dense', 'sparse']
            for conn in parsed.values():
                if conn['type'] not in valid_types:
                    raise ValueError(f"Invalid connection type: {conn['type']}")
                    
                # Validate sparsity for sparse connections
                if conn['type'] == 'sparse' and 'sparsity' not in conn:
                    raise ValueError("Missing sparsity parameter for sparse connection")
                    
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing connections: {str(e)}")
            raise
            
    def _parse_learning_params(self, learning: Dict) -> Dict:
        """Parse learning parameters"""
        try:
            parsed = learning.copy()
            
            # Validate active learning
            if 'active' in parsed:
                active = parsed['active']
                if 'enabled' not in active:
                    raise ValueError("Missing active learning enabled flag")
                    
                if active['enabled']:
                    required = ['beta', 'max_precision', 'precision_growth']
                    if not all(key in active for key in required):
                        raise ValueError("Missing required active learning parameters")
                        
            # Validate message passing
            if 'message_passing' in parsed:
                mp = parsed['message_passing']
                required = ['max_iterations', 'convergence_threshold']
                if not all(key in mp for key in required):
                    raise ValueError("Missing required message passing parameters")
                    
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing learning params: {str(e)}")
            raise
            
    def _parse_data_config(self, data: Dict) -> Dict:
        """Parse data configuration"""
        try:
            parsed = data.copy()
            
            # Validate data type
            if 'type' not in parsed:
                raise ValueError("Missing data type")
                
            # Validate input configuration
            if 'input' in parsed:
                input_config = parsed['input']
                required = ['size', 'channels', 'preprocessing']
                if not all(key in input_config for key in required):
                    raise ValueError("Missing required input configuration")
                    
            # Validate output configuration
            if 'output' in parsed:
                output_config = parsed['output']
                required = ['classes', 'encoding']
                if not all(key in output_config for key in required):
                    raise ValueError("Missing required output configuration")
                    
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing data config: {str(e)}")
            raise
            
    def _parse_validation_config(self, validation: Dict) -> Dict:
        """Parse validation configuration"""
        try:
            parsed = validation.copy()
            
            # Validate metrics
            if 'metrics' in parsed:
                valid_metrics = [
                    'accuracy',
                    'confusion_matrix',
                    'precision',
                    'recall',
                    'f1_score'
                ]
                for metric in parsed['metrics']:
                    if metric not in valid_metrics:
                        raise ValueError(f"Invalid metric: {metric}")
                        
            # Validate early stopping
            if 'early_stopping' in parsed:
                es = parsed['early_stopping']
                required = ['enabled', 'patience', 'min_delta']
                if not all(key in es for key in required):
                    raise ValueError("Missing required early stopping parameters")
                    
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing validation config: {str(e)}")
            raise
            
    def _add_computed_fields(self, spec: Dict) -> Dict:
        """Add computed fields to specification"""
        try:
            # Add total parameters
            if 'hierarchy' in spec and 'dimensions' in spec['hierarchy']:
                total_params = 0
                dims = spec['hierarchy']['dimensions']
                
                for level_config in dims.values():
                    state_dim = level_config['state']
                    factor_dim = level_config['factor']
                    
                    # A matrix parameters
                    total_params += state_dim * factor_dim
                    # B matrix parameters
                    total_params += factor_dim * factor_dim
                    # D matrix parameters
                    total_params += factor_dim
                    
                spec['computed'] = {
                    'total_parameters': total_params,
                    'memory_estimate_mb': total_params * 4 / (1024 * 1024)  # 4 bytes per float
                }
                
            return spec
            
        except Exception as e:
            self.logger.error(f"Error adding computed fields: {str(e)}")
            raise