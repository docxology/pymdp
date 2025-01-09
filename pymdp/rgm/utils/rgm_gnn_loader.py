"""
RGM GNN Specification Loader
=========================

Loads and validates GNN specifications for the Renormalization Generative Model.
"""

import yaml
from pathlib import Path
from typing import Dict, List
import logging

from .rgm_logging import RGMLogging

class RGMGNNLoader:
    """Loads and validates GNN specifications."""
    
    def __init__(self):
        """Initialize GNN loader."""
        self.logger = RGMLogging.get_logger("rgm.gnn_loader")
        
    def load_specifications(self, gnn_dir: Path) -> Dict:
        """
        Load and merge GNN specifications from directory.
        
        Args:
            gnn_dir: Directory containing GNN specification files
            
        Returns:
            Merged specification dictionary
            
        Raises:
            ValueError: If specifications are invalid
        """
        try:
            # Load base specification first
            base_spec = self._load_base_spec(gnn_dir)
            
            # Load and merge other specifications
            for gnn_file in gnn_dir.glob("*.gnn"):
                if gnn_file.name == "rgm_base.gnn":
                    continue
                    
                spec = self._load_spec_file(gnn_file)
                if spec:
                    self._merge_specs(base_spec, spec)
                    
            # Validate final merged specification
            self._validate_merged_spec(base_spec)
            
            return base_spec
            
        except Exception as e:
            self.logger.error(f"Error loading specifications: {str(e)}")
            raise
            
    def _load_base_spec(self, gnn_dir: Path) -> Dict:
        """Load base GNN specification."""
        base_file = gnn_dir / "rgm_base.gnn"
        if not base_file.exists():
            raise ValueError("Base specification (rgm_base.gnn) not found")
            
        return self._load_spec_file(base_file)
        
    def _load_spec_file(self, file_path: Path) -> Dict:
        """Load single GNN specification file."""
        try:
            with open(file_path) as f:
                content = f.read()
                if content.startswith('"""'):
                    _, content = content.split('"""', 2)[1:]
                    
                spec = yaml.safe_load(content)
                if spec is None:
                    self.logger.warning(f"Empty specification in {file_path.name}")
                    return {}
                    
                return spec
                
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing {file_path.name}: {str(e)}")
            raise
            
    def _merge_specs(self, base_spec: Dict, new_spec: Dict):
        """Merge new specification into base specification."""
        for key, value in new_spec.items():
            if key not in base_spec:
                base_spec[key] = value
            elif isinstance(value, dict):
                if key == 'matrices':
                    # Special handling for matrix specifications
                    if 'matrices' not in base_spec:
                        base_spec['matrices'] = {}
                    for matrix_type, matrices in value.items():
                        if matrix_type not in base_spec['matrices']:
                            base_spec['matrices'][matrix_type] = {}
                        base_spec['matrices'][matrix_type].update(matrices)
                else:
                    # Regular dictionary merge
                    if isinstance(base_spec[key], dict):
                        self._merge_specs(base_spec[key], value)
                        
    def _validate_merged_spec(self, spec: Dict):
        """Validate merged specification."""
        # Check required sections
        required_sections = ['hierarchy', 'matrices']
        missing = [s for s in required_sections if s not in spec]
        if missing:
            raise ValueError(f"Missing required sections: {missing}")
            
        # Validate hierarchy
        if 'hierarchy' in spec:
            self._validate_hierarchy(spec)
            
        # Validate matrices
        if 'matrices' in spec:
            self._validate_matrices(spec['matrices'])
            
    def _validate_matrices(self, matrices: Dict):
        """
        Validate matrix specifications.
        
        Args:
            matrices: Dictionary of matrix specifications
            
        Raises:
            ValueError: If matrix specifications are invalid
        """
        required_types = ['recognition', 'generative', 'lateral']
        missing = [t for t in required_types if t not in matrices]
        if missing:
            raise ValueError(f"Missing matrix types: {missing}")
            
        # Matrix type prefixes
        prefixes = {
            'recognition': 'R',
            'generative': 'G',
            'lateral': 'L'
        }
        
        # Validate each matrix type has complete specifications
        for matrix_type, specs in matrices.items():
            prefix = prefixes[matrix_type]
            expected_names = [f"{prefix}{i}" for i in range(3)]
            missing_matrices = [name for name in expected_names if name not in specs]
            if missing_matrices:
                raise ValueError(
                    f"Missing {matrix_type} matrices: {missing_matrices}"
                )
                
            # Validate shape specifications
            for name, shape in specs.items():
                if not isinstance(shape, list) or len(shape) != 2:
                    raise ValueError(
                        f"Invalid shape specification for {name}: "
                        f"expected [rows, cols], got {shape}"
                    )
                    
                if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                    raise ValueError(
                        f"Invalid dimensions in {name}: "
                        f"expected positive integers, got {shape}"
                    )
                    
    def _validate_hierarchy(self, spec: Dict):
        """
        Validate hierarchy specification.
        
        Args:
            spec: Specification dictionary containing hierarchy information
            
        Raises:
            ValueError: If hierarchy specification is invalid
        """
        hierarchy = spec['hierarchy']
        
        # Check required fields
        required_fields = ['levels', 'dimensions']
        missing = [f for f in required_fields if f not in hierarchy]
        if missing:
            raise ValueError(f"Missing required hierarchy fields: {missing}")
            
        # Validate levels
        levels = hierarchy['levels']
        if not isinstance(levels, int) or levels < 1:
            raise ValueError(f"Invalid number of levels: {levels}")
            
        # Validate dimensions for each level
        dimensions = hierarchy['dimensions']
        for level in range(levels):
            level_key = f'level{level}'
            if level_key not in dimensions:
                raise ValueError(f"Missing dimensions for {level_key}")
                
            level_dims = dimensions[level_key]
            
            # Check required dimension types
            required_dims = ['input', 'state', 'factor']
            missing_dims = [d for d in required_dims if d not in level_dims]
            if missing_dims:
                raise ValueError(f"Missing dimensions {missing_dims} for {level_key}")
                
            # Validate dimension values
            for dim_name, value in level_dims.items():
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"Invalid {dim_name} dimension for {level_key}: "
                        f"expected positive integer, got {value}"
                    )
                    
            # Validate dimension relationships
            if level > 0:
                prev_level = dimensions[f'level{level-1}']
                if level_dims['input'] != prev_level['state']:
                    raise ValueError(
                        f"Dimension mismatch between levels {level-1} and {level}: "
                        f"state {prev_level['state']} ≠ input {level_dims['input']}"
                    )
                    
        # Validate MNIST-specific dimensions
        if dimensions['level0']['input'] != 784:  # 28x28 MNIST images
            raise ValueError(
                f"Invalid input dimension for MNIST: "
                f"expected 784, got {dimensions['level0']['input']}"
            )
            
        if dimensions['level2']['factor'] != 10:  # 10 digit classes
            raise ValueError(
                f"Invalid output dimension for MNIST: "
                f"expected 10, got {dimensions['level2']['factor']}"
            )
            
        self.logger.debug(
            f"✓ Validated hierarchy: {levels} levels with dimensions "
            f"{[dimensions[f'level{i}'] for i in range(levels)]}"
        )