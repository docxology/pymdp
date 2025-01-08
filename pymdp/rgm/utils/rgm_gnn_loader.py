"""
RGM GNN Loader
=============

Loads and parses GNN specifications for RGM.
Handles file loading, validation, and merging of specifications.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rgm_experiment_utils import RGMExperimentUtils
from rgm_gnn_validator import RGMGNNValidator
from rgm_gnn_parser import RGMGNNParser

class RGMGNNLoader:
    """Loads and processes GNN specifications"""
    
    def __init__(self):
        """Initialize GNN loader"""
        self.logger = RGMExperimentUtils.get_logger('gnn_loader')
        self.experiment = RGMExperimentUtils.get_experiment()
        self.validator = RGMGNNValidator()
        self.parser = RGMGNNParser()
        
    def load_gnn_specs(self) -> Dict[str, Dict]:
        """
        Load all GNN specifications.
        
        Returns:
            Dictionary of parsed GNN specifications
        """
        try:
            self.logger.info("Loading GNN specifications...")
            
            # Get GNN spec directory
            spec_dir = self.experiment['dirs']['gnn_specs']
            if not spec_dir.exists():
                raise FileNotFoundError(f"GNN spec directory not found: {spec_dir}")
                
            # Load base spec first
            base_spec = self._load_base_spec()
            
            # Load additional specs
            specs = {'base': base_spec}
            for spec_file in spec_dir.glob("*.gnn"):
                if spec_file.name != 'rgm_base.gnn':
                    spec_name = spec_file.stem
                    spec = self._load_spec(spec_file)
                    specs[spec_name] = spec
                    
            # Validate and merge specs
            merged_specs = self._process_specs(specs)
            
            # Save merged specs
            self._save_specs(merged_specs)
            
            return merged_specs
            
        except Exception as e:
            self.logger.error(f"Error loading GNN specs: {str(e)}")
            raise
            
    def _load_base_spec(self) -> Dict:
        """Load base GNN specification"""
        try:
            # First try experiment directory
            base_path = self.experiment['dirs']['gnn_specs'] / "rgm_base.gnn"
            
            # Fall back to package specs if not found
            if not base_path.exists():
                base_path = Path(__file__).parent.parent / "models" / "rgm_base.gnn"
                
            if not base_path.exists():
                raise FileNotFoundError("Base GNN spec not found")
                
            with open(base_path) as f:
                spec = json.load(f)
                
            # Validate base spec
            is_valid, messages = self.validator.validate_gnn_spec(spec, "rgm_base.gnn")
            if not is_valid:
                raise ValueError(f"Invalid base GNN spec: {'; '.join(messages)}")
                
            return spec
            
        except Exception as e:
            self.logger.error(f"Error loading base GNN spec: {str(e)}")
            raise
            
    def _load_spec(self, spec_path: Path) -> Dict:
        """Load single GNN specification"""
        try:
            with open(spec_path) as f:
                spec = json.load(f)
                
            # Validate spec
            is_valid, messages = self.validator.validate_gnn_spec(spec, spec_path.name)
            if not is_valid:
                raise ValueError(f"Invalid GNN spec {spec_path.name}: {'; '.join(messages)}")
                
            return spec
            
        except Exception as e:
            self.logger.error(f"Error loading GNN spec {spec_path}: {str(e)}")
            raise
            
    def _process_specs(self, specs: Dict[str, Dict]) -> Dict[str, Dict]:
        """Process and merge GNN specifications"""
        try:
            # Parse specs
            parsed_specs = {}
            for name, spec in specs.items():
                parsed = self.parser.parse_gnn_spec(spec)
                parsed_specs[name] = parsed
                
            # Merge specs
            merged = self._merge_specs(parsed_specs)
            
            # Validate merged specs
            is_valid, messages = self.validator.validate_gnn_spec(merged, "merged")
            if not is_valid:
                raise ValueError(f"Invalid merged specs: {'; '.join(messages)}")
                
            return merged
            
        except Exception as e:
            self.logger.error(f"Error processing GNN specs: {str(e)}")
            raise
            
    def _merge_specs(self, specs: Dict[str, Dict]) -> Dict:
        """Merge multiple GNN specifications"""
        try:
            # Start with base spec
            merged = specs['base'].copy()
            
            def deep_merge(d1: Dict, d2: Dict):
                """Recursively merge dictionaries"""
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                        
            # Merge additional specs
            for name, spec in specs.items():
                if name != 'base':
                    deep_merge(merged, spec)
                    
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging GNN specs: {str(e)}")
            raise
            
    def _save_specs(self, specs: Dict):
        """Save processed GNN specifications"""
        try:
            # Create output directory
            output_dir = self.experiment['dirs']['config'] / "processed_specs"
            output_dir.mkdir(exist_ok=True)
            
            # Save individual specs
            for name, spec in specs.items():
                spec_path = output_dir / f"{name}_processed.json"
                with open(spec_path, 'w') as f:
                    json.dump(spec, f, indent=2)
                    
            # Save merged specs
            merged_path = output_dir / "merged_specs.json"
            with open(merged_path, 'w') as f:
                json.dump(specs, f, indent=2)
                
            self.logger.info(f"Saved processed specs to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving GNN specs: {str(e)}")
            raise
            
    def validate_spec_compatibility(self, specs: Dict[str, Dict]) -> Tuple[bool, List[str]]:
        """Validate compatibility between specifications"""
        try:
            messages = []
            
            # Check hierarchy compatibility
            if not self._check_hierarchy_compatibility(specs):
                messages.append("Incompatible hierarchy specifications")
                
            # Check matrix shape compatibility
            if not self._check_matrix_compatibility(specs):
                messages.append("Incompatible matrix specifications")
                
            # Check learning parameter compatibility
            if not self._check_learning_compatibility(specs):
                messages.append("Incompatible learning parameters")
                
            return len(messages) == 0, messages
            
        except Exception as e:
            self.logger.error(f"Error validating spec compatibility: {str(e)}")
            raise
            
    def _check_hierarchy_compatibility(self, specs: Dict[str, Dict]) -> bool:
        """Check hierarchy compatibility between specs"""
        try:
            base_hierarchy = specs['base']['hierarchy']
            
            for name, spec in specs.items():
                if name != 'base':
                    hierarchy = spec.get('hierarchy', {})
                    
                    # Check level dimensions
                    if 'dimensions' in hierarchy:
                        for level, dims in hierarchy['dimensions'].items():
                            if level in base_hierarchy['dimensions']:
                                base_dims = base_hierarchy['dimensions'][level]
                                if dims != base_dims:
                                    self.logger.error(
                                        f"Dimension mismatch in {name} for {level}: "
                                        f"{dims} vs {base_dims}"
                                    )
                                    return False
                                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking hierarchy compatibility: {str(e)}")
            raise
            
    def _check_matrix_compatibility(self, specs: Dict[str, Dict]) -> bool:
        """Check matrix specification compatibility"""
        try:
            base_matrices = specs['base'].get('matrices', {})
            
            for name, spec in specs.items():
                if name != 'base':
                    matrices = spec.get('matrices', {})
                    
                    # Check matrix shapes
                    for matrix_name, matrix_spec in matrices.items():
                        if matrix_name in base_matrices:
                            base_spec = base_matrices[matrix_name]
                            if matrix_spec != base_spec:
                                self.logger.error(
                                    f"Matrix spec mismatch in {name} for {matrix_name}"
                                )
                                return False
                                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking matrix compatibility: {str(e)}")
            raise
            
    def _check_learning_compatibility(self, specs: Dict[str, Dict]) -> bool:
        """Check learning parameter compatibility"""
        try:
            base_learning = specs['base'].get('learning', {})
            
            for name, spec in specs.items():
                if name != 'base':
                    learning = spec.get('learning', {})
                    
                    # Check required parameters
                    required = ['active_learning', 'message_passing']
                    for param in required:
                        if param in learning and param in base_learning:
                            if learning[param] != base_learning[param]:
                                self.logger.error(
                                    f"Learning parameter mismatch in {name} for {param}"
                                )
                                return False
                                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking learning compatibility: {str(e)}")
            raise