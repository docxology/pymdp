"""
RGM GNN Specification Loader
===========================

Loads and validates GNN specifications for RGM.
Handles file loading, validation, and merging of specifications.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rgm_experiment_utils import RGMExperimentUtils
from rgm_gnn_validator import RGMGNNValidator

class RGMGNNSpecLoader:
    """Loads and validates GNN specifications"""
    
    def __init__(self):
        """Initialize GNN specification loader"""
        self.logger = RGMExperimentUtils.get_logger('gnn_spec')
        self.validator = RGMGNNValidator()
        self.experiment = RGMExperimentUtils.get_experiment()
        
    def load_gnn_specs(self) -> Dict:
        """
        Load and validate all GNN specifications.
        
        Returns:
            Dictionary of validated and merged specifications
        """
        try:
            # Load base specifications
            base_specs = self._load_base_specs()
            
            # Load additional specifications
            additional_specs = self._load_additional_specs()
            
            # Merge specifications
            merged_specs = self._merge_specifications(base_specs, additional_specs)
            
            # Save merged specifications
            self._save_merged_specs(merged_specs)
            
            return merged_specs
            
        except Exception as e:
            self.logger.error(f"Error loading GNN specifications: {str(e)}")
            raise
            
    def _load_base_specs(self) -> Dict:
        """Load required base specifications"""
        base_specs = {}
        
        # Define required base specs
        required_specs = [
            'rgm_base.gnn',
            'rgm_hierarchical_level.gnn',
            'rgm_mnist.gnn'
        ]
        
        # Load each required spec
        for spec_name in required_specs:
            spec_path = self._get_spec_path(spec_name)
            self.logger.info(f"Loading base spec: {spec_name}")
            
            try:
                # Load and validate spec
                spec = self._load_spec_file(spec_path)
                is_valid, messages = self.validator.validate_gnn_spec(spec, spec_name)
                
                if not is_valid:
                    raise ValueError(
                        f"Invalid base specification {spec_name}: "
                        f"{'; '.join(messages)}"
                    )
                    
                base_specs[spec_name] = spec
                
            except Exception as e:
                self.logger.error(f"Error loading {spec_name}: {str(e)}")
                raise
                
        return base_specs
        
    def _load_additional_specs(self) -> Dict:
        """Load optional additional specifications"""
        additional_specs = {}
        
        try:
            # Get all GNN spec files
            spec_dir = self.experiment['dirs']['gnn_specs']
            spec_files = list(spec_dir.glob("*.gnn"))
            
            # Filter out base specs
            base_files = {'rgm_base.gnn', 'rgm_hierarchical_level.gnn', 'rgm_mnist.gnn'}
            additional_files = [f for f in spec_files if f.name not in base_files]
            
            # Load each additional spec
            for spec_path in additional_files:
                spec_name = spec_path.name
                self.logger.info(f"Loading additional spec: {spec_name}")
                
                try:
                    # Load and validate spec
                    spec = self._load_spec_file(spec_path)
                    is_valid, messages = self.validator.validate_gnn_spec(spec, spec_name)
                    
                    if is_valid:
                        additional_specs[spec_name] = spec
                    else:
                        self.logger.warning(
                            f"Skipping invalid spec {spec_name}: {'; '.join(messages)}"
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Error loading {spec_name}: {str(e)}, skipping")
                    continue
                    
            return additional_specs
            
        except Exception as e:
            self.logger.error(f"Error loading additional specs: {str(e)}")
            raise
            
    def _merge_specifications(self, 
                            base_specs: Dict,
                            additional_specs: Dict) -> Dict:
        """Merge all specifications into final configuration"""
        try:
            # Start with base specification
            merged = base_specs['rgm_base.gnn'].copy()
            
            # Helper function for deep merge
            def deep_merge(d1: Dict, d2: Dict):
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        deep_merge(d1[k], v)
                    else:
                        d1[k] = v
                        
            # Merge hierarchical level spec
            deep_merge(merged, base_specs['rgm_hierarchical_level.gnn'])
            
            # Merge MNIST spec
            deep_merge(merged, base_specs['rgm_mnist.gnn'])
            
            # Merge additional specs
            for spec in additional_specs.values():
                deep_merge(merged, spec)
                
            # Validate merged specification
            is_valid, messages = self.validator.validate_gnn_spec(
                merged, 
                "merged_spec"
            )
            
            if not is_valid:
                raise ValueError(
                    f"Invalid merged specification: {'; '.join(messages)}"
                )
                
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging specifications: {str(e)}")
            raise
            
    def _get_spec_path(self, spec_name: str) -> Path:
        """Get path to specification file"""
        # First check experiment directory
        exp_path = self.experiment['dirs']['gnn_specs'] / spec_name
        if exp_path.exists():
            return exp_path
            
        # Then check package models directory
        pkg_path = Path(__file__).parent.parent / "models" / spec_name
        if pkg_path.exists():
            return pkg_path
            
        raise FileNotFoundError(f"GNN specification not found: {spec_name}")
        
    def _load_spec_file(self, spec_path: Path) -> Dict:
        """Load specification from file"""
        try:
            with open(spec_path) as f:
                spec = json.load(f)
                
            return spec
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {spec_path.name}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading {spec_path.name}: {str(e)}")
            
    def _save_merged_specs(self, merged_specs: Dict):
        """Save merged specifications"""
        try:
            # Save to experiment directory
            save_path = self.experiment['dirs']['gnn_specs'] / "merged_specs.json"
            
            with open(save_path, 'w') as f:
                json.dump(merged_specs, f, indent=2)
                
            self.logger.info(f"Saved merged specifications to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving merged specifications: {str(e)}")
            raise 