"""
RGM GNN Specification Validator
===========================

Validates GNN specifications for consistency and completeness.
"""

from typing import Dict, List
from pathlib import Path

class RGMGNNValidator:
    """Validates GNN specifications for the RGM pipeline."""
    
    @staticmethod
    def validate_hierarchy_consistency(specs: List[Dict]) -> bool:
        """
        Validate that hierarchy specifications are consistent across files.
        
        Args:
            specs: List of loaded GNN specifications
            
        Returns:
            True if consistent, raises ValueError otherwise
        """
        hierarchy_specs = [
            spec['hierarchy'] for spec in specs 
            if 'hierarchy' in spec
        ]
        
        if not hierarchy_specs:
            raise ValueError("No hierarchy specifications found")
            
        # Check levels consistency
        levels = set(spec['levels'] for spec in hierarchy_specs)
        if len(levels) > 1:
            raise ValueError(f"Inconsistent number of levels: {levels}")
            
        # Check dimensions consistency
        base_dims = hierarchy_specs[0]['dimensions']
        for spec in hierarchy_specs[1:]:
            if spec['dimensions'] != base_dims:
                raise ValueError("Inconsistent level dimensions across specifications")
                
        return True
        
    @staticmethod
    def validate_matrix_consistency(specs: List[Dict]) -> bool:
        """Validate matrix specifications consistency."""
        matrix_specs = [
            spec['matrices'] for spec in specs 
            if 'matrices' in spec
        ]
        
        if not matrix_specs:
            raise ValueError("No matrix specifications found")
            
        # Check matrix shapes consistency
        base_matrices = matrix_specs[0]
        for spec in matrix_specs[1:]:
            for matrix_type in ['recognition', 'generative', 'lateral']:
                if matrix_type not in spec:
                    continue
                    
                if spec[matrix_type] != base_matrices[matrix_type]:
                    raise ValueError(
                        f"Inconsistent {matrix_type} matrix shapes "
                        "across specifications"
                    )
                    
        return True