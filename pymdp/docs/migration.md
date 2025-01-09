# Migration Guide

## Version 2.0 Changes

### Deprecated Methods

1. `RGMGNNLoader.load_gnn_specs`
   - Deprecated in favor of `load_specifications`
   - Will be removed in version 3.0
   - Migration example:
   ```python
   # Old code
   loader = RGMGNNLoader()
   specs = loader.load_gnn_specs(gnn_dir)
   
   # New code
   loader = RGMGNNLoader()
   specs = loader.load_specifications(gnn_dir)
   ```

### API Changes

1. GNN Specification Format
   - Now uses YAML exclusively
   - JSON format deprecated
   - Migration example:
   ```yaml
   # Old JSON format
   {
     "matrices": {
       "recognition": {"A0": [784, 256]}
     }
   }
   
   # New YAML format
   matrices:
     recognition:
       A0: [784, 256]
   ```

2. Matrix Initialization
   - Now supports multiple initialization methods
   - Added precision matrix initialization
   - Example:
   ```yaml
   initialization:
     method: "orthogonal"
     gain: 1.0
     precision:
       initial: 1.0
       learn_rate: 0.01
   ```

## Future Changes

1. Version 3.0 (Planned)
   - Remove deprecated methods
   - Full YAML migration
   - Enhanced error messages
   - Improved validation

2. Version 4.0 (Proposed)
   - Dynamic hierarchy support
   - Automated dimension inference
   - Extended free energy components 