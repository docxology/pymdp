# GNN Specifications

## Directory Structure

The RGM requires the following GNN specification files:

1. `rgm_base.gnn`
   - Base configuration
   - Architecture parameters
   - Free energy components

2. `rgm_mnist.gnn`
   - MNIST-specific settings
   - Data preprocessing
   - Evaluation metrics

3. `rgm_message_passing.gnn`
   - Message passing rules
   - Belief propagation
   - Update schedules

4. `rgm_hierarchical_level.gnn`
   - Level specifications
   - Matrix dimensions
   - Activation functions

## File Format

GNN specifications use YAML format:

```yaml
# Example rgm_base.gnn
matrices:
  recognition:
    A0: [784, 256]  # Input → Level 0
    A1: [256, 64]   # Level 0 → Level 1
    A2: [64, 16]    # Level 1 → Level 2
    
  generative:
    B0: [256, 784]  # Level 0 → Input
    B1: [64, 256]   # Level 1 → Level 0
    B2: [16, 64]    # Level 2 → Level 1
    
  lateral:
    D0: [256, 256]  # Level 0 context
    D1: [64, 64]    # Level 1 context
    D2: [16, 16]    # Level 2 context

initialization:
  method: "orthogonal"
  gain: 1.0
  
free_energy:
  accuracy:
    weight: 1.0
  complexity:
    weight: 0.1
  entropy:
    weight: 0.01
```

## Validation

The GNN loader validates:
1. Directory structure
2. Required files
3. File contents
4. Matrix dimensions
5. Free energy components 