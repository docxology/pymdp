# Generalized Notation Notation (GNN) Notation

## Overview
The Generalized Notation Notation (GNN) notation is a flexible format for specifying active inference models. It provides a structured way to define generative models, including state spaces, observation mappings, transition dynamics, and preferences.

## Specification

### Basic Structure
```json
{
    "modelName": "model_identifier",
    "modelType": ["type1", "type2"],
    "stateSpace": {
        "factors": ["factor1", "factor2"],
        "sizes": [3, 2],
        "labels": [
            ["state1", "state2", "state3"],
            ["option1", "option2"]
        ]
    }
}
```

### Components

#### Model Metadata
- `modelName`: Unique identifier
- `modelType`: List of model characteristics
- `version`: Optional specification version

#### State Space
- `factors`: Named state factors
- `sizes`: Dimensions of each factor
- `labels`: Human-readable state names

#### Matrices
- `A`: Observation model
- `B`: Transition dynamics
- `C`: Preferences (agent only)
- `D`: Initial state priors

### Matrix Formats

#### A Matrix (Observation Model)
```python
{
    "type": "categorical",
    "dimensions": ["observation", "state"],
    "values": [
        [0.9, 0.1],
        [0.1, 0.9]
    ]
}
```

#### B Matrix (Transitions)
```python
{
    "type": "categorical",
    "dimensions": ["next_state", "current_state", "action"],
    "values": [
        [[0.8, 0.2], [0.2, 0.8]],
        [[0.7, 0.3], [0.3, 0.7]]
    ]
}
```

## Usage

### File Creation
```python
from pymdp.gnn.gnn_matrix_factory import GNNMatrixFactory

# Load GNN specification
model = GNNMatrixFactory.load_gnn("model.gnn")

# Generate matrices
A = model.generate_A_matrix()
B = model.generate_B_matrix()
```

### Validation
```python
from pymdp.gnn.gnn_utils import GNNUtils

# Validate GNN file
is_valid = GNNUtils.validate_gnn("model.gnn")

# Validate matrices
is_valid = GNNUtils.validate_matrices(A, B, C, D)
```

## Implementation

### Matrix Generation
1. Parse GNN specification
2. Validate structure and types
3. Generate numpy arrays
4. Apply normalization
5. Validate outputs

### Validation Rules
- Matrices must be properly normalized
- Dimensions must match state space
- Values must be valid probabilities
- Required components must exist

## File Types

### Environment GNN
```json
{
    "modelType": ["Environment"],
    "stateSpace": {...},
    "matrices": {
        "A": {...},
        "B": {...},
        "D": {...}
    }
}
```

### Agent GNN
```json
{
    "modelType": ["Agent"],
    "stateSpace": {...},
    "matrices": {
        "A": {...},
        "B": {...},
        "C": {...},
        "D": {...}
    }
}
```

## Utilities

### Matrix Operations
- Normalization
- Validation
- Format conversion
- Visualization

### File Operations
- Loading/saving
- Validation
- Default generation
- Format conversion

## Best Practices

1. **Naming Conventions**
   - Use descriptive model names
   - Follow consistent factor naming
   - Include clear state labels

2. **Structure**
   - Keep models modular
   - Separate environment/agent
   - Include documentation

3. **Validation**
   - Validate before use
   - Check normalizations
   - Verify dimensions

4. **Documentation**
   - Comment complex structures
   - Document assumptions
   - Include examples

## Dependencies
- Python 3.8+
- NumPy
- JSON

## References
- Active Inference Papers
- Matrix Operations Guide
- Probability Theory Basics


