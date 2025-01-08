# RGM GNN Technical Specification
===============================

## Overview

The RGM (Recursive Generative Model) GNN framework implements a hierarchical generative model for MNIST digit recognition using message passing and active learning. This document outlines the technical specifications and architecture.

## Core Components

### 1. Pipeline Structure

```
pymdp/rgm/
├── models/               # GNN specifications
│   ├── rgm_base.gnn     # Base model architecture
│   ├── rgm_mnist.gnn    # MNIST-specific configuration
│   └── rgm_active_learning.gnn
├── utils/               # Utility modules
│   ├── rgm_core_utils.py
│   ├── rgm_matrix_*.py  # Matrix operations
│   └── rgm_gnn_*.py     # GNN handling
└── Run_RGM.py          # Main entry point
```

### 2. Matrix Hierarchy

- **A matrices**: State transitions between levels
- **B matrices**: Factor transitions within levels
- **D matrices**: Prior distributions
- **E matrix**: Output mapping

### 3. Numerical Properties

- Condition number threshold: 1e4
- Numerical stability epsilon: 1e-12
- Normalization constraints:
  - Column normalization for A, B matrices
  - Sum-to-one for D, E matrices
  - Non-negativity for all matrices

## Implementation Details

### 1. Message Passing

```python
class RGMMessagePassing:
    def run_message_passing(self, beliefs, matrices, precision):
        # Bottom-up pass
        # Top-down pass
        # Convergence check
```

Key Features:
- Bidirectional message passing
- Precision-weighted updates
- Damping factor: 0.9
- Convergence threshold: 1e-5

### 2. Active Learning

```json
{
    "active_learning": {
        "beta": 0.2,
        "max_precision": 50.0,
        "precision_growth": "exponential",
        "growth_rate": 1.1
    }
}
```

### 3. Matrix Operations

- SVD-based conditioning
- Hierarchical normalization
- Sparsity control
- Numerical stability checks

### 4. Data Processing

```python
class RGMMNISTDataset:
    # Custom MNIST dataset with:
    # - Resizing to 32x32
    # - Normalization
    # - Data augmentation
```

## Configuration Schema

### 1. Model Configuration

```json
{
    "hierarchy": {
        "n_levels": 4,
        "dimensions": {
            "level0": {"state": 1024, "factor": 256},
            "level1": {"state": 256, "factor": 64},
            "level2": {"state": 64, "factor": 16},
            "level3": {"state": 16, "factor": 10}
        }
    }
}
```

### 2. Learning Parameters

```json
{
    "learning": {
        "precision_init": 1.0,
        "learning_rate": 0.1,
        "message_passing": {
            "max_iterations": 5,
            "convergence_threshold": 1e-5
        }
    }
}
```

## Error Handling

1. **Validation Checks**:
   - Matrix properties
   - Configuration consistency
   - Numerical stability

2. **Error Recovery**:
   - Checkpoint saving
   - State persistence
   - Error logging

## Performance Metrics

1. **Training Metrics**:
   - ELBO (Evidence Lower BOund)
   - Classification accuracy
   - Confusion matrix

2. **Validation Metrics**:
   - Per-class accuracy
   - F1 scores
   - Early stopping criteria

## Visualization

1. **Matrix Analysis**:
   - Singular value spectra
   - Condition number tracking
   - Sparsity patterns

2. **Learning Curves**:
   - ELBO history
   - Accuracy progression
   - Precision adaptation

## Directory Structure

```
experiments/
├── {timestamp}/
│   ├── logs/
│   ├── render/
│   │   ├── matrices/
│   │   ├── config/
│   │   └── visualizations/
│   ├── simulation/
│   │   ├── checkpoints/
│   │   └── results/
│   └── analysis/
```

## Usage Example

```python
# Initialize experiment
experiment = RGMExperimentUtils.initialize_experiment(
    name="rgm_mnist",
    base_dir=Path("experiments")
)

# Run pipeline
pipeline = RGMPipelineManager()
results = pipeline.execute_pipeline()
```

## Dependencies

- NumPy >= 1.19.0
- PyTorch >= 1.7.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0

## Future Developments

1. **Planned Features**:
   - Multi-GPU support
   - Additional datasets
   - Advanced visualization tools

2. **Optimizations**:
   - Matrix operation efficiency
   - Memory management
   - Parallel processing

## References

1. Active Inference and Learning
2. Hierarchical Message Passing
3. Numerical Methods for Matrix Operations