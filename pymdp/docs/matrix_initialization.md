# Matrix Initialization in RGM

## Overview

The RGM uses carefully initialized matrices to implement the Free Energy Principle:

1. Recognition Matrices (A)
   - Orthogonal initialization for efficient gradient flow
   - Row normalization for proper belief encoding
   - Dimensionality reduction for hierarchical compression

2. Generative Matrices (B)
   - Transpose of recognition matrices initially
   - Column normalization for proper prediction generation
   - Dimensionality expansion for reconstruction

3. Lateral Matrices (D)
   - Identity plus small noise initially
   - Symmetric positive definite for precision properties
   - Square dimensions for proper uncertainty weighting

## Implementation

### Matrix Types

1. Recognition Matrices:
```python
# Initialize recognition matrix A0
A0 = initialize_orthogonal((784, 256))  # Input → Level 0
A1 = initialize_orthogonal((256, 64))   # Level 0 → Level 1
A2 = initialize_orthogonal((64, 16))    # Level 1 → Level 2
```

2. Generative Matrices:
```python
# Initialize generative matrix B0 as transpose of A0
B0 = A0.t()  # Level 0 → Input reconstruction
B1 = A1.t()  # Level 1 → Level 0 reconstruction
B2 = A2.t()  # Level 2 → Level 1 reconstruction
```

3. Lateral Matrices:
```python
# Initialize lateral connections with identity + noise
D0 = initialize_identity_plus_noise((256, 256))  # Level 0
D1 = initialize_identity_plus_noise((64, 64))    # Level 1
D2 = initialize_identity_plus_noise((16, 16))    # Level 2
```

## Free Energy Considerations

The initialization methods are chosen to facilitate free energy minimization:

1. Orthogonal Initialization
   - Preserves gradient magnitudes
   - Enables efficient belief propagation
   - Supports hierarchical feature extraction

2. Normal Initialization
   - Provides good starting point for learning
   - Allows flexibility in weight distribution
   - Controls initial complexity cost

3. Identity Plus Noise
   - Starts with reasonable precision estimates
   - Enables learning of uncertainty structure
   - Supports message passing between levels 