# Renormalization Generative Model (RGM)

## Overview

This implementation provides a hierarchical generative model based on renormalization group principles and the Free Energy Principle. The model learns to:

1. Extract hierarchical features from data through successive coarse-graining
2. Generate samples through top-down message passing
3. Perform inference through bidirectional message passing
4. Minimize variational free energy during learning

## Architecture

The model consists of multiple hierarchical levels, each containing:

- Recognition model (bottom-up)
- Generative model (top-down) 
- Lateral connections within each level
- Message passing between levels

### Matrix Structure

Each level l has three key matrices:
- A[l]: Recognition weights (bottom-up)
- B[l]: Generative weights (top-down)
- D[l]: Lateral connections (within-level)

### Dimensions

For MNIST data (28x28 pixels):
- Input: 784 dimensions (flattened 28x28)
- Level 0: 784 → 256 dimensions
- Level 1: 256 → 64 dimensions 
- Level 2: 64 → 16 dimensions

## Training

The model is trained to:
1. Minimize reconstruction error
2. Regularize latent representations
3. Maintain consistency across levels
4. Minimize variational free energy

## Usage

```python
from rgm.Run_RGM import RGMPipeline

# Initialize and run pipeline
pipeline = RGMPipeline()
pipeline.run()
```

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks.
3. Wilson, K. G. (1975). The renormalization group: Critical phenomena and the Kondo problem. 