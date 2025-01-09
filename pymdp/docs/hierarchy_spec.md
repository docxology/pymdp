# RGM Hierarchy Specification

## Overview

The Renormalization Generative Model uses a hierarchical structure to process MNIST digits:

```
Level 2 (Top)
- Input: 64 (from Level 1 state)
- State: 16 (high-level features)
- Factor: 10 (digit classes)

Level 1 (Middle)
- Input: 256 (from Level 0 state)
- State: 64 (mid-level features)
- Factor: 32 (part compositions)

Level 0 (Bottom)
- Input: 784 (28x28 MNIST image)
- State: 256 (low-level features)
- Factor: 64 (basic features)
```

## Dimension Requirements

1. Input/State Connections:
   - Level N input must match Level N-1 state dimension
   - Enables proper information flow between levels

2. MNIST-Specific:
   - Level 0 input: 784 (28x28 pixels)
   - Level 2 factor: 10 (digit classes)

3. Dimension Reduction:
   - Each level reduces dimensionality
   - Captures increasingly abstract features 