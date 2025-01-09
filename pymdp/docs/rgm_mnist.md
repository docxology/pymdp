# RGM for MNIST Digit Recognition

## Overview

This implementation applies the Renormalization Generative Model (RGM) to MNIST digit recognition, demonstrating:

1. Hierarchical feature extraction
2. Generative reconstruction
3. Active inference for classification

## Architecture

### Input Processing
- MNIST digits: 28x28 pixels → 784-dimensional vector
- Normalized to [0,1] range
- Flattened for matrix operations

### Hierarchical Levels
1. Level 0 (Low-level features)
   - Input: 784 dimensions
   - Latent: 256 dimensions
   - Features: Edges, strokes, local patterns

2. Level 1 (Mid-level features)
   - Input: 256 dimensions
   - Latent: 64 dimensions
   - Features: Digit parts, curves, loops

3. Level 2 (High-level features)
   - Input: 64 dimensions
   - Latent: 16 dimensions
   - Features: Digit class, global structure

### Matrix Structure
Each level l has three key matrices:

1. Recognition Matrix (A[l])
   - Bottom-up processing
   - Maps input to latent representation
   - Learns feature detectors

2. Generative Matrix (B[l])
   - Top-down processing
   - Reconstructs lower level from latent
   - Learns generative model

3. Lateral Connections (D[l])
   - Within-level processing
   - Square matrix (NxN)
   - Learns feature relationships

## Training Process

1. Forward Pass (Recognition)
   - Input → Level 0 → Level 1 → Level 2
   - Extract hierarchical features
   - Compute latent states

2. Backward Pass (Generation)
   - Level 2 → Level 1 → Level 0 → Output
   - Reconstruct input
   - Update predictions

3. Update Step
   - Minimize free energy
   - Update matrix weights
   - Apply constraints

## Performance Metrics

1. Reconstruction Error
   - MSE between input and reconstruction
   - Measures generative model quality

2. Classification Accuracy
   - Using Level 2 features
   - 10-way digit classification

3. Free Energy
   - Accuracy term (reconstruction)
   - Complexity term (KL divergence)
   - Prior term (hierarchical consistency) 