# RGM for MNIST Digit Recognition

## Overview

This implementation applies the Renormalization Generative Model (RGM) to MNIST digit recognition, demonstrating:

1. Hierarchical feature extraction through multi-level processing
2. Generative reconstruction via top-down inference
3. Active inference for robust digit classification
4. Bidirectional message passing between levels

## Architecture

### Input Processing
- MNIST digits: 28x28 pixels → 784-dimensional vector
- Normalized to [0,1] range and standardized (mean=0.1307, std=0.3081)
- Flattened for matrix operations
- Batch processing: [batch_size, 784]
- Optional data augmentation (rotation ±10°, translation ±10%)

### Hierarchical Levels

1. Level 0 (Low-level features)
   - Input: 784 dimensions (28x28 image)
   - State: 256 dimensions
   - Features: Edges, strokes, local patterns
   - Batch normalized activations
   - Dropout during training (rate=0.1)

2. Level 1 (Mid-level features)
   - Input: 256 dimensions
   - State: 64 dimensions
   - Features: Digit parts, curves, loops
   - Batch normalized activations
   - Dropout during training (rate=0.1)

3. Level 2 (High-level features)
   - Input: 64 dimensions
   - State: 16 dimensions
   - Features: Digit class, global structure
   - Batch normalized activations
   - Dropout during training (rate=0.1)

### Matrix Structure

Each level l has three types of matrices:

1. Recognition Matrices (R[l])
   - Bottom-up processing (feature extraction)
   - R0: [256, 784] - Edge and basic shape detection
   - R1: [64, 256] - Part composition
   - R2: [16, 64] - Digit classification
   - Initialized with Xavier/Glorot uniform initialization
   - Trainable bias terms initialized to zero

2. Generative Matrices (G[l])
   - Top-down processing (reconstruction)
   - G0: [784, 256] - Image reconstruction
   - G1: [256, 64] - Part generation
   - G2: [64, 16] - Digit generation
   - Initialized with Xavier/Glorot uniform initialization
   - Trainable bias terms initialized to zero

3. Lateral Matrices (L[l])
   - Within-level processing (feature refinement)
   - L0: [256, 256] - Low-level feature relationships
   - L1: [64, 64] - Mid-level feature relationships
   - L2: [16, 16] - High-level digit relationships
   - Initialized as identity matrices with small noise (scale=0.01)

## Training Process

### Forward Pass (Recognition)
1. Input Preparation
   - Normalize images to [0,1] and standardize
   - Flatten to 784-dimensional vectors
   - Apply data augmentation if enabled
   - Validate input dimensions

2. Feature Extraction
   - Apply recognition matrices sequentially
   - Process through lateral connections
   - Apply batch normalization and ReLU
   - Apply dropout during training

3. State Computation
   - Compute states at each level
   - Store for prediction error calculation

### Backward Pass (Generation)
1. State Reconstruction
   - Generate reconstructions top-down
   - Apply batch normalization and ReLU
   - Use sigmoid for final reconstruction

2. Loss Computation
   - Reconstruction loss (MSE)
   - State regularization (L2 norm)
   - Weighted combination (reconstruction: 1.0, regularization: 0.1)

### Training Configuration

1. Optimization
   - Adam optimizer
     - Learning rate: 0.001
     - Betas: (0.9, 0.999)
     - Weight decay: 1e-5
     - AMSGrad: disabled
   - Learning rate scheduling
     - Reduce on plateau
     - Monitor: validation loss
     - Factor: 0.5
     - Patience: 5
     - Minimum LR: 1e-6
     - Threshold: 1e-4
     - Cooldown: 0

2. Training Parameters
   - Batch size: 128
   - Number of epochs: 100
   - Minimum epochs: 10
   - Logging interval: 100 batches
   - Checkpoint interval: 5 epochs
   - Validation interval: 5 epochs

3. Early Stopping
   - Monitor: validation loss
   - Patience: 10 epochs
   - Minimum delta: 0.001
   - Minimum epochs: 10

4. Metrics Tracking
   - Training metrics:
     - Loss
     - MSE
     - PSNR
     - State sparsity
   - Validation metrics:
     - Loss
     - MSE
     - PSNR
     - State sparsity

### Checkpointing

1. Save Strategy
   - Best model (lowest validation loss)
   - Last model state
   - Regular intervals (every 5 epochs)
   - Includes:
     - Model state
     - Optimizer state
     - Training state
     - Configuration

2. Loading
   - Automatic device mapping
   - Optional optimizer state restoration
   - Training state restoration

## Visualization

1. Training Progress
   - Loss curves (train/val)
   - Accuracy curves
   - Learning rate changes
   - Metric evolution

2. Model Analysis
   - Confusion matrices
   - Latent space visualization
   - Sample reconstructions (10x10 grid)
   - State activation patterns

3. Output Format
   - PNG format
   - 300 DPI resolution
   - Regular updates (every 5 epochs)

## Logging

1. Console and File Logging
   - Level: INFO
   - Formatted timestamps
   - Component identification
   - Automatic rotation (10MB limit)
   - 5 backup files

2. TensorBoard Integration
   - Enabled by default
   - Metrics tracking
   - Image samples
   - Update frequency: 10 steps
   - Auto-flush: 10 seconds

## Implementation Details

1. Device Management
   - Automatic device selection
   - CUDA support when available
   - Fallback to CPU
   - Deterministic mode support

2. Reproducibility
   - Fixed random seed (42)
   - Deterministic operations
   - Documented configurations

3. Error Handling
   - Input validation
   - Dimension checking
   - Graceful degradation
   - Informative error messages 