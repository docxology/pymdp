# RGM Model State Management

## Overview

The RGM model maintains several types of state during training and inference:

1. Matrices (Trainable Parameters)
   - Recognition (R0, R1, R2): Bottom-up feature extraction
     - Transform input data into increasingly abstract representations
     - Trained through backpropagation
   - Generative (G0, G1, G2): Top-down prediction generation
     - Generate predictions at each level from higher-level states
     - Minimize prediction errors
   - Lateral (L0, L1, L2): Within-level context integration
     - Refine representations using contextual information
     - Enforce consistency constraints
   - All matrices are PyTorch Parameters for automatic differentiation

2. Level States
   - Level 0: Low-level features (256 units)
     - Processes raw pixel data (784 → 256)
     - ReLU activation for feature detection
     - Sparse coding of basic features
   - Level 1: Mid-level features (64 units)
     - Composes basic features into parts (256 → 64)
     - ReLU activation for part composition
     - Hierarchical abstraction
   - Level 2: High-level features (16 units)
     - Integrates parts into digit concepts (64 → 16)
     - ReLU activation for category formation
     - Final classification layer

3. Loss Components
   - Reconstruction Loss: MSE between input and predictions
   - Prediction Loss: MSE between level states and predictions
   - Sparsity Loss: L1 regularization on state activations
   - Total Loss: Weighted combination of all components

4. Training State
   - Training mode flag (for dropout, etc.)
   - Current epoch counter
   - Global step counter
   - Parameter gradients
   - Optimizer state
   - Loss history

## Neural Network Integration

The model state inherits from nn.Module to provide:
1. Automatic parameter registration
2. Forward/backward pass computation
3. GPU acceleration support
4. Integration with PyTorch training tools
5. Loss computation and optimization

## Forward Pass

The forward computation includes:
1. Bottom-up recognition pass
   - Input → Level 0 → Level 1 → Level 2
   - Feature extraction and composition
2. Top-down generation pass
   - Level 2 → Level 1 → Level 0 → Reconstruction
   - Prediction generation
3. Prediction error computation
   - At each level of the hierarchy
   - Error-driven learning
4. Loss computation
   - Reconstruction quality
   - Prediction accuracy
   - Representation sparsity

## State Management

All state tensors are automatically:
1. Moved to specified device (CPU/CUDA)
2. Maintained in consistent precision (float32)
3. Updated atomically during training
4. Tracked for gradient computation
5. Optimized through backpropagation

## Error Handling

The state manager provides:
1. Validation of matrix and state dimensions
2. Safe access to matrices and states
3. Proper device placement
4. Informative error messages
5. Training diagnostics

## Checkpointing

State can be saved and loaded via:
1. state_dict(): Get complete state snapshot
2. load_state_dict(): Restore from snapshot
3. All tensors maintain device placement
4. Training state is preserved
5. Optimizer state included 

## Visualization

The model provides visualization capabilities:
1. Input-reconstruction pairs during training
2. Feature visualization at each level
3. Reconstruction error tracking
4. Loss component monitoring
5. State activation patterns

### MNIST Visualization

For MNIST specifically:
1. Original vs reconstructed digit pairs
2. Feature detector visualization
3. Hierarchical representation analysis
4. Confusion matrix on test set
5. t-SNE of learned representations

## Training Monitoring

The training process tracks:
1. Loss components (reconstruction, prediction, sparsity)
2. Reconstruction quality metrics
3. Feature activation statistics
4. Gradient magnitudes
5. Learning rate scheduling

## Implementation Notes

1. The model is implemented as a PyTorch nn.Module
2. Forward pass returns dictionary with all relevant outputs
3. Visualization methods integrated with training loop
4. Automatic logging of metrics and images
5. Configurable visualization frequency 