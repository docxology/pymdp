# Theoretical Framework

## Free Energy Principle

The Free Energy Principle (FEP) posits that biological systems minimize their variational free energy, which can be decomposed into:

1. Accuracy term: How well the model explains the data
2. Complexity term: How much the model deviates from prior beliefs
3. Surprise term: How unexpected the data is under the model

## Renormalization Group Theory

RGM implements successive coarse-graining inspired by renormalization group theory:

1. Each level performs feature extraction at different scales
2. Information flows bidirectionally between levels
3. The hierarchy learns increasingly abstract representations

## Message Passing

The model implements three types of message passing:

1. Bottom-up (recognition): x[l] = f(A[l]x[l-1])
2. Top-down (generation): x̂[l-1] = g(B[l]x[l])
3. Lateral (within-level): x[l] = h(D[l]x[l])

Where:
- x[l]: Activities at level l
- A[l]: Recognition weights
- B[l]: Generative weights
- D[l]: Lateral connections
- f,g,h: Activation functions 

## Matrix Dimensions

The RGM uses a hierarchical structure with specific matrix dimensions:

### Level 0 (Input Level)
- A0: [784, 256] - Recognition (bottom-up)
- B0: [256, 784] - Generation (top-down)
- D0: [256, 256] - Lateral connections

### Level 1 (Intermediate Level)
- A1: [256, 64] - Recognition
- B1: [64, 256] - Generation
- D1: [64, 64] - Lateral connections

### Level 2 (Top Level)
- A2: [64, 16] - Recognition
- B2: [16, 64] - Generation
- D2: [16, 16] - Lateral connections

The dimensions follow a coarse-graining pattern:
1. Each level reduces dimensionality by ~4x
2. Lateral connections are always square
3. Recognition and generation matrices are transposes 