# RGM Matrix Naming Convention

## Matrix Types and Prefixes

The RGM uses three types of matrices with standardized naming:

1. Recognition Matrices (R, formerly A)
   - Bottom-up information flow
   - R0, R1, R2 for each level
   - Extract features and compose parts
   - Maps input → state at each level

2. Generative Matrices (G, formerly B)
   - Top-down information flow
   - G0, G1, G2 for each level
   - Generate predictions and reconstruct input
   - Maps state → input at each level

3. Lateral Matrices (L, formerly D)
   - Within-level connections
   - L0, L1, L2 for each level
   - Provide contextual information
   - Square matrices for state refinement

## Matrix Dimensions

Each matrix connects specific dimensions in the hierarchy:

```
Level 2 (Top)
R2: [16, 64]   - Final classification
G2: [64, 16]   - Digit generation
L2: [16, 16]   - High-level context

Level 1 (Middle)
R1: [64, 256]  - Part composition
G1: [256, 64]  - Part generation
L1: [64, 64]   - Mid-level context

Level 0 (Bottom)
R0: [256, 784] - Feature extraction
G0: [784, 256] - Image reconstruction
L0: [256, 256] - Low-level context
```

## Legacy Support

For backward compatibility, the system supports both naming conventions:
- A/B/D: Original naming scheme
- R/G/L: Current standardized naming

The model initializer automatically converts between these conventions. 