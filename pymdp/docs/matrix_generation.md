# Matrix Generation in RGM

## Theoretical Background

The matrices in RGM implement key components of the Free Energy Principle:

1. Recognition Matrices (A):
   - Implement recognition model q(causes|data)
   - Transform observations to latent states
   - Learn efficient feature extractors

2. Generative Matrices (B):
   - Implement generative model p(data|causes)
   - Transform latent states to predictions
   - Learn to reconstruct observations

3. Lateral Matrices (D):
   - Implement precision-weighting
   - Modulate prediction errors
   - Learn uncertainty structure

## Generation Process

### 1. Initialization
- Orthogonal initialization for recognition/generative matrices (efficient message passing)
- Identity + noise for lateral connections (initial uniform precision)
- Controlled by initialization parameters in GNN specs

### 2. Constraints
- Row normalization for recognition matrices (proper belief encoding)
- Column normalization for generative matrices (proper prediction generation)
- Symmetry and positive definiteness for lateral connections (valid precision matrices)

### 3. Dimensions
Each level implements successive coarse-graining:
```
Level 0: 784 → 256 (Low-level features)
Level 1: 256 → 64  (Mid-level features)
Level 2: 64 → 16   (High-level causes)
```

### 4. Properties
- Recognition and generative matrices are transposes (conjugate priors)
- Lateral connections are symmetric positive definite (precision properties)
- Dimensions reduce by ~4x at each level (efficient compression)

## Free Energy Implementation

The matrices work together to minimize variational free energy:

1. Recognition Path:
   ```
   Input → A0 → D0 → A1 → D1 → A2 → D2 → Latent States
   ```

2. Generative Path:
   ```
   Latent States → B2 → D1 → B1 → D0 → B0 → Predictions
   ```

3. Prediction Errors:
   ```
   Error = Input - Predictions
   Weighted_Error = D * Error  # Precision weighting
   ```

## Visualization

The renderer generates visualizations for each matrix:
- Heatmaps showing weight patterns
- Color-coded connection strengths
- Saved in the experiment directory

## Usage

```python
from rgm.rgm_renderer import RGMRenderer

# Initialize renderer
renderer = RGMRenderer(exp_dir)

# Generate matrices
matrices = renderer.render_matrices()
``` 