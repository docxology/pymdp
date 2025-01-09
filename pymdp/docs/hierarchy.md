# Hierarchical Structure in RGM

## Overview

The RGM implements a hierarchical generative model with three levels:

1. Level 0 (Input Level)
   - Processes raw MNIST images (784 dimensions)
   - Extracts low-level features (edges, strokes)
   - Uses 256 latent dimensions

2. Level 1 (Intermediate Level)
   - Processes Level 0 features
   - Extracts mid-level features (digit parts)
   - Uses 64 latent dimensions

3. Level 2 (Top Level)
   - Processes Level 1 features
   - Extracts high-level features (digit identity)
   - Uses 16 latent dimensions

## Level Components

Each level contains:

1. Recognition Matrix (A)
   ```python
   def recognize(self, input_data):
       """Bottom-up feature extraction."""
       activity = torch.matmul(input_data, self.A)
       return self.activation_forward(activity)
   ```

2. Generative Matrix (B)
   ```python
   def generate(self, latent_state):
       """Top-down prediction generation."""
       prediction = torch.matmul(latent_state, self.B)
       return self.activation_backward(prediction)
   ```

3. Lateral Matrix (D)
   ```python
   def modulate(self, state):
       """Within-level contextual modulation."""
       context = torch.matmul(state, self.D)
       return self.activation_lateral(context)
   ```

## Level Interactions

1. Bottom-up Processing:
   ```
   Input → A0 → Level 0 → A1 → Level 1 → A2 → Level 2
   ```

2. Top-down Processing:
   ```
   Level 2 → B2 → Level 1 → B1 → Level 0 → B0 → Output
   ```

3. Lateral Processing (at each level):
   ```
   State → D → Context → Updated State
   ```

## Free Energy Minimization

Each level contributes to free energy minimization:

1. Accuracy Term:
   - Prediction errors between levels
   - Precision-weighted reconstruction

2. Complexity Term:
   - KL divergence at each level
   - Hierarchical regularization

3. Prior Term:
   - Top-level constraints
   - Empirical priors from context 