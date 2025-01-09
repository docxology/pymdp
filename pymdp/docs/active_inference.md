# Active Inference in RGM

## Overview

The RGM implements active inference through:
1. Hierarchical belief updating
2. Precision-weighted prediction errors
3. Expected free energy minimization

## Components

### 1. Belief Updating
- Bottom-up recognition (perception)
- Top-down generation (prediction)
- Lateral message passing (context)

### 2. Precision Weighting
- Learned precision matrices
- Uncertainty modulation
- Attention allocation

### 3. Action Selection
- Expected free energy computation
- Policy evaluation
- Action sampling

## Implementation

### Message Passing
```python
def update_beliefs(self, observation):
    # Bottom-up pass (recognition)
    prediction_errors = []
    current = observation
    for level in self.hierarchy:
        # Compute prediction error
        predicted = level.generate(level.state)
        error = current - predicted
        
        # Weight by precision
        weighted_error = level.precision * error
        prediction_errors.append(weighted_error)
        
        # Update state
        level.state = level.recognize(current)
        current = level.state
        
    # Top-down pass (generation)
    for level in reversed(self.hierarchy):
        level.state = level.refine(level.state)
        
    return prediction_errors
```

### Active Learning
```python
def select_action(self, policies):
    # Compute expected free energy for each policy
    G = []
    for policy in policies:
        # Ambiguity term
        ambiguity = self.compute_ambiguity(policy)
        
        # Risk term
        risk = self.compute_risk(policy)
        
        # Novelty term
        novelty = self.compute_novelty(policy)
        
        # Combined EFE
        G.append(self.combine_efe_terms(ambiguity, risk, novelty))
        
    # Sample action using softmax
    action_probs = softmax(-G)  # Negative because we minimize free energy
    action = sample_action(action_probs)
    
    return action
``` 