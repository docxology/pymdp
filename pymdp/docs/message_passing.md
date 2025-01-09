# Message Passing in RGM

## Overview

The RGM implements belief propagation through three types of messages:

1. Bottom-up Messages (Recognition)
   - Carry prediction errors upward
   - Precision-weighted by uncertainty
   - Update beliefs about causes

2. Top-down Messages (Generation)
   - Carry predictions downward
   - Generate expected observations
   - Implement prior constraints

3. Lateral Messages (Context)
   - Share information within levels
   - Modulate prediction errors
   - Implement empirical priors

## Implementation

### Message Types

1. Prediction Error Messages:
```python
def compute_prediction_error(observation, prediction, precision):
    """Compute precision-weighted prediction error."""
    error = observation - prediction
    weighted_error = precision * error
    return weighted_error
```

2. Prediction Messages:
```python
def generate_prediction(state, generative_matrix):
    """Generate predictions from current state."""
    prediction = torch.matmul(state, generative_matrix)
    return prediction
```

3. Context Messages:
```python
def update_context(state, lateral_matrix, iterations=5):
    """Update state using lateral connections."""
    current = state
    for _ in range(iterations):
        context = torch.matmul(current, lateral_matrix)
        current = torch.sigmoid(context)
    return current
```

### Message Passing Schedule

1. Forward Pass:
```python
def forward_pass(self, observation):
    """Bottom-up pass through hierarchy."""
    current = observation
    states = []
    for level in self.hierarchy:
        # Compute prediction error
        prediction = level.generate_prediction()
        error = self.compute_prediction_error(
            current, prediction, level.precision
        )
        
        # Update state
        state = level.update_state(error)
        states.append(state)
        
        # Pass to next level
        current = state
        
    return states
```

2. Backward Pass:
```python
def backward_pass(self, states):
    """Top-down pass through hierarchy."""
    for level, state in zip(reversed(self.hierarchy), reversed(states)):
        # Generate prediction
        prediction = level.generate_prediction(state)
        
        # Update lower level
        level.update_lower_level(prediction)
```

3. Lateral Pass:
```python
def lateral_pass(self, states):
    """Within-level message passing."""
    for level, state in zip(self.hierarchy, states):
        # Update context
        context = level.update_context(state)
        
        # Modulate state
        level.modulate_state(context)
``` 