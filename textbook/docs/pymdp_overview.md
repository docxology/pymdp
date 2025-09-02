# PyMDP Package Overview

## Introduction

PyMDP is a Python implementation of active inference and message passing algorithms for partially observable Markov decision processes (POMDPs). It provides a comprehensive toolkit for building, training, and deploying active inference agents.

## Package Structure

```text
pymdp/
├── __init__.py          # Main package exports
├── agent.py             # High-level Agent class
├── inference.py         # State inference algorithms
├── control.py           # Action selection and planning
├── learning.py          # Parameter learning methods
├── utils.py            # Utility functions and data structures
├── maths.py            # Mathematical operations
├── algos/              # Core algorithms
│   ├── fpi.py          # Fixed-point iteration
│   └── mmp.py          # Marginal message passing
├── envs/               # Environment implementations  
│   ├── grid_worlds.py  # Grid world environments
│   └── tmaze.py        # T-maze environment
└── rgm/                # Relational Generative Models
    └── ...             # RGM-specific modules
```

## Core Components

### 1. Data Structures

#### Object Arrays
PyMDP uses object arrays to handle multi-dimensional probability distributions:

```python
from pymdp.utils import obj_array_zeros, obj_array_uniform

# Create array of zeros with specified shapes
A = obj_array_zeros([[3, 2], [4, 2]])  # Two arrays: (3,2) and (4,2)

# Create uniform distributions
prior = obj_array_uniform([2, 3])  # Two factors with 2 and 3 states
```

#### Key Functions
- `obj_array_zeros(shapes)`: Create arrays of zeros
- `obj_array_uniform(shapes)`: Create uniform distributions  
- `is_obj_array(arr)`: Check if array is object array

### 2. Mathematical Operations

```python
from pymdp.maths import softmax, dot, kl_divergence

# Softmax with temperature
probs = softmax(logits, temperature=1.0)

# Dot product operations
result = dot(matrix, vector)

# KL divergence
kl = kl_divergence(p, q)
```

### 3. Inference Module

#### State Inference
```python
from pymdp.inference import infer_states, update_posterior_states

# Basic inference
posterior = infer_states(observations, A_matrix, prior)

# With additional parameters
posterior = infer_states(
    observations,
    A_matrix, 
    prior,
    num_iter=16,        # Number of iterations
    precision=1.0       # Precision parameter
)
```

#### Message Passing
```python
from pymdp.algos.mmp import marginal_message_passing

# Advanced message passing
qs = marginal_message_passing(A, obs, prior, num_iter=10)
```

### 4. Control Module

#### Action Selection
```python
from pymdp.control import sample_action, infer_policies

# Sample action from policy distribution
action = sample_action(policy_probs)

# Infer policies
policies = [[0, 1], [1, 0], [0, 0]]  # Sequences of actions
q_pi, G = infer_policies(A, B, C, policies, qs)
```

### 5. Learning Module

#### Parameter Learning
```python
from pymdp.learning import update_obs_model, update_trans_model

# Update observation model
A_updated = update_obs_model(A, obs, qs)

# Update transition model  
B_updated = update_trans_model(B, actions, qs_prev, qs_current)
```

### 6. Agent Class

The `Agent` class provides a high-level interface:

```python
from pymdp import Agent

# Create agent with model specification
agent = Agent(A=A, B=B, C=C, D=D)

# Perception step
beliefs = agent.infer_states(observation)

# Action step  
action = agent.sample_action()

# Learning step
agent.update_A(observation)
agent.update_B(action)
```

## Usage Patterns

### Basic Active Inference Loop

```python
import numpy as np
from pymdp import Agent
from pymdp.utils import obj_array_zeros

# 1. Define generative model
num_obs, num_states, num_actions = 4, 3, 2

# Observation model
A = obj_array_zeros([[num_obs, num_states]])
A[0] = np.random.rand(num_obs, num_states)
A[0] = A[0] / A[0].sum(axis=0, keepdims=True)  # Normalize

# Transition model
B = obj_array_zeros([[num_states, num_states, num_actions]])  
for a in range(num_actions):
    B[0][:, :, a] = np.random.rand(num_states, num_states)
    B[0][:, :, a] = B[0][:, :, a] / B[0][:, :, a].sum(axis=0, keepdims=True)

# Preferences and priors
C = obj_array_zeros([num_obs])
C[0] = np.random.randn(num_obs)

D = obj_array_zeros([num_states])
D[0] = np.ones(num_states) / num_states

# 2. Create agent
agent = Agent(A=A, B=B, C=C, D=D)

# 3. Active inference loop
for t in range(num_steps):
    # Observe environment
    observation = env.step(action)
    
    # Infer states
    beliefs = agent.infer_states([observation])
    
    # Select action
    action = agent.sample_action()
    
    # Optional: learn
    agent.update_A([observation])
    agent.update_B([action])
```

### Multi-Modal Observations

```python
# Multiple observation modalities
A = obj_array_zeros([[3, 2], [2, 2]])  # Vision and proprioception

# Multi-modal observations
obs = [1, 0]  # Visual obs = 1, proprioceptive obs = 0

qs = infer_states(obs, A, prior)
```

### Hierarchical Models

```python
# Multiple state factors (hierarchical)
A = obj_array_zeros([[4, 2, 3]])  # Obs depends on two state factors

# Prior over factors
prior = obj_array_uniform([2, 3])

qs = infer_states([obs], A, prior)
# Returns: [qs_factor1, qs_factor2]
```

## Advanced Features

### Custom Algorithms
- Fixed-point iteration (FPI)
- Marginal message passing (MMP)  
- Variational message passing (VMP)

### Environment Integration
- Grid world environments
- T-maze tasks
- Custom environment wrappers

### Visualization Tools
- Policy visualization
- Belief state plotting
- Learning curve analysis

## Best Practices

1. **Model Specification**
   - Ensure probability matrices are properly normalized
   - Use appropriate precision parameters
   - Validate model dimensions

2. **Numerical Stability**
   - Use log-space computations when possible
   - Handle edge cases (zero probabilities)
   - Set appropriate convergence criteria

3. **Performance Optimization**
   - Use vectorized operations
   - Cache computations when possible
   - Profile critical sections

## Common Pitfalls

1. **Dimension Mismatches**: Ensure A, B, C matrices have compatible dimensions
2. **Normalization**: Probability matrices must sum to 1 along appropriate axes
3. **Initialization**: Poor initialization can lead to local minima in inference
4. **Precision**: Too high precision can cause numerical instability

## See Also

### Glossary
- **A (Observation Model)**: Likelihood `P(o|s)` mapping hidden states to observations.
- **B (Transition Model)**: Dynamics `P(s'|s,a)` mapping previous states and actions to next states.
- **C (Preferences)**: Prior preferences over observations (often in log space).
- **D (Initial States)**: Prior over initial hidden states.
- **VFE (Variational Free Energy)**: `F = Complexity - Accuracy`, minimized during inference.
- **EFE (Expected Free Energy)**: Policy-level objective mixing risk and epistemic value.
- **Policy (π)**: Action sequence evaluated via EFE.

### Navigation
- Quick theory and chapters: `active_inference_basics.md`
- Core utilities and agent loop: `pymdp_core_guide.md`
- Validation and runner: `validation_guide.md`
- Worked examples directory: `../examples/`

## References

1. PyMDP GitHub Repository: https://github.com/infer-actively/pymdp
2. Heins, C., et al. (2022). pymdp: A Python library for active inference in discrete state spaces
3. Friston, K., et al. (2017). Active inference: a process theory
