# Active Inference Basics

## Introduction

Active inference is a unified framework for understanding perception, action, and learning in biological and artificial agents. It provides a principled approach to decision-making under uncertainty, grounded in the free energy principle.

## Core Principles

### The Free Energy Principle

The free energy principle states that any self-organizing system that maintains its organization over time must minimize its free energy. Free energy (F) provides an upper bound on surprise:

```
F ≥ -ln P(o)
```

Where:
- `o` represents observations
- `P(o)` is the probability of those observations under the agent's model

### Generative Models

Active inference agents maintain generative models that specify:

1. **Observation Model (A)**: `P(o_t | s_t)`
   - Likelihood of observations given hidden states
   - Matrix form: `A[o, s]` = probability of observation `o` from state `s`

2. **Transition Model (B)**: `P(s_t | s_{t-1}, a_{t-1})`
   - Dynamics of hidden states given actions
   - Tensor form: `B[s', s, a]` = probability of transitioning to `s'` from `s` via action `a`

3. **Prior Preferences (C)**: `P(o_t)`
   - Prior beliefs about preferred observations
   - Vector form: `C[o]` = log preference for observation `o`

4. **Initial States (D)**: `P(s_0)`
   - Prior beliefs about initial hidden states
   - Vector form: `D[s]` = initial probability of state `s`

## Mathematical Framework

### State Estimation

The agent maintains beliefs about hidden states `Q(s_t)` by minimizing free energy:

```
Q*(s_t) = arg min_Q F[Q(s_t)]
```

Where free energy decomposes into:
```
F = D_KL[Q(s_t) || P(s_t | o_{1:t-1})] - E_Q[ln P(o_t | s_t)]
```

This leads to the update rule:
```
Q(s_t) ∝ P(s_t | o_{1:t-1}) · P(o_t | s_t)^γ
```

Where `γ` is the precision parameter.

### Action Selection

Actions are selected to minimize expected free energy `G`:

```
G(π) = E_Q[F(o_τ, s_τ | π)]
```

Where `π` represents a policy (sequence of actions).

Expected free energy decomposes into:
1. **Risk**: Deviation from preferences
2. **Ambiguity**: Expected uncertainty about states

## PyMDP Implementation

### Basic Model Setup

```python
import numpy as np
import pymdp
from pymdp.utils import obj_array_zeros

# Define model dimensions
num_obs = 4      # Number of possible observations
num_states = 3   # Number of hidden states  
num_actions = 2  # Number of possible actions

# Observation model
A = obj_array_zeros([[num_obs, num_states]])
A[0] = np.array([[0.9, 0.1, 0.0],   # obs 0: likely from state 0
                 [0.1, 0.8, 0.1],   # obs 1: likely from state 1
                 [0.0, 0.1, 0.9],   # obs 2: likely from state 2
                 [0.0, 0.0, 0.0]])  # obs 3: impossible

# Transition model
B = obj_array_zeros([[num_states, num_states, num_actions]])
# Action 0: stay in place with some noise
B[0][:, :, 0] = np.array([[0.8, 0.1, 0.1],
                          [0.1, 0.8, 0.1], 
                          [0.1, 0.1, 0.8]])
# Action 1: deterministic transitions
B[0][:, :, 1] = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])

# Preferences (log probabilities)
C = obj_array_zeros([num_obs])
C[0] = np.array([2.0, 0.0, 0.0, -4.0])  # Prefer obs 0, avoid obs 3

# Prior over initial states
D = obj_array_zeros([num_states])
D[0] = np.array([1.0, 0.0, 0.0])  # Start in state 0
```

### Running Inference

```python
from pymdp.inference import infer_states

# Current observation
obs = [1]  # Observed state 1

# Infer current state
qs = infer_states(obs, A, D)
print(f"Posterior beliefs: {qs[0]}")
```

### Action Selection

```python
from pymdp.control import sample_action, infer_policies

# Define policies (action sequences)
policies = [[0], [1]]  # Single-step policies

# Infer policy preferences
q_pi, G = infer_policies(A, B, C, policies, qs)

# Sample action
action = sample_action(q_pi)
print(f"Selected action: {action}")
```

## Key Concepts Summary

1. **Perception as Inference**: Perception involves inferring hidden states that caused observations
2. **Action as Inference**: Actions are selected by inferring policies that minimize expected free energy
3. **Learning as Inference**: Model parameters are updated to minimize free energy over time
4. **Precision**: Controls the confidence in beliefs and the explore-exploit trade-off

## Next Steps

- Read [`pomdp_theory.md`](pomdp_theory.md) for mathematical details
- Explore [`model_specification.md`](model_specification.md) for practical implementation
- Work through examples in `../examples/` directory
- Study [`free_energy_principle.md`](free_energy_principle.md) for theoretical depth

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

2. Parr, T., & Friston, K. J. (2017). Uncertainty, epistemics and active inference. *Neural Computation*, 29(1), 1-31.

3. Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: a synthesis. *Journal of Mathematical Psychology*, 99, 102447.
