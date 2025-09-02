# PyMDP Core Utilities Guide

## Overview

The PyMDP Core Utilities provide a comprehensive, standardized interface to all PyMDP functionality, ensuring that textbook examples use real PyMDP methods exclusively. This guide covers the core utilities, their usage patterns, and best practices.

## Core Components

### 1. PyMDPCore Class

The `PyMDPCore` class provides static methods for all PyMDP operations:

```python
from pymdp_core import PyMDPCore

# Create agent
agent = PyMDPCore.create_agent(A, B, C, D)

# Infer states
qs = PyMDPCore.infer_states(agent, observation)

# Infer policies
q_pi, G = PyMDPCore.infer_policies(agent)

# Sample action
action = PyMDPCore.sample_action(agent)

# Run complete agent step
qs, q_pi, action = PyMDPCore.run_agent_step(agent, observation)
```

### 2. Convenience Functions

For backward compatibility and ease of use:

```python
from pymdp_core import create_agent, infer_states, infer_policies, sample_action

# Direct function calls
agent = create_agent(A, B, C, D)
qs = infer_states(agent, observation)
q_pi, G = infer_policies(agent)
action = sample_action(agent)
```

## Key Features

### Real PyMDP Method Usage

All utilities use authentic PyMDP methods exclusively:

- **Agent Creation**: Uses `pymdp.agent.Agent` class
- **State Inference**: Uses `agent.infer_states()` method
- **Policy Inference**: Uses `agent.infer_policies()` method
- **Action Sampling**: Uses `agent.sample_action()` method
- **VFE Computation**: Uses `pymdp.maths.calc_free_energy()` when available
- **EFE Computation**: Uses `pymdp.control.calc_expected_free_energy()` when available

### Comprehensive Validation

Built-in validation using real PyMDP utilities:

```python
from pymdp_core import validate_matrices

# Validate all matrices
results = validate_matrices(A, B, C, D)

# Check results
print(f"A matrix valid: {results['A']['valid']}")
print(f"B matrix valid: {results['B']['valid']}")
print(f"C matrix valid: {results['C']['valid']}")
print(f"D matrix valid: {results['D']['valid']}")
```

### Error Handling

Robust error handling with fallbacks:

```python
# VFE computation with fallback
vfe, components, posterior = PyMDPCore.compute_vfe(A, observation, prior)

# EFE computation with fallback
efe, components = PyMDPCore.compute_efe(A, B, C, beliefs, policy)
```

## Usage Patterns

### 1. Basic Agent Setup

```python
from pymdp_core import PyMDPCore
from pymdp.utils import obj_array_zeros

# Create matrices
A = obj_array_zeros([[3, 3]])
A[0] = np.eye(3)  # Perfect observation model

B = obj_array_zeros([[3, 3, 2]])
# ... populate B matrix

C = obj_array_zeros([[3]])
C[0] = np.array([-1, 0, 1])  # Linear preferences

D = obj_array_zeros([[3]])
D[0] = np.ones(3) / 3  # Uniform prior

# Create agent
agent = PyMDPCore.create_agent(A, B, C, D)
```

### 2. Agent Loop

```python
# Standard agent loop
for observation in observations:
    qs, q_pi, action = PyMDPCore.run_agent_step(agent, observation)
    
    # Process results
    print(f"Beliefs: {qs[0]}")
    print(f"Action: {action}")
```

### 3. VFE Analysis

```python
# Compute VFE for analysis
vfe, components, posterior = PyMDPCore.compute_vfe(A, observation, prior)

print(f"VFE: {vfe}")
print(f"Complexity: {components.get('complexity')}")
print(f"Accuracy: {components.get('accuracy')}")
```

### 4. EFE Analysis

```python
# Compute EFE for policy evaluation
beliefs = np.array([0.5, 0.3, 0.2])
policy = [0, 1]  # Two-step policy

efe, components = PyMDPCore.compute_efe(A, B, C, beliefs, policy)

print(f"EFE: {efe}")
print(f"Pragmatic Value: {components['pragmatic_value']}")
print(f"Epistemic Value: {components['epistemic_value']}")
```

## Best Practices

### 1. Always Use Core Utilities (thin orchestrator)

```python
# ✅ Good: Use core utilities
from pymdp_core import PyMDPCore
agent = PyMDPCore.create_agent(A, B, C, D)

# ❌ Avoid: Direct PyMDP imports in examples
from pymdp.agent import Agent
agent = Agent(A=A, B=B, C=C, D=D)
```

### 2. Validate Matrices (A/B/C/D)

```python
# Always validate matrices before use
validation_results = PyMDPCore.validate_matrices(A, B, C, D)

if not all(result['valid'] for result in validation_results.values()):
    print("Matrix validation failed!")
    return
```

### 3. Handle Errors Gracefully

```python
try:
    qs = PyMDPCore.infer_states(agent, observation)
except Exception as e:
    print(f"Inference failed: {e}")
    # Handle error appropriately
```

### 4. Use Standardized Output

```python
# Use ExampleRunner for consistent output
from example_utils import ExampleRunner

runner = ExampleRunner("my_example")
runner.save_results(results, "results.json")
runner.save_visualization(fig, "plot.png")
```

## Integration with Examples

### Thin Orchestrator Pattern

Examples should be thin orchestrators that use core utilities:

```python
def demonstrate_concept():
    """Demonstrate a PyMDP concept using core utilities."""
    
    # 1. Setup using core utilities
    agent = PyMDPCore.create_agent(A, B, C, D)
    
    # 2. Run analysis using core utilities
    results = []
    for observation in test_observations:
        qs, q_pi, action = PyMDPCore.run_agent_step(agent, observation)
        results.append({'observation': observation, 'beliefs': qs, 'action': action})
    
    # 3. Validate using core utilities
    validation = PyMDPCore.validate_matrices(A, B, C, D)
    
    # 4. Return structured results
    return {
        'results': results,
        'validation': validation,
        'agent': agent
    }
```

### Standardized Structure

All examples should follow this structure:

1. **Imports**: Use core utilities
2. **Setup**: Create matrices and agent using core utilities
3. **Execution**: Run analysis using core utilities
4. **Validation**: Validate results using core utilities
5. **Output**: Save results using ExampleRunner

## Chapter Cross-Links

- Bayesian updating (VFE): `textbook/examples/02_bayes_rule.py`, `04_state_inference.py`, `05_sequential_inference.py`
- A/B model building: `textbook/examples/03_observation_models.py`, `03_observation_models_refactored.py`, `07_transition_models.py`
- EFE and control: `textbook/examples/08_preferences_and_control.py`, `09_policy_inference.py`
- POMDP agents: `textbook/examples/10_simple_pomdp.py`, `11_gridworld_pomdp.py`, `12_tmaze_pomdp.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src directory is in Python path
2. **Matrix Validation Failures**: Check matrix normalization
3. **Inference Failures**: Verify observation format (integers, not one-hot)
4. **Agent Creation Failures**: Check matrix dimensions and types

### Debug Mode

Enable debug mode for detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
agent = PyMDPCore.create_agent(A, B, C, D, verbose=True)
```

## Performance Considerations

### Efficient Usage

1. **Reuse Agents**: Create agent once, use multiple times
2. **Batch Operations**: Process multiple observations together
3. **Memory Management**: Use appropriate data types
4. **Validation**: Validate once, use multiple times

### Optimization Tips

```python
# ✅ Good: Reuse agent
agent = PyMDPCore.create_agent(A, B, C, D)
for obs in observations:
    qs = PyMDPCore.infer_states(agent, obs)

# ❌ Avoid: Recreate agent
for obs in observations:
    agent = PyMDPCore.create_agent(A, B, C, D)  # Inefficient
    qs = PyMDPCore.infer_states(agent, obs)
```

## Conclusion

The PyMDP Core Utilities provide a robust, standardized interface to all PyMDP functionality. By using these utilities, examples ensure:

- **Authenticity**: Real PyMDP methods exclusively
- **Consistency**: Standardized patterns across examples
- **Reliability**: Comprehensive error handling and validation
- **Maintainability**: Centralized utilities for easy updates

For more information, see the individual utility documentation and example implementations.
