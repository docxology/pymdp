# PyMDP Textbook Documentation

## Overview

This documentation provides comprehensive guidance for the PyMDP textbook examples, focusing on the use of real PyMDP methods exclusively and the thin orchestrator pattern for example implementation.

## Documentation Structure

### Core Guides

1. **[PyMDP Core Guide](pymdp_core_guide.md)** - Comprehensive guide to PyMDP core utilities
2. **[Validation Guide](validation_guide.md)** - Guide to validation utilities and best practices
3. **[Active Inference Basics](active_inference_basics.md)** - Fundamental concepts
4. **[PyMDP Overview](pymdp_overview.md)** - Overview of PyMDP integration

### Key Concepts

#### Thin Orchestrator Pattern

Examples should be thin orchestrators that use comprehensive shared utilities:

```python
# ✅ Good: Thin orchestrator using shared utilities
from pymdp_core import PyMDPCore
from example_utils import ExampleRunner, MatrixBuilder

def demonstrate_concept():
    # Setup using shared utilities
    A = MatrixBuilder.create_observation_model(3, 3, "identity")
    agent = PyMDPCore.create_agent(A, B, C, D)
    
    # Execute using shared utilities
    results = []
    for obs in observations:
        qs, q_pi, action = PyMDPCore.run_agent_step(agent, obs)
        results.append({'observation': obs, 'beliefs': qs, 'action': action})
    
    return results
```

#### Real PyMDP Methods

All examples must use authentic PyMDP methods exclusively:

- **Agent Operations**: `agent.infer_states()`, `agent.infer_policies()`, `agent.sample_action()`
- **Core Utilities**: `PyMDPCore.create_agent()`, `PyMDPCore.compute_vfe()`, `PyMDPCore.compute_efe()`
- **Matrix Operations**: `pymdp.utils.obj_array_zeros()`, `pymdp.maths.softmax()`
- **Validation**: `validate_matrices()`, `validate_agent()`

#### Comprehensive Validation

All examples are validated to ensure:

1. **Method Usage**: Only real PyMDP methods are used
2. **Matrix Validation**: All matrices meet PyMDP requirements
3. **Agent Operations**: All agent operations work correctly
4. **Error Handling**: Robust error handling and fallbacks

## Quick Start

### 1. Basic Example Structure

```python
#!/usr/bin/env python3
"""
Example: [Title] - Refactored
============================

Brief description using PyMDP core utilities.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import PyMDP core utilities
from pymdp_core import PyMDPCore, create_agent, infer_states
from example_utils import ExampleRunner, MatrixBuilder
from validation import validate_matrices

# Setup
OUTPUT_DIR = Path(__file__).parent / "outputs" / "example_name"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
runner = ExampleRunner("example_name", OUTPUT_DIR)

def demonstrate_concept():
    """Demonstrate concept using PyMDP core utilities."""
    
    # 1. Create matrices using MatrixBuilder
    A = MatrixBuilder.create_observation_model(3, 3, "identity")
    B = MatrixBuilder.create_transition_model(3, 2, "deterministic")
    C = MatrixBuilder.create_preferences(3, "linear")
    D = MatrixBuilder.create_prior(3, "uniform")
    
    # 2. Validate matrices
    validation = validate_matrices(A, B, C, D)
    assert all(result['valid'] for result in validation.values())
    
    # 3. Create agent using PyMDP core
    agent = create_agent(A, B, C, D)
    
    # 4. Run analysis
    results = []
    for obs in [0, 1, 2]:
        qs, q_pi, action = PyMDPCore.run_agent_step(agent, obs)
        results.append({'observation': obs, 'beliefs': qs, 'action': action})
    
    return {
        'results': results,
        'validation': validation,
        'agent': agent
    }

def main():
    """Main function."""
    print("🚀 Example using PyMDP core utilities")
    
    try:
        results = demonstrate_concept()
        
        # Save results
        runner.save_results(results, "results.json")
        
        # Create summary
        summary = runner.create_summary()
        
        print("✅ Example completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 2. Running Examples

```bash
# Run individual example
python textbook/examples/01_probability_basics.py

# Run all examples
bash textbook/examples/run_all.sh

# Run with validation
python textbook/tests/run_all_tests.py --validation-only
```

### 3. Chapter Map (Docs → Examples)

- Bayesian Updating and VFE: `active_inference_basics.md` → `examples/02`, `04`, `05`
- Observation/Transition Models (A/B): `pymdp_core_guide.md` → `examples/03`, `03_refactored`, `07`
- Preferences/Control and EFE: `active_inference_basics.md` → `examples/08`, `09`
- POMDP Agents: `pymdp_overview.md` → `examples/10`, `11`, `12`

### 4. Runner and Authenticity

- Use `run_all.sh --verbose` to verify runtime, outputs, and authenticity
- Logs: `textbook/examples/logs/`; Outputs: `textbook/examples/outputs/`

### 3. Validation

```bash
# Validate single example
python -c "from validation import validate_example; print(validate_example('example.py'))"

# Validate all examples
python -c "from validation import validate_all_examples; print(validate_all_examples('textbook/examples'))"

# Run comprehensive test suite
python textbook/tests/run_all_tests.py
```

## Best Practices

### 1. Use Shared Utilities

Always use the comprehensive shared utilities:

```python
# ✅ Good
from pymdp_core import PyMDPCore
from example_utils import ExampleRunner, MatrixBuilder

# ❌ Avoid
from pymdp.agent import Agent
from pymdp.utils import obj_array_zeros
```

### 2. Validate Everything

Validate matrices and operations:

```python
# Validate matrices
validation = validate_matrices(A, B, C, D)
assert all(result['valid'] for result in validation.values())

# Validate agent
agent_validation = validate_agent(agent, [0, 1, 2])
assert agent_validation['overall_valid']
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
runner = ExampleRunner("example_name")
runner.save_results(results, "results.json")
runner.save_visualization(fig, "plot.png")
summary = runner.create_summary()
```

## Testing

### Running Tests

```bash
# Run all tests
python textbook/tests/run_all_tests.py

# Run specific test
python -m pytest textbook/tests/test_pymdp_core.py -v

# Run with coverage
python textbook/tests/run_all_tests.py --coverage
```

### Test Structure

Tests are organized by component:

- `test_pymdp_core.py` - Tests for PyMDP core utilities
- `test_example_utils.py` - Tests for example utilities
- `test_validation.py` - Tests for validation utilities

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src directory is in Python path
2. **Matrix Validation Failures**: Check matrix normalization
3. **Inference Failures**: Verify observation format (integers, not one-hot)
4. **Agent Creation Failures**: Check matrix dimensions and types

### Debug Mode

Enable debug mode for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
agent = PyMDPCore.create_agent(A, B, C, D, verbose=True)
```

### Getting Help

1. Check the validation reports for specific issues
2. Review the core utility documentation
3. Look at the refactored examples for patterns
4. Run the test suite to identify problems

## Contributing

### Adding New Examples

1. Follow the thin orchestrator pattern
2. Use PyMDP core utilities exclusively
3. Include comprehensive validation
4. Add tests for new functionality
5. Update documentation

### Modifying Existing Examples

1. Maintain backward compatibility
2. Use shared utilities where possible
3. Validate all changes
4. Update tests as needed
5. Document changes

## Conclusion

The PyMDP textbook examples provide a comprehensive learning resource for active inference using real PyMDP methods. By following the thin orchestrator pattern and using the shared utilities, examples maintain:

- **Quality**: High-quality, reliable implementations
- **Consistency**: Standardized patterns across examples
- **Maintainability**: Easy to update and modify
- **Authenticity**: Real PyMDP methods exclusively

For more information, see the individual guide documents and example implementations.