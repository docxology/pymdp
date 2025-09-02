# PyMDP Validation Guide

## Overview

The PyMDP Validation utilities provide comprehensive validation of PyMDP examples and operations, ensuring that all code uses real PyMDP methods exclusively. This guide covers validation tools, usage patterns, and best practices.

## Validation Components

### 1. PyMDPValidator Class

The `PyMDPValidator` class provides comprehensive validation capabilities:

```python
from validation import PyMDPValidator

validator = PyMDPValidator()

# Validate example file
results = validator.validate_example_file("example.py")

# Validate matrices
results = validator.validate_matrices(A, B, C, D)

# Validate agent operations
results = validator.validate_agent_operations(agent, test_observations)
```

### 2. Convenience Functions

For quick validation tasks:

```python
from validation import validate_example, validate_matrices, validate_agent

# Validate single example
results = validate_example("example.py")

# Validate matrices
results = validate_matrices(A, B, C, D)

# Validate agent
results = validate_agent(agent, [0, 1, 2])
```

### 3. Validation Suite

For comprehensive validation of all examples:

```python
from validation import validate_all_examples, create_validation_report

# Validate all examples
results = validate_all_examples("textbook/examples")

# Create validation report
create_validation_report(results, "validation_report.json")
```

## Validation Types

### 1. Example File Validation

Validates that example files use real PyMDP methods:

```python
results = validator.validate_example_file("example.py")

# Results structure
{
    'valid': True/False,
    'pymdp_methods_used': ['Agent.infer_states', 'PyMDPCore.create_agent'],
    'non_pymdp_methods': ['manual_inference', 'custom_vfe'],
    'imports': ['pymdp', 'pymdp.agent'],
    'recommendations': ['Use PyMDPCore.infer_states instead of manual_inference'],
    'coverage_score': 0.85
}
```

### 2. Matrix Validation (A/B/C/D) Criteria

Validates PyMDP matrices using real PyMDP utilities:

```python
results = validator.validate_matrices(A, B, C, D)

# Results structure
{
    'A': {
        'valid': True/False,
        'is_normalized': True/False,
        'num_modalities': 1,
        'shapes': [(3, 3)],
        'issues': []
    },
    'B': {
        'valid': True/False,
        'is_normalized': True/False,
        'num_factors': 1,
        'shapes': [(3, 3, 2)],
        'issues': []
    }
}
```

### 3. Agent Operation Validation Criteria

Validates agent operations using real PyMDP methods:

```python
results = validator.validate_agent_operations(agent, [0, 1, 2])

# Results structure
{
    'inference_valid': True/False,
    'policy_inference_valid': True/False,
    'action_sampling_valid': True/False,
    'issues': [],
    'test_results': [
        {
            'observation': 0,
            'state_inference_success': True,
            'policy_inference_success': True,
            'action_sampling_success': True
        }
    ],
    'overall_valid': True/False
}
```

## Usage Patterns

### 1. Single Example Validation

```python
from validation import validate_example

# Validate a single example
results = validate_example("textbook/examples/01_probability_basics.py")

if results['valid']:
    print(f"✅ Example is valid (coverage: {results['coverage_score']:.2%})")
else:
    print("❌ Example has issues:")
    for issue in results['recommendations']:
        print(f"  - {issue}")
```

### 2. Matrix Validation

```python
from validation import validate_matrices
from pymdp.utils import obj_array_zeros

# Create test matrices
A = obj_array_zeros([[3, 3]])
A[0] = np.eye(3)

B = obj_array_zeros([[3, 3, 2]])
# ... populate B

# Validate matrices
results = validate_matrices(A, B)

# Check results
for matrix_name, result in results.items():
    status = "✅ VALID" if result['valid'] else "❌ INVALID"
    print(f"{matrix_name} matrix: {status}")
    
    if not result['valid']:
        for issue in result.get('issues', []):
            print(f"  Issue: {issue}")
```

### 3. Agent Validation (Operations)

```python
from validation import validate_agent
from pymdp_core import create_agent

# Create agent
agent = create_agent(A, B, C, D)

# Validate agent operations
results = validate_agent(agent, [0, 1, 2])

if results['overall_valid']:
    print("✅ Agent operations are valid")
else:
    print("❌ Agent has issues:")
    for issue in results['issues']:
        print(f"  - {issue}")
```

### 4. Comprehensive Validation

```python
from validation import validate_all_examples, create_validation_report

# Validate all examples
results = validate_all_examples("textbook/examples")

# Create report
create_validation_report(results, "validation_report.json")

# Check summary
summary = results['summary']
print(f"Total files: {summary['total_files']}")
print(f"Valid files: {summary['valid_files']}")
print(f"Validation rate: {summary['validation_rate']:.2%}")
print(f"Average coverage: {summary['average_coverage']:.2%}")
```

## Validation Criteria

### 1. PyMDP Method Usage

Examples must use real PyMDP methods:

- Valid methods:
- `pymdp.agent.Agent`
- `agent.infer_states()`
- `agent.infer_policies()`
- `agent.sample_action()`
- `pymdp.utils.obj_array_zeros`
- `pymdp.maths.softmax`
- `PyMDPCore.create_agent`
- `PyMDPCore.infer_states`

- Invalid methods:

- `manual_inference()`
- `custom_vfe_calculation()`
- `numpy.random.choice` (for action sampling)
- Direct matrix operations without PyMDP utilities

### 2. Matrix Validation (A/B/C/D)

Matrices must meet PyMDP requirements:

**✅ Valid Matrices:**

- Normalized columns (sum to 1)
- Correct dimensions
- Proper object array structure
- Valid probability distributions

**❌ Invalid Matrices:**

- Non-normalized columns
- Incorrect dimensions
- Invalid probability values
- Wrong data types

### 3. Agent Operation Validation

Agent operations must work correctly:

**✅ Valid Operations:**

- Successful state inference
- Successful policy inference
- Successful action sampling
- Proper error handling

**❌ Invalid Operations:**

- Inference failures
- Dimension mismatches
- Type errors
- Unhandled exceptions

## Best Practices

### 1. Regular Validation

```python
# Validate during development
from validation import validate_example

def test_example():
    # Run example
    results = run_example()
    
    # Validate
    validation = validate_example("my_example.py")
    assert validation['valid'], f"Validation failed: {validation['recommendations']}"
    
    return results
```

### 2. Continuous Integration

```python
# In CI/CD pipeline
from validation import validate_all_examples

def ci_validation():
    results = validate_all_examples("textbook/examples")
    
    # Fail if validation rate is too low
    validation_rate = results['summary']['validation_rate']
    assert validation_rate > 0.9, f"Validation rate too low: {validation_rate:.2%}"
    
    return results
```

### 3. Pre-commit Validation

```python
# Pre-commit hook
from validation import validate_example
import sys

def pre_commit_validation():
    modified_files = get_modified_python_files()
    
    for file_path in modified_files:
        if 'examples' in file_path:
            results = validate_example(file_path)
            if not results['valid']:
                print(f"❌ {file_path} failed validation")
                for rec in results['recommendations']:
                    print(f"  - {rec}")
                sys.exit(1)
    
    print("✅ All examples passed validation")
```

## Troubleshooting

### Common Validation Issues

1. **Non-PyMDP Methods**: Replace with PyMDP core utilities
2. **Matrix Normalization**: Ensure columns sum to 1
3. **Dimension Mismatches**: Check matrix shapes
4. **Import Issues**: Use proper PyMDP imports

### Debug Mode

Enable debug mode for detailed validation information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation with debug info
results = validate_example("example.py", debug=True)
```

### Validation Reports

Generate detailed validation reports:

```python
from validation import create_validation_report

# Create comprehensive report
results = validate_all_examples("textbook/examples")
create_validation_report(results, "detailed_validation_report.json")

# Report includes:
# - File-by-file validation results
# - Method usage analysis
# - Recommendations for improvements
# - Coverage statistics
```

## Integration with Testing

### Test-Driven Validation

```python
import pytest
from validation import validate_example, validate_matrices

def test_example_validation():
    """Test that example passes validation."""
    results = validate_example("example.py")
    assert results['valid'], f"Validation failed: {results['recommendations']}"

def test_matrix_validation():
    """Test that matrices are valid."""
    results = validate_matrices(A, B, C, D)
    for matrix_name, result in results.items():
        assert result['valid'], f"{matrix_name} matrix validation failed"
```

### Automated Validation

```python
# Automated validation script
def run_validation_suite():
    """Run comprehensive validation suite."""
    
    # Validate all examples
    results = validate_all_examples("textbook/examples")
    
    # Check results
    if results['summary']['validation_rate'] < 0.9:
        print("❌ Validation rate too low")
        return False
    
    # Generate report
    create_validation_report(results, "validation_report.json")
    
    print("✅ Validation suite passed")
    return True
```

## Runner Integration and Conclusion

The PyMDP Validation utilities ensure that all examples use real PyMDP methods exclusively. Use the `run_all.sh` runner to perform end-to-end checks:

### Runner (`run_all.sh`) Checks

- Runtime success (nonzero exit codes flagged)
- Output verification (required files exist, non-empty; JSON parseable)
- Authenticity checks (logs contain only real-method messages; no fallback strings)

Run:

```bash
bash textbook/examples/run_all.sh --verbose
```

Artifacts:

- Logs: `textbook/examples/logs/*.log`
- Outputs: `textbook/examples/outputs/*`

Following these practices ensures:

- **Quality**: Examples maintain high quality standards
- **Consistency**: All examples follow the same patterns
- **Reliability**: Examples work correctly with PyMDP
- **Maintainability**: Easy to identify and fix issues

For topic mapping and links to worked examples, see `active_inference_basics.md` and `pymdp_core_guide.md`.
