"""
Tests for Validation Utilities
==============================

Comprehensive tests for the validation utilities to ensure they correctly
validate PyMDP examples and operations.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validation import (
    PyMDPValidator, validate_example, validate_matrices, validate_agent,
    validate_all_examples, create_validation_report
)
from pymdp_core import PyMDPCore
from pymdp import utils
from pymdp.utils import obj_array_zeros, obj_array_uniform


class TestPyMDPValidator:
    """Test PyMDPValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PyMDPValidator()
        
        # Create test matrices
        self.A = obj_array_zeros([[3, 3]])
        self.A[0] = np.eye(3)
        
        self.B = obj_array_zeros([[3, 3, 2]])
        for a in range(2):
            for s in range(3):
                next_s = (s + a) % 3
                self.B[0][next_s, s, a] = 1.0
        
        self.C = obj_array_zeros([[3]])
        self.C[0] = np.array([-1, 0, 1])
        
        self.D = obj_array_zeros([[3]])
        self.D[0] = np.ones(3) / 3
    
    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator is not None
        assert hasattr(self.validator, 'pymdp_methods')
        assert hasattr(self.validator, 'pymdp_classes')
        assert hasattr(self.validator, 'pymdp_functions')
        
        # Check that PyMDP methods are registered
        assert 'Agent' in self.validator.pymdp_methods
        assert 'inference' in self.validator.pymdp_methods
        assert 'control' in self.validator.pymdp_methods
        assert 'learning' in self.validator.pymdp_methods
        assert 'utils' in self.validator.pymdp_methods
        assert 'maths' in self.validator.pymdp_methods
    
    def test_validate_matrices(self):
        """Test matrix validation."""
        results = self.validator.validate_matrices(self.A, self.B, self.C, self.D)
        
        assert 'A' in results
        assert 'B' in results
        assert 'C' in results
        assert 'D' in results
        
        # Check A matrix validation
        assert results['A']['valid'] == True
        assert results['A']['is_normalized'] == True
        assert results['A']['num_modalities'] == 1
        
        # Check B matrix validation
        assert results['B']['valid'] == True
        assert results['B']['is_normalized'] == True
        assert results['B']['num_factors'] == 1
        
        # Check C matrix validation
        assert results['C']['valid'] == True
        assert results['C']['num_modalities'] == 1
        
        # Check D matrix validation
        assert results['D']['valid'] == True
        assert results['D']['is_normalized'] == True
        assert results['D']['num_factors'] == 1
    
    def test_validate_matrices_invalid(self):
        """Test validation of invalid matrices."""
        # Create non-normalized A matrix
        A_invalid = obj_array_zeros([[3, 3]])
        A_invalid[0] = np.ones((3, 3))  # Not normalized
        
        results = self.validator.validate_matrices(A_invalid)
        
        assert 'A' in results
        assert results['A']['valid'] == False
        assert results['A']['is_normalized'] == False
        assert 'issues' in results['A']
        assert len(results['A']['issues']) > 0
    
    def test_validate_agent_operations(self):
        """Test agent operation validation."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        test_observations = [0, 1, 2]
        
        results = self.validator.validate_agent_operations(agent, test_observations)
        
        assert 'inference_valid' in results
        assert 'policy_inference_valid' in results
        assert 'action_sampling_valid' in results
        assert 'issues' in results
        assert 'test_results' in results
        assert 'overall_valid' in results
        
        assert len(results['test_results']) == 3
        
        for test_result in results['test_results']:
            assert 'observation' in test_result
            assert 'state_inference_success' in test_result
            assert 'policy_inference_success' in test_result
            assert 'action_sampling_success' in test_result
    
    def test_validate_example_file(self):
        """Test example file validation."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import pymdp
from pymdp.agent import Agent
from pymdp.utils import obj_array_zeros

# Create matrices
A = obj_array_zeros([[3, 3]])
A[0] = np.eye(3)

# Create agent
agent = Agent(A=A, B=B, C=C, D=D)

# Use PyMDP methods
qs = agent.infer_states([0])
q_pi, G = agent.infer_policies()
action = agent.sample_action()
""")
            temp_file = f.name
        
        try:
            results = self.validator.validate_example_file(temp_file)
            
            assert 'valid' in results
            assert 'pymdp_methods_used' in results
            assert 'non_pymdp_methods' in results
            assert 'imports' in results
            assert 'recommendations' in results
            assert 'coverage_score' in results
            
            # Check that PyMDP methods were detected
            assert len(results['pymdp_methods_used']) > 0
            assert 'Agent' in str(results['pymdp_methods_used'])
            assert 'infer_states' in str(results['pymdp_methods_used'])
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_example_file_invalid(self):
        """Test validation of file with non-PyMDP methods."""
        # Create a temporary test file with non-PyMDP methods
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import numpy as np

# Use non-PyMDP methods
def manual_inference(A, obs, prior):
    likelihood = A[obs, :]
    posterior = likelihood * prior
    return posterior / np.sum(posterior)

def manual_vfe_calculation(posterior, prior):
    return np.sum(posterior * np.log(posterior / prior))

# Non-PyMDP usage
result = manual_inference(A, 0, prior)
vfe = manual_vfe_calculation(result, prior)
""")
            temp_file = f.name
        
        try:
            results = self.validator.validate_example_file(temp_file)
            
            assert 'valid' in results
            assert results['valid'] == False  # Should be invalid
            assert len(results['non_pymdp_methods']) > 0
            assert len(results['recommendations']) > 0
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_example_file_syntax_error(self):
        """Test validation of file with syntax errors."""
        # Create a temporary test file with syntax errors
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import pymdp

# Syntax error: missing colon
def invalid_function()
    return "error"

# More syntax errors
if True
    print("error")
""")
            temp_file = f.name
        
        try:
            results = self.validator.validate_example_file(temp_file)
            
            assert 'valid' in results
            assert results['valid'] == False
            assert 'error' in results
            assert 'Syntax error' in results['error']
            
        finally:
            os.unlink(temp_file)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.A = obj_array_zeros([[3, 3]])
        self.A[0] = np.eye(3)
        
        self.B = obj_array_zeros([[3, 3, 2]])
        for a in range(2):
            for s in range(3):
                next_s = (s + a) % 3
                self.B[0][next_s, s, a] = 1.0
        
        self.C = obj_array_zeros([[3]])
        self.C[0] = np.array([-1, 0, 1])
        
        self.D = obj_array_zeros([[3]])
        self.D[0] = np.ones(3) / 3
    
    def test_validate_example(self):
        """Test validate_example convenience function."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import pymdp
from pymdp.agent import Agent

agent = Agent(A=A, B=B, C=C, D=D)
qs = agent.infer_states([0])
""")
            temp_file = f.name
        
        try:
            results = validate_example(temp_file)
            
            assert 'valid' in results
            assert 'pymdp_methods_used' in results
            assert 'coverage_score' in results
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_matrices(self):
        """Test validate_matrices convenience function."""
        results = validate_matrices(self.A, self.B, self.C, self.D)
        
        assert 'A' in results
        assert 'B' in results
        assert 'C' in results
        assert 'D' in results
        
        # All should be valid
        assert results['A']['valid'] == True
        assert results['B']['valid'] == True
        assert results['C']['valid'] == True
        assert results['D']['valid'] == True
    
    def test_validate_agent(self):
        """Test validate_agent convenience function."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        test_observations = [0, 1]
        
        results = validate_agent(agent, test_observations)
        
        assert 'inference_valid' in results
        assert 'policy_inference_valid' in results
        assert 'action_sampling_valid' in results
        assert 'overall_valid' in results
        assert len(results['test_results']) == 2


class TestValidationSuite:
    """Test validation suite functions."""
    
    def test_validate_all_examples_nonexistent_dir(self):
        """Test validation with non-existent directory."""
        results = validate_all_examples("nonexistent_directory")
        
        assert 'error' in results
        assert 'not found' in results['error']
    
    def test_validate_all_examples_empty_dir(self):
        """Test validation with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = validate_all_examples(temp_dir)
            
            assert 'summary' in results
            assert 'file_results' in results
            assert results['summary']['total_files'] == 0
            assert results['summary']['valid_files'] == 0
    
    def test_validate_all_examples_with_files(self):
        """Test validation with actual Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test Python files
            test_files = [
                "valid_example.py",
                "invalid_example.py",
                "syntax_error.py"
            ]
            
            # Valid example
            with open(os.path.join(temp_dir, "valid_example.py"), 'w') as f:
                f.write("""
import pymdp
from pymdp.agent import Agent
from pymdp.utils import obj_array_zeros

A = obj_array_zeros([[3, 3]])
A[0] = np.eye(3)

agent = Agent(A=A, B=B, C=C, D=D)
qs = agent.infer_states([0])
""")
            
            # Invalid example
            with open(os.path.join(temp_dir, "invalid_example.py"), 'w') as f:
                f.write("""
import numpy as np

def manual_inference(A, obs, prior):
    return A[obs, :] * prior

result = manual_inference(A, 0, prior)
""")
            
            # Syntax error example
            with open(os.path.join(temp_dir, "syntax_error.py"), 'w') as f:
                f.write("""
import pymdp

def invalid_function()
    return "error"
""")
            
            results = validate_all_examples(temp_dir)
            
            assert 'summary' in results
            assert 'file_results' in results
            
            summary = results['summary']
            assert summary['total_files'] == 3
            assert summary['valid_files'] >= 1  # At least one should be valid
            assert summary['invalid_files'] >= 1  # At least one should be invalid
            
            # Check individual file results
            file_results = results['file_results']
            assert 'valid_example.py' in file_results
            assert 'invalid_example.py' in file_results
            assert 'syntax_error.py' in file_results
    
    def test_create_validation_report(self):
        """Test creating validation report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test validation results
            test_results = {
                'summary': {
                    'total_files': 2,
                    'valid_files': 1,
                    'invalid_files': 1,
                    'average_coverage': 0.75,
                    'validation_rate': 0.5
                },
                'file_results': {
                    'test1.py': {
                        'valid': True,
                        'pymdp_methods_used': ['Agent.infer_states'],
                        'coverage_score': 1.0
                    },
                    'test2.py': {
                        'valid': False,
                        'non_pymdp_methods': ['manual_inference'],
                        'coverage_score': 0.5
                    }
                }
            }
            
            report_path = os.path.join(temp_dir, "validation_report.json")
            create_validation_report(test_results, report_path)
            
            # Check report was created
            assert os.path.exists(report_path)
            
            # Check report content
            import json
            with open(report_path, 'r') as f:
                loaded_report = json.load(f)
            
            assert loaded_report['summary']['total_files'] == 2
            assert loaded_report['summary']['valid_files'] == 1
            assert 'test1.py' in loaded_report['file_results']
            assert 'test2.py' in loaded_report['file_results']


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_validate_matrices_none(self):
        """Test validation with None matrices."""
        validator = PyMDPValidator()
        
        # Should handle None gracefully
        results = validator.validate_matrices(None, None, None, None)
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_validate_matrices_partial(self):
        """Test validation with partial matrices."""
        validator = PyMDPValidator()
        
        A = obj_array_zeros([[3, 3]])
        A[0] = np.eye(3)
        
        # Only validate A matrix
        results = validator.validate_matrices(A)
        
        assert 'A' in results
        assert 'B' not in results
        assert 'C' not in results
        assert 'D' not in results
    
    def test_validate_agent_invalid_agent(self):
        """Test validation with invalid agent."""
        validator = PyMDPValidator()
        
        # Create invalid agent (None)
        results = validator.validate_agent_operations(None, [0, 1])
        
        # Should handle gracefully
        assert 'issues' in results
        assert len(results['issues']) > 0
        assert results['overall_valid'] == False


if __name__ == "__main__":
    pytest.main([__file__])
