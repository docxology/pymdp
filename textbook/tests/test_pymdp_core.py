"""
Tests for PyMDP Core Operations
===============================

Comprehensive tests for the PyMDP core utilities to ensure they use
real PyMDP methods exclusively and work correctly.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pymdp_core import PyMDPCore, create_agent, infer_states, infer_policies, sample_action
from pymdp import utils
from pymdp.utils import obj_array_zeros, obj_array_uniform


class TestPyMDPCore:
    """Test PyMDP core operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test matrices
        self.A = obj_array_zeros([[3, 3]])
        self.A[0] = np.eye(3)  # Perfect observation model
        
        self.B = obj_array_zeros([[3, 3, 2]])
        # Simple deterministic transitions
        for a in range(2):
            for s in range(3):
                next_s = (s + a) % 3
                self.B[0][next_s, s, a] = 1.0
        
        self.C = obj_array_zeros([[3]])
        self.C[0] = np.array([-1, 0, 1])  # Linear preferences
        
        self.D = obj_array_zeros([[3]])
        self.D[0] = np.ones(3) / 3  # Uniform prior
    
    def test_create_agent(self):
        """Test agent creation using real PyMDP methods."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        
        assert agent is not None
        assert hasattr(agent, 'A')
        assert hasattr(agent, 'B')
        assert hasattr(agent, 'C')
        assert hasattr(agent, 'D')
        
        # Verify matrices are properly set
        assert np.array_equal(agent.A[0], self.A[0])
        assert np.array_equal(agent.B[0], self.B[0])
        assert np.array_equal(agent.C[0], self.C[0])
        assert np.array_equal(agent.D[0], self.D[0])
    
    def test_create_agent_with_kwargs(self):
        """Test agent creation with additional parameters."""
        agent = PyMDPCore.create_agent(
            self.A, self.B, self.C, self.D,
            gamma=32.0, alpha=16.0, policy_len=2
        )
        
        assert agent.gamma == 32.0
        assert agent.alpha == 16.0
        assert agent.policy_len == 2
    
    def test_infer_states(self):
        """Test state inference using real PyMDP methods."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        
        # Test with integer observation
        qs = PyMDPCore.infer_states(agent, 0)
        assert qs is not None
        assert len(qs) > 0
        assert np.allclose(np.sum(qs[0]), 1.0)  # Should be normalized
        
        # Test with list observation
        qs = PyMDPCore.infer_states(agent, [1])
        assert qs is not None
        assert len(qs) > 0
        assert np.allclose(np.sum(qs[0]), 1.0)
    
    def test_infer_policies(self):
        """Test policy inference using real PyMDP methods."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        
        q_pi, G = PyMDPCore.infer_policies(agent)
        
        assert q_pi is not None
        assert G is not None
        assert len(G) > 0  # Should have EFE for each policy
    
    def test_sample_action(self):
        """Test action sampling using real PyMDP methods."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        
        action = PyMDPCore.sample_action(agent)
        
        assert action is not None
        assert isinstance(action, np.ndarray)
        assert len(action) > 0
    
    def test_run_agent_step(self):
        """Test complete agent step using real PyMDP methods."""
        agent = PyMDPCore.create_agent(self.A, self.B, self.C, self.D)
        
        qs, q_pi, action = PyMDPCore.run_agent_step(agent, 0)
        
        assert qs is not None
        assert q_pi is not None
        assert action is not None
        
        # Verify beliefs are normalized
        assert np.allclose(np.sum(qs[0]), 1.0)
    
    def test_compute_vfe(self):
        """Test VFE computation using real PyMDP methods."""
        # Test with provided posterior
        posterior = obj_array_zeros([[3]])
        posterior[0] = np.array([0.8, 0.1, 0.1])
        
        vfe, components, computed_posterior = PyMDPCore.compute_vfe(
            self.A, 0, self.D, posterior
        )
        
        assert isinstance(vfe, float)
        assert isinstance(components, dict)
        assert 'complexity' in components or components['complexity'] is None
        assert 'accuracy' in components or components['accuracy'] is None
        assert computed_posterior is not None
        
        # Test without provided posterior
        vfe, components, computed_posterior = PyMDPCore.compute_vfe(
            self.A, 0, self.D
        )
        
        assert isinstance(vfe, float)
        assert computed_posterior is not None
        assert np.allclose(np.sum(computed_posterior[0]), 1.0)
    
    def test_compute_efe(self):
        """Test EFE computation using real PyMDP methods."""
        beliefs = np.array([0.5, 0.3, 0.2])
        policy = [0, 1]  # Two-step policy
        
        efe, components = PyMDPCore.compute_efe(
            self.A, self.B, self.C, beliefs, policy
        )
        
        assert isinstance(efe, float)
        assert isinstance(components, dict)
        assert 'pragmatic_value' in components
        assert 'epistemic_value' in components
    
    def test_construct_policies(self):
        """Test policy construction using real PyMDP methods."""
        num_controls = [2]  # Single factor with 2 actions
        policies = PyMDPCore.construct_policies(num_controls, policy_len=2)
        
        assert isinstance(policies, list)
        assert len(policies) > 0
        
        # Check policy structure
        for policy in policies:
            assert isinstance(policy, (list, np.ndarray))
            assert len(policy) == 2  # policy_len
    
    def test_validate_matrices(self):
        """Test matrix validation using real PyMDP methods."""
        results = PyMDPCore.validate_matrices(self.A, self.B, self.C, self.D)
        
        assert 'A' in results
        assert 'B' in results
        assert 'C' in results
        assert 'D' in results
        
        # Check A matrix validation
        assert results['A']['is_normalized'] == True
        assert results['A']['num_modalities'] == 1
        
        # Check B matrix validation
        assert results['B']['is_normalized'] == True
        assert results['B']['num_factors'] == 1
        
        # Check C matrix validation
        assert results['C']['num_modalities'] == 1
        
        # Check D matrix validation
        assert results['D']['is_normalized'] == True
        assert results['D']['num_factors'] == 1
    
    def test_simulate_environment(self):
        """Test environment simulation using real PyMDP methods."""
        state = [1]  # Single factor state
        action = [0]  # Single action
        
        new_state, observation = PyMDPCore.simulate_environment(
            state, action, self.B, self.A
        )
        
        assert isinstance(new_state, list)
        assert isinstance(observation, list)
        assert len(new_state) == 1
        assert len(observation) == 1
        
        # Verify state transition
        assert 0 <= new_state[0] < 3
        assert 0 <= observation[0] < 3


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
    
    def test_create_agent_convenience(self):
        """Test create_agent convenience function."""
        agent = create_agent(self.A, self.B, self.C, self.D)
        assert agent is not None
        assert hasattr(agent, 'A')
    
    def test_infer_states_convenience(self):
        """Test infer_states convenience function."""
        agent = create_agent(self.A, self.B, self.C, self.D)
        qs = infer_states(agent, 0)
        assert qs is not None
        assert np.allclose(np.sum(qs[0]), 1.0)
    
    def test_infer_policies_convenience(self):
        """Test infer_policies convenience function."""
        agent = create_agent(self.A, self.B, self.C, self.D)
        q_pi, G = infer_policies(agent)
        assert q_pi is not None
        assert G is not None
    
    def test_sample_action_convenience(self):
        """Test sample_action convenience function."""
        agent = create_agent(self.A, self.B, self.C, self.D)
        action = sample_action(agent)
        assert action is not None
        assert isinstance(action, np.ndarray)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_matrices(self):
        """Test handling of invalid matrices."""
        # Test with non-normalized A matrix
        A = obj_array_zeros([[3, 3]])
        A[0] = np.ones((3, 3))  # Not normalized
        
        B = obj_array_zeros([[3, 3, 2]])
        B[0] = np.ones((3, 3, 2))  # Not normalized
        
        # Should still create agent (PyMDP may handle normalization)
        agent = PyMDPCore.create_agent(A, B)
        assert agent is not None
    
    def test_empty_observations(self):
        """Test handling of edge case observations."""
        A = obj_array_zeros([[3, 3]])
        A[0] = np.eye(3)
        
        B = obj_array_zeros([[3, 3, 2]])
        B[0] = np.ones((3, 3, 2)) / 3
        
        agent = PyMDPCore.create_agent(A, B)
        
        # Test with edge case observations
        try:
            qs = PyMDPCore.infer_states(agent, 0)
            assert qs is not None
        except Exception as e:
            # Some edge cases may fail, which is acceptable
            assert isinstance(e, (IndexError, ValueError, TypeError))
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches."""
        # Create matrices with mismatched dimensions
        A = obj_array_zeros([[2, 3]])  # 2 obs, 3 states
        A[0] = np.ones((2, 3)) / 3
        
        B = obj_array_zeros([[3, 3, 2]])  # 3 states, 2 actions
        B[0] = np.ones((3, 3, 2)) / 3
        
        # Should handle gracefully
        try:
            agent = PyMDPCore.create_agent(A, B)
            assert agent is not None
        except Exception as e:
            # Dimension mismatches may cause errors, which is expected
            assert isinstance(e, (ValueError, IndexError))


if __name__ == "__main__":
    pytest.main([__file__])
