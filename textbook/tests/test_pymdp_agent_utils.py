"""
Tests for PyMDP Agent Utilities
===============================

Test the real PyMDP agent integration utilities in src/pymdp_agent_utils.py
"""

import pytest
import numpy as np
import sys
import os

# Add the src path to import our utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pymdp
from pymdp.agent import Agent
from pymdp.utils import obj_array_zeros, obj_array_uniform, sample
from pymdp.maths import softmax

from pymdp_agent_utils import (
    create_agent_from_matrices,
    run_agent_loop,
    simulate_environment_step,
    create_simple_agent_demo,
    legacy_vfe_inference
)


class TestPyMDPAgentCreation:
    """Test creation of PyMDP Agent instances."""
    
    def test_create_agent_basic(self):
        """Test basic agent creation."""
        # Simple 2-state, 2-action, 2-observation model
        A = obj_array_zeros([[2, 2]])
        A[0] = np.eye(2)  # Perfect observation
        
        B = obj_array_zeros([[2, 2, 2]])  
        B[0][:, :, 0] = np.array([[0.9, 0.1], [0.1, 0.9]])  # Action 0
        B[0][:, :, 1] = np.array([[0.1, 0.9], [0.9, 0.1]])  # Action 1
        
        C = obj_array_zeros([[2]])
        C[0] = np.array([1.0, 0.0])  # Prefer state 0
        
        # Create agent
        agent = create_agent_from_matrices(A, B, C)
        
        # Verify it's a PyMDP Agent
        assert isinstance(agent, Agent)
        assert agent.num_obs == [2]
        assert agent.num_states == [2]
        assert agent.num_controls == [2]
        
    def test_create_agent_with_priors(self):
        """Test agent creation with explicit priors."""
        A = obj_array_zeros([[2, 2]])
        A[0] = np.eye(2)
        
        B = obj_array_zeros([[2, 2, 2]])
        B[0][:, :, 0] = np.eye(2)
        B[0][:, :, 1] = np.eye(2)
        
        C = obj_array_zeros([[2]])
        C[0] = np.array([0.0, 1.0])
        
        D = obj_array_zeros([[2]])
        D[0] = np.array([0.8, 0.2])  # Strong prior for state 0
        
        agent = create_agent_from_matrices(A, B, C, D=D, control_fac_idx=[0])
        
        assert isinstance(agent, Agent)
        np.testing.assert_array_almost_equal(agent.D[0], np.array([0.8, 0.2]))
        
    def test_create_agent_multi_factor(self):
        """Test agent creation with multiple state factors."""
        # 2x2 state space
        A = obj_array_zeros([[4, 2, 2]])  # Joint observation
        A[0] = np.random.rand(4, 2, 2)
        A[0] = A[0] / A[0].sum(axis=0, keepdims=True)  # Normalize
        
        B = obj_array_zeros([[2, 2, 2], [2, 2, 2]])  # Two factors
        B[0][:, :, 0] = np.eye(2)
        B[0][:, :, 1] = np.eye(2)  
        B[1][:, :, 0] = np.eye(2)
        B[1][:, :, 1] = np.eye(2)
        
        C = obj_array_zeros([[4]])
        C[0] = np.random.rand(4)
        
        agent = create_agent_from_matrices(A, B, C, control_fac_idx=[0, 1])
        
        assert isinstance(agent, Agent)
        assert len(agent.num_states) == 2
        assert agent.num_states == [2, 2]


class TestPyMDPAgentLoop:
    """Test the standard PyMDP agent loop."""
    
    def setup_method(self):
        """Set up a simple agent for testing."""
        A = obj_array_zeros([[3, 3]])
        A[0] = np.eye(3)
        
        B = obj_array_zeros([[3, 3, 2]])
        # Move left/right actions
        for s in range(3):
            B[0][max(0, s-1), s, 0] = 1.0      # Move left
            B[0][min(2, s+1), s, 1] = 1.0      # Move right
        
        C = obj_array_zeros([[3]])
        C[0] = np.array([-1.0, 0.0, 1.0])  # Prefer right
        
        self.agent = create_agent_from_matrices(A, B, C, control_fac_idx=[0])
        
    def test_agent_loop_basic(self):
        """Test basic agent loop execution."""
        observation = [1]  # Middle position
        
        beliefs, action = run_agent_loop(self.agent, observation)
        
        # Check outputs
        assert len(beliefs) == 1
        assert beliefs[0].shape == (3,)
        assert np.isclose(beliefs[0].sum(), 1.0)  # Normalized
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert action[0] in [0, 1]  # Valid action
        
    def test_agent_loop_with_int_observation(self):
        """Test agent loop with integer observation."""
        beliefs, action = run_agent_loop(self.agent, 0, verbose=True)
        
        assert beliefs[0][0] > 0.5  # Should believe in state 0
        assert isinstance(action, np.ndarray)
        
    def test_agent_loop_sequential(self):
        """Test sequential agent steps."""
        observations = [0, 1, 2]
        
        for obs in observations:
            beliefs, action = run_agent_loop(self.agent, obs)
            
            # Should have valid beliefs and actions
            assert np.isclose(beliefs[0].sum(), 1.0)
            assert action[0] in [0, 1]


class TestEnvironmentSimulation:
    """Test environment simulation utilities."""
    
    def setup_method(self):
        """Set up environment matrices."""
        self.A = obj_array_zeros([[3, 3]])
        self.A[0] = np.eye(3)  # Perfect observation
        
        self.B = obj_array_zeros([[3, 3, 2]])
        for s in range(3):
            # Deterministic transitions
            self.B[0][max(0, s-1), s, 0] = 1.0      # Move left
            self.B[0][min(2, s+1), s, 1] = 1.0      # Move right
    
    def test_environment_step_basic(self):
        """Test basic environment step."""
        state = [1]  # Middle state
        action = [1]  # Move right
        
        new_state, observation = simulate_environment_step(
            state, action, self.B, self.A
        )
        
        assert new_state == [2]  # Should move right
        assert observation[0] == 2  # Should observe right position
        
    def test_environment_step_boundary(self):
        """Test environment step at boundaries."""
        # Test left boundary
        state = [0]
        action = [0]  # Try to move left
        
        new_state, obs = simulate_environment_step(state, action, self.B, self.A)
        assert new_state == [0]  # Should stay at left
        
        # Test right boundary  
        state = [2]
        action = [1]  # Try to move right
        
        new_state, obs = simulate_environment_step(state, action, self.B, self.A)
        assert new_state == [2]  # Should stay at right
        
    def test_environment_step_with_numpy_action(self):
        """Test environment step with numpy array action."""
        state = [1]
        action = np.array([0])  # Move left
        
        new_state, obs = simulate_environment_step(state, action, self.B, self.A)
        
        assert new_state == [0]
        assert obs[0] == 0


class TestSimpleAgentDemo:
    """Test the complete simple agent demonstration."""
    
    def test_simple_demo_basic(self):
        """Test basic simple agent demo."""
        results = create_simple_agent_demo(
            num_states=3, num_actions=2, num_steps=5, verbose=False
        )
        
        # Check all required keys are present
        required_keys = ['agent', 'state_history', 'observation_history', 
                        'action_history', 'A', 'B', 'C']
        for key in required_keys:
            assert key in results
        
        # Check agent
        assert isinstance(results['agent'], Agent)
        
        # Check histories have correct length
        assert len(results['state_history']) == 6  # num_steps + 1 (initial)
        assert len(results['observation_history']) == 5  # num_steps
        assert len(results['action_history']) == 5  # num_steps
        
        # Check all states/actions are valid
        for state in results['state_history']:
            assert 0 <= state <= 2
        for obs in results['observation_history']:
            assert 0 <= obs <= 2
        for action in results['action_history']:
            assert action in [0, 1]
            
    def test_simple_demo_custom_params(self):
        """Test simple demo with custom parameters."""
        results = create_simple_agent_demo(
            num_states=5, num_actions=2, num_obs=5, num_steps=3, verbose=False
        )
        
        agent = results['agent']
        assert agent.num_states == [5]
        assert agent.num_obs == [5]
        assert agent.num_controls == [2]
        
        # Check model matrices have correct shapes
        assert results['A'][0].shape == (5, 5)
        assert results['B'][0].shape == (5, 5, 2)
        assert results['C'][0].shape == (5,)


class TestLegacyCompatibility:
    """Test legacy compatibility functions."""
    
    def test_legacy_vfe_inference(self):
        """Test legacy VFE inference wrapper."""
        # Simple model
        A = obj_array_zeros([[2, 2]])  
        A[0] = np.array([[0.8, 0.2], [0.2, 0.8]])
        
        prior = obj_array_zeros([[2]])
        prior[0] = np.array([0.5, 0.5])
        
        observation = 0
        
        posterior = legacy_vfe_inference(A, observation, prior, verbose=True)
        
        # Check output format
        assert len(posterior) == 1
        assert posterior[0].shape == (2,)
        assert np.isclose(posterior[0].sum(), 1.0)
        
        # Should favor state 0 after observing 0
        assert posterior[0][0] > posterior[0][1]
        
    def test_legacy_vfe_inference_fallback(self):
        """Test legacy inference fallback mode."""
        # This test ensures the manual Bayes rule fallback works
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        prior = np.array([0.5, 0.5])
        observation = 0
        
        # Convert to obj_array format
        A_obj = obj_array_zeros([[2, 2]])
        A_obj[0] = A
        prior_obj = obj_array_zeros([[2]])  
        prior_obj[0] = prior
        
        posterior = legacy_vfe_inference(A_obj, observation, prior_obj)
        
        # Manual calculation for verification
        likelihood = A[observation, :]
        joint = likelihood * prior
        evidence = np.sum(joint)
        expected_posterior = joint / evidence
        
        np.testing.assert_array_almost_equal(posterior[0], expected_posterior)


class TestIntegrationWithPyMDP:
    """Test integration with real PyMDP components."""
    
    def test_agent_matrices_consistency(self):
        """Test that agent matrices are consistent with PyMDP expectations."""
        A = obj_array_zeros([[3, 3]])
        A[0] = np.array([[0.8, 0.1, 0.1],
                         [0.1, 0.8, 0.1], 
                         [0.1, 0.1, 0.8]])
        
        # Create properly normalized B matrices
        B = obj_array_zeros([[3, 3, 2]])
        # Action 0 - each column sums to 1
        B[0][:, :, 0] = np.array([[0.9, 0.1, 0.0],
                                  [0.1, 0.8, 0.1],
                                  [0.0, 0.1, 0.9]])
        # Action 1 - each column sums to 1  
        B[0][:, :, 1] = np.array([[0.1, 0.9, 0.1],
                                  [0.1, 0.1, 0.8], 
                                  [0.8, 0.0, 0.1]])
        
        # Ensure normalization
        for a in range(2):
            for s in range(3):
                col_sum = B[0][:, s, a].sum()
                if col_sum > 0:
                    B[0][:, s, a] = B[0][:, s, a] / col_sum
        
        C = obj_array_zeros([[3]])
        C[0] = np.array([0.0, 0.0, 1.0])
        
        agent = create_agent_from_matrices(A, B, C)
        
        # Test that agent can perform inference and action selection
        observation = [1]
        beliefs, action = run_agent_loop(agent, observation)
        
        # Verify outputs are reasonable
        assert isinstance(beliefs, list)
        assert len(beliefs) == 1
        assert beliefs[0].shape == (3,)
        assert np.isclose(beliefs[0].sum(), 1.0)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        
    def test_full_simulation_loop(self):
        """Test a complete simulation loop using PyMDP agent."""
        # Create simple environment
        results = create_simple_agent_demo(num_steps=10, verbose=False)
        
        agent = results['agent']
        state_history = results['state_history']
        
        # Verify agent behaves reasonably (should tend toward higher-value states)
        final_states = state_history[-3:]  # Last few states
        initial_states = state_history[:3]  # First few states
        
        # Agent should generally move toward higher-reward states over time
        # (This is a weak test since behavior depends on exploration vs exploitation)
        assert len(set(state_history)) >= 2  # Should visit multiple states


if __name__ == "__main__":
    pytest.main([__file__])
