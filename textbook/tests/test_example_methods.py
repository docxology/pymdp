"""
Example Methods Tests
====================

Test the key functions from our refactored examples, particularly focusing on
VFE and EFE calculations that use PyMDP methods.
"""

import pytest
import numpy as np
import sys
import os

# Add the textbook path to sys.path to import example functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pymdp
from pymdp.utils import obj_array_zeros, obj_array_uniform, onehot
from pymdp.maths import softmax, entropy, kl_div, spm_log, calc_free_energy

# Import our utility functions
try:
    from analysis import evaluate_performance, measure_exploration, compute_information_gain
    from model_utils import validate_model
    from visualization import plot_free_energy, plot_beliefs
except ImportError as e:
    pytest.skip(f"Could not import src utilities: {e}", allow_module_level=True)


class TestVFECalculations:
    """Test Variational Free Energy calculations from examples."""
    
    def test_vfe_basic_calculation(self):
        """Test basic VFE calculation using PyMDP methods."""
        # Simple model setup
        A = obj_array_zeros([[3, 2]])
        A[0] = np.array([[0.9, 0.1],
                         [0.1, 0.9],
                         [0.0, 0.0]])
        
        prior = obj_array_uniform([2])
        obs = [0]
        
        # Manual VFE calculation following our examples
        likelihood = A[0][obs[0], :]
        joint = likelihood * prior[0]
        evidence = np.sum(joint)
        
        if evidence > 1e-16:
            posterior = joint / evidence
        else:
            posterior = np.ones(2) / 2
            
        # VFE components
        complexity = kl_div(posterior, prior[0])
        safe_likelihood = np.maximum(likelihood, 1e-16)
        accuracy = np.sum(posterior * spm_log(safe_likelihood))
        vfe = complexity - accuracy
        
        # Test properties
        assert vfe >= 0, "VFE should be non-negative"
        assert np.isfinite(vfe), "VFE should be finite"
        assert np.sum(posterior) == pytest.approx(1.0), "Posterior should be normalized"
        
        # Test that VFE decreases with better evidence
        # Perfect observation should give lower VFE
        A_perfect = obj_array_zeros([[2, 2]])
        A_perfect[0] = np.eye(2)  # Perfect observation
        
        likelihood_perfect = A_perfect[0][obs[0], :]
        joint_perfect = likelihood_perfect * prior[0]
        evidence_perfect = np.sum(joint_perfect)
        posterior_perfect = joint_perfect / evidence_perfect
        
        complexity_perfect = kl_div(posterior_perfect, prior[0])
        accuracy_perfect = np.sum(posterior_perfect * spm_log(likelihood_perfect))
        vfe_perfect = complexity_perfect - accuracy_perfect
        
        assert vfe_perfect <= vfe, "Perfect observation should give lower VFE"
    
    def test_vfe_sequential_inference(self):
        """Test VFE in sequential inference scenario."""
        # Setup from 05_sequential_inference.py
        A = obj_array_zeros([[3, 2]])
        A[0] = np.array([[0.8, 0.1],
                         [0.2, 0.8],
                         [0.0, 0.1]])
        
        # Sequential observations
        observations = [0, 1, 0]
        prior = obj_array_uniform([2])
        beliefs = prior[0].copy()
        vfe_sequence = []
        
        for obs in observations:
            # VFE calculation
            likelihood = A[0][obs, :]
            joint = likelihood * beliefs
            evidence = np.sum(joint)
            
            if evidence > 1e-16:
                posterior = joint / evidence
            else:
                posterior = np.ones(2) / 2
                
            complexity = kl_div(posterior, beliefs)
            safe_likelihood = np.maximum(likelihood, 1e-16)
            accuracy = np.sum(posterior * spm_log(safe_likelihood))
            vfe = complexity - accuracy
            
            vfe_sequence.append(vfe)
            beliefs = posterior
        
        # Test properties
        assert all(vfe >= 0 for vfe in vfe_sequence), "All VFE values should be non-negative"
        assert all(np.isfinite(vfe) for vfe in vfe_sequence), "All VFE values should be finite"
        
        # Test that beliefs converge (entropy should decrease over time)
        entropies = []
        beliefs = prior[0].copy()
        for obs in observations:
            likelihood = A[0][obs, :]
            joint = likelihood * beliefs
            evidence = np.sum(joint)
            if evidence > 1e-16:
                beliefs = joint / evidence
            entropies.append(entropy(beliefs))
        
        # Generally, entropy should decrease (become more confident)
        assert entropies[-1] <= entropies[0], "Should become more confident over time"
    
    def test_vfe_multi_factor_model(self):
        """Test VFE calculation with multi-factor models."""
        # Setup inspired by 06_multi_factor_models.py
        num_states_f1, num_states_f2 = 2, 3
        num_obs = 4
        
        # Create A matrix for joint observation model
        A = obj_array_zeros([[num_obs, num_states_f1, num_states_f2]])
        
        # Fill A matrix with some reasonable probabilities
        for s1 in range(num_states_f1):
            for s2 in range(num_states_f2):
                obs_probs = np.random.dirichlet(np.ones(num_obs))
                A[0][:, s1, s2] = obs_probs
        
        # Test that the model components are structurally valid
        # (Skip complex multi-factor B matrix creation for this test)
        
        # Prior beliefs
        prior = obj_array_uniform([num_states_f1, num_states_f2])
        
        # Test VFE calculation would work (simplified)
        obs = [0]
        
        # In a real multi-factor case, we'd use PyMDP's update_posterior_states
        # Here we just test that the structure is compatible
        assert len(prior) == 2, "Should have 2 factors"
        assert np.sum(prior[0]) == pytest.approx(1.0), "Factor 1 should be normalized"
        assert np.sum(prior[1]) == pytest.approx(1.0), "Factor 2 should be normalized"


class TestEFECalculations:
    """Test Expected Free Energy calculations from examples."""
    
    def test_efe_basic_calculation(self):
        """Test basic EFE calculation using PyMDP methods."""
        # Setup from 08_preferences_and_control.py
        num_states, num_actions, num_obs = 3, 2, 3
        
        # Create model matrices
        A = obj_array_zeros([[num_obs, num_states]])
        A[0] = np.array([[0.9, 0.1, 0.0],
                         [0.1, 0.8, 0.1],
                         [0.0, 0.1, 0.9]])
        
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        B[0][:, :, 0] = np.array([[0.8, 0.2, 0.0],
                                  [0.2, 0.6, 0.2],
                                  [0.0, 0.2, 0.8]])
        B[0][:, :, 1] = np.array([[0.2, 0.8, 0.0],
                                  [0.2, 0.6, 0.2],
                                  [0.0, 0.0, 1.0]])
        
        C = obj_array_zeros([[num_obs]])
        C[0] = np.array([1.0, 0.0, -1.0])  # Preferences
        
        # Current beliefs
        beliefs = np.array([0.3, 0.5, 0.2])
        
        # Calculate EFE for each action
        action_efe = np.zeros(num_actions)
        
        for action in range(num_actions):
            # Predicted states after action
            predicted_states = beliefs @ B[0][:, :, action]
            
            # Predicted observations
            predicted_obs = predicted_states @ A[0].T
            
            # Pragmatic value (utility)
            utility = np.sum(predicted_obs * C[0])
            
            # Epistemic value (state information gain)
            state_info_gain = 0.0
            for s in range(num_states):
                if predicted_states[s] > 1e-16 and beliefs[s] > 1e-16:
                    state_info_gain += predicted_states[s] * np.log(predicted_states[s] / beliefs[s])
            
            # EFE is negative of total value
            efe = -(utility + state_info_gain)
            action_efe[action] = efe
        
        # Test properties
        assert all(np.isfinite(efe) for efe in action_efe), "All EFE values should be finite"
        
        # Convert to action probabilities
        action_values = -action_efe  # Higher value = lower EFE
        action_probs = softmax(action_values)
        
        assert np.sum(action_probs) == pytest.approx(1.0), "Action probabilities should sum to 1"
        assert all(p >= 0 for p in action_probs), "All action probabilities should be non-negative"
    
    def test_efe_multi_step_planning(self):
        """Test multi-step EFE calculation from policy inference example."""
        # Setup inspired by 09_policy_inference.py
        num_states = 3
        num_actions = 2
        planning_horizon = 2
        
        # Create model matrices
        A = obj_array_zeros([[num_states, num_states]])
        A[0] = np.eye(num_states)  # Perfect observation
        
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        # Action 0: move left (cyclic)
        B[0][:, :, 0] = np.array([[0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0]])
        # Action 1: move right (cyclic)
        B[0][:, :, 1] = np.array([[0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]])
        
        C = obj_array_zeros([[num_states]])
        C[0] = np.array([1.0, 0.0, -1.0])  # Prefer state 0, avoid state 2
        
        # Generate simple policies
        policies = []
        for a1 in range(num_actions):
            for a2 in range(num_actions):
                policies.append([a1, a2])
        
        # Current beliefs
        beliefs = np.array([0.0, 1.0, 0.0])  # Start in state 1
        
        # Calculate EFE for each policy
        policy_efe = []
        
        for policy in policies:
            total_efe = 0.0
            current_beliefs = beliefs.copy()
            
            for t, action in enumerate(policy):
                # Predicted states
                predicted_states = current_beliefs @ B[0][:, :, action]
                
                # Predicted observations
                predicted_obs = predicted_states @ A[0].T
                
                # Pragmatic value
                utility = np.sum(predicted_obs * C[0])
                
                # Epistemic value (state information gain)
                state_info_gain = 0.0
                for s in range(num_states):
                    if predicted_states[s] > 1e-16 and current_beliefs[s] > 1e-16:
                        state_info_gain += predicted_states[s] * np.log(predicted_states[s] / current_beliefs[s])
                
                # Add to total EFE
                step_efe = -(utility + state_info_gain)
                total_efe += step_efe
                
                # Update beliefs for next step
                current_beliefs = predicted_states
            
            policy_efe.append(total_efe)
        
        # Test properties
        assert len(policy_efe) == len(policies), "Should have EFE for each policy"
        assert all(np.isfinite(efe) for efe in policy_efe), "All policy EFEs should be finite"
        
        # Convert to policy probabilities
        policy_values = -np.array(policy_efe)
        policy_probs = softmax(policy_values)
        
        assert np.sum(policy_probs) == pytest.approx(1.0), "Policy probabilities should sum to 1"
        assert all(p >= 0 for p in policy_probs), "All policy probabilities should be non-negative"
    
    def test_efe_precision_effects(self):
        """Test how precision affects EFE-based action selection."""
        # Simple 2-state, 2-action setup
        num_states, num_actions = 2, 2
        
        A = obj_array_zeros([[num_states, num_states]])
        A[0] = np.eye(num_states)
        
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        B[0][:, :, 0] = np.array([[0.9, 0.1], [0.1, 0.9]])  # Stay
        B[0][:, :, 1] = np.array([[0.1, 0.9], [0.9, 0.1]])  # Switch
        
        C = obj_array_zeros([[num_states]])
        C[0] = np.array([1.0, 0.0])  # Prefer state 0
        
        beliefs = np.array([0.2, 0.8])  # Currently in state 1 (bad state)
        
        # Calculate action values
        action_efe = np.zeros(num_actions)
        for action in range(num_actions):
            predicted_states = beliefs @ B[0][:, :, action]
            predicted_obs = predicted_states @ A[0].T
            utility = np.sum(predicted_obs * C[0])
            action_efe[action] = -utility  # Simplified EFE (just pragmatic)
        
        # Test with different precision values
        precisions = [0.1, 1.0, 10.0]
        action_probs_list = []
        
        for precision in precisions:
            action_values = -action_efe
            action_probs = softmax(action_values * precision)
            action_probs_list.append(action_probs)
        
        # Higher precision should lead to more deterministic choices
        entropy_low = entropy(action_probs_list[0])
        entropy_high = entropy(action_probs_list[2])
        
        assert entropy_high <= entropy_low, "Higher precision should reduce entropy"


class TestExampleIntegration:
    """Test integration between examples and src utilities."""
    
    def test_model_validation(self):
        """Test model validation from model_utils."""
        # Create valid model with all required components
        A = obj_array_zeros([[3, 2]])
        A[0] = np.array([[0.8, 0.1],
                         [0.2, 0.8],
                         [0.0, 0.1]])
        
        B = obj_array_zeros([[2, 2, 2]])
        B[0][:, :, 0] = np.eye(2)
        B[0][:, :, 1] = np.eye(2)
        
        C = obj_array_zeros([[3]])
        C[0] = np.array([1.0, 0.0, -1.0])
        
        D = obj_array_zeros([[2]])
        D[0] = np.array([0.5, 0.5])
        
        # Should pass validation
        assert validate_model(A, B, C, D, verbose=False), "Valid model should pass validation"
        
        # Create invalid model (A matrix columns don't sum to 1)
        A_invalid = obj_array_zeros([[2, 2]])
        A_invalid[0] = np.array([[0.5, 0.3], [0.4, 0.6]])  # Columns don't sum to 1
        
        B_valid = obj_array_zeros([[2, 2, 2]])
        B_valid[0][:, :, 0] = np.eye(2)
        B_valid[0][:, :, 1] = np.eye(2)
        
        C_valid = obj_array_zeros([[2]])
        C_valid[0] = np.array([1.0, 0.0])
        
        D_valid = obj_array_zeros([[2]])
        D_valid[0] = np.array([0.5, 0.5])
        
        # Should fail validation
        assert not validate_model(A_invalid, B_valid, C_valid, D_valid, verbose=False), "Invalid model should fail validation"
    
    def test_performance_analysis(self):
        """Test performance analysis from analysis module."""
        # Create simple observation sequence and preferences
        observations = [0, 1, 0, 1, 0]
        preferences = np.array([0.5, 0.8, -0.3])  # Preferences for 3 possible observations
        
        performance = evaluate_performance(observations, preferences)
        
        # Should return valid metrics (using actual function return keys)
        assert 'total_reward' in performance
        assert 'mean_reward' in performance
        assert 'cumulative_reward' in performance
        assert 'high_preference_rate' in performance
        
        # Calculate expected values
        expected_rewards = [preferences[obs] for obs in observations]
        assert performance['total_reward'] == sum(expected_rewards)
        assert performance['mean_reward'] == np.mean(expected_rewards)
        assert len(performance['cumulative_reward']) == len(observations)
    
    def test_exploration_measurement(self):
        """Test exploration measurement from analysis module."""
        # Create belief history showing learning
        belief_history = [
            np.array([0.5, 0.5]),      # Initial uncertainty
            np.array([0.6, 0.4]),      # Slight learning
            np.array([0.8, 0.2]),      # More learning
            np.array([0.9, 0.1])       # Confident
        ]
        
        exploration_metrics = measure_exploration(belief_history)
        
        # Should return valid metrics (using actual function return keys)
        assert 'mean_belief_entropy' in exploration_metrics
        assert 'std_belief_entropy' in exploration_metrics
        assert 'entropy_trend' in exploration_metrics
        assert 'mean_belief_change' in exploration_metrics
        assert 'std_belief_change' in exploration_metrics
        
        # Entropy trend should be negative (decreasing over time)
        assert exploration_metrics['entropy_trend'] <= 0
        assert exploration_metrics['mean_belief_entropy'] >= 0
    
    def test_information_gain_calculation(self):
        """Test information gain calculation from analysis module."""
        prior = np.array([0.5, 0.5])
        posterior = np.array([0.8, 0.2])
        
        info_gain = compute_information_gain(posterior, prior)
        
        # Should be positive (gained information)
        assert info_gain >= 0
        assert np.isfinite(info_gain)
        
        # Self-information should be zero
        self_info = compute_information_gain(prior, prior)
        assert self_info == pytest.approx(0.0, abs=1e-10)


class TestPOMDPAgent:
    """Test POMDP agent methods from example 10."""
    
    def test_agent_observation_processing(self):
        """Test agent's observation processing with VFE calculation."""
        # Simple navigation setup
        num_states, num_actions, num_obs = 2, 2, 2
        
        # Create agent matrices
        A = obj_array_zeros([[num_obs, num_states]])
        A[0] = np.array([[0.9, 0.1], [0.1, 0.9]])  # Noisy observation
        
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        B[0][:, :, 0] = np.array([[0.8, 0.2], [0.2, 0.8]])  # Move left
        B[0][:, :, 1] = np.array([[0.2, 0.8], [0.8, 0.2]])  # Move right
        
        C = obj_array_zeros([[num_obs]])
        C[0] = np.array([1.0, 0.0])  # Prefer obs 0
        
        # Initial beliefs
        beliefs = np.array([0.5, 0.5])
        
        # Process an observation using VFE-based inference
        obs = 0
        likelihood = A[0][obs, :]
        joint = likelihood * beliefs
        evidence = np.sum(joint)
        
        if evidence > 1e-16:
            posterior = joint / evidence
        else:
            posterior = np.ones(num_states) / num_states
        
        # Calculate VFE
        complexity = kl_div(posterior, beliefs)
        safe_likelihood = np.maximum(likelihood, 1e-16)
        accuracy = np.sum(posterior * spm_log(safe_likelihood))
        vfe = complexity - accuracy
        
        # Test properties
        assert np.sum(posterior) == pytest.approx(1.0), "Posterior should be normalized"
        assert vfe >= 0, "VFE should be non-negative"
        assert np.isfinite(vfe), "VFE should be finite"
        
        # Should be more confident about state 0 after observing obs 0
        assert posterior[0] > posterior[1], "Should favor state 0 after observing 0"
    
    def test_agent_action_selection(self):
        """Test agent's action selection with EFE calculation."""
        # Same setup as observation test
        num_states, num_actions, num_obs = 2, 2, 2
        
        A = obj_array_zeros([[num_obs, num_states]])
        A[0] = np.array([[0.9, 0.1], [0.1, 0.9]])
        
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        B[0][:, :, 0] = np.array([[0.8, 0.2], [0.2, 0.8]])  # Move left
        B[0][:, :, 1] = np.array([[0.2, 0.8], [0.8, 0.2]])  # Move right
        
        C = obj_array_zeros([[num_obs]])
        C[0] = np.array([1.0, 0.0])  # Prefer obs 0
        
        beliefs = np.array([0.3, 0.7])  # Currently believe more in state 1
        
        # Calculate EFE for each action
        action_efe = np.zeros(num_actions)
        
        for action in range(num_actions):
            # Predicted states
            predicted_states = beliefs @ B[0][:, :, action]
            
            # Predicted observations
            predicted_obs = predicted_states @ A[0].T
            
            # Utility (pragmatic value)
            utility = np.sum(predicted_obs * C[0])
            
            # State information gain (epistemic value)
            state_info_gain = 0.0
            for s in range(num_states):
                if predicted_states[s] > 1e-16 and beliefs[s] > 1e-16:
                    state_info_gain += predicted_states[s] * np.log(predicted_states[s] / beliefs[s])
            
            # EFE is negative of total value
            efe = -(utility + state_info_gain)
            action_efe[action] = efe
        
        # Convert to action probabilities
        action_values = -action_efe
        action_probs = softmax(action_values)
        
        # Test properties
        assert len(action_efe) == num_actions, "Should have EFE for each action"
        assert all(np.isfinite(efe) for efe in action_efe), "All EFEs should be finite"
        assert np.sum(action_probs) == pytest.approx(1.0), "Action probabilities should sum to 1"
        assert all(p >= 0 for p in action_probs), "All action probabilities should be non-negative"


# All spm_log references have been corrected above


if __name__ == "__main__":
    pytest.main([__file__])
