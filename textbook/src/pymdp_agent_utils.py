"""
PyMDP Agent Utilities
=====================

Utilities for creating and working with real PyMDP Agent instances,
following the patterns from the official PyMDP examples.

This module now uses the comprehensive PyMDP core utilities.
"""

import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pymdp_core import PyMDPCore
from pymdp import utils
from pymdp.utils import obj_array_zeros, obj_array_uniform, sample, onehot
from pymdp.inference import update_posterior_states

try:
    # Prefer real PyMDP free energy function when available
    from pymdp.maths import calc_free_energy, spm_dot, spm_log_single
    spm_log = spm_log_single  # Alias for compatibility
except Exception:  # pragma: no cover - keep compatibility across versions
    calc_free_energy = None
    spm_log = None


def create_agent_from_matrices(A, B, C, D=None, control_fac_idx=None, **kwargs):
    """
    Create a PyMDP Agent instance from matrices using PyMDP core utilities.
    
    Parameters
    ----------
    A : obj_array or list
        Observation model matrices
    B : obj_array or list  
        Transition model matrices
    C : obj_array or list
        Preference vectors
    D : obj_array or list, optional
        Prior beliefs over initial states
    control_fac_idx : list, optional
        Indices of controllable factors
    **kwargs
        Additional arguments for Agent initialization
        
    Returns
    -------
    agent : pymdp.agent.Agent
        Initialized PyMDP agent
    """
    return PyMDPCore.create_agent(A, B, C, D, control_fac_idx=control_fac_idx, **kwargs)


def run_agent_loop(agent, observation, verbose=False):
    """
    Run the standard PyMDP agent loop using PyMDP core utilities.
    
    Parameters
    ----------
    agent : pymdp.agent.Agent
        PyMDP agent instance
    observation : list or np.ndarray or int
        Current observation
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    beliefs : obj_array
        Posterior beliefs over states
    action : np.ndarray
        Selected action
    """
    qs, q_pi, action = PyMDPCore.run_agent_step(agent, observation)
    
    if verbose:
        print(f"Observation: {observation}")
        print(f"Beliefs: {[b.round(3) for b in qs]}")
        print(f"Action: {action}")
    
    # Convert beliefs to list format for compatibility
    beliefs_list = [qs[i] for i in range(len(qs))]
    
    return beliefs_list, action


def infer_states_via_pymdp(A, observation, prior):
    """
    Infer posterior states using PyMDP core utilities.
    
    Parameters
    ----------
    A : obj_array
        Observation model(s)
    observation : list[int] | int
        Observation(s)
    prior : obj_array
        Prior beliefs (object array of factors)

    Returns
    -------
    posterior : obj_array
        Posterior beliefs over hidden states (object array)
    """
    # Create temporary agent for inference
    temp_agent = PyMDPCore.create_agent(A, np.eye(3)[:, :, np.newaxis])
    temp_agent.qs = prior
    
    return PyMDPCore.infer_states(temp_agent, observation)


def compute_vfe_using_pymdp(A, observation, prior, posterior=None):
    """
    Compute Variational Free Energy using PyMDP core utilities.

    Parameters
    ----------
    A : obj_array
        Observation model(s)
    observation : list[int] | int
        Observation(s)
    prior : obj_array
        Prior beliefs
    posterior : obj_array, optional
        Posterior beliefs (to avoid recomputation)

    Returns
    -------
    vfe : float
        Variational Free Energy
    components : dict
        Dict with keys: complexity, accuracy
    posterior : obj_array
        Posterior beliefs used for VFE
    """
    vfe, components, posterior = PyMDPCore.compute_vfe(A, observation, prior, posterior)
    return vfe, components, posterior


def compute_policy_efe(A, B, C, beliefs, policy, policy_len=None, verbose=False):
    """
    Compute Expected Free Energy using PyMDP core utilities.

    Parameters
    ----------
    A : obj_array
        Observation model(s)
    B : obj_array
        Transition model(s)
    C : obj_array
        Preferences over observations
    beliefs : np.ndarray
        Current beliefs over states
    policy : list[int] | np.ndarray
        Action sequence
    policy_len : int, optional
        Planning horizon
    verbose : bool
        Print decomposition per step

    Returns
    -------
    result : dict
        Keys: efe, pragmatic_value, epistemic_value, expected_obs, final_beliefs
    """
    efe, components = PyMDPCore.compute_efe(A, B, C, beliefs, policy)
    
    return {
        'efe': efe,
        'pragmatic_value': components.get('pragmatic_value', 0.0),
        'epistemic_value': components.get('epistemic_value', 0.0),
        'expected_obs': None,  # Could be computed if needed
        'final_beliefs': beliefs,  # Could be computed if needed
    }


def simulate_environment_step(state, action, B_matrices, A_matrices, verbose=False):
    """
    Simulate environment dynamics using PyMDP core utilities.
    
    Parameters
    ----------
    state : list
        Current environmental state (one per factor)
    action : np.ndarray or list  
        Action taken by agent
    B_matrices : obj_array
        Transition model matrices
    A_matrices : obj_array
        Observation model matrices  
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    new_state : list
        New environmental state
    observation : list
        Generated observation
    """
    return PyMDPCore.simulate_environment(state, action, B_matrices, A_matrices, verbose=verbose)


def create_simple_agent_demo(num_states=3, num_actions=2, num_obs=None, num_steps=10, verbose=True):
    """
    Create a simple agent demonstration following the PyMDP example patterns.
    
    Parameters
    ----------
    num_states : int
        Number of states in the environment
    num_actions : int
        Number of possible actions
    num_obs : int, optional
        Number of observations (defaults to num_states)
    num_steps : int
        Number of simulation steps
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    results : dict
        Simulation results including agent, state history, etc.
    """
    
    if num_obs is None:
        num_obs = num_states
    
    if verbose:
        print("Creating simple PyMDP agent demonstration")
        print(f"States: {num_states}, Actions: {num_actions}, Observations: {num_obs}")
    
    # Build generative model matrices
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.eye(num_states, num_obs)  # Perfect observation
    
    B = obj_array_zeros([[num_states, num_states, num_actions]])
    # Simple left-right actions - ensure normalization
    for s in range(num_states):
        for a in range(num_actions):
            if a == 0:  # Move left
                next_s = max(0, s - 1)
            else:  # Move right  
                next_s = min(num_states - 1, s + 1)
            B[0][next_s, s, a] = 1.0
    
    # Ensure B matrix is normalized (columns sum to 1)
    for a in range(num_actions):
        for s in range(num_states):
            col_sum = B[0][:, s, a].sum()
            if col_sum > 0:
                B[0][:, s, a] = B[0][:, s, a] / col_sum
    
    # Preferences - prefer rightmost state
    C = obj_array_zeros([[num_obs]])
    C[0] = np.linspace(-1, 1, num_obs)  # Increasing preference
    
    # Create agent
    agent = create_agent_from_matrices(A, B, C, control_fac_idx=[0])
    
    # Simulation
    state = [num_states // 2]  # Start in middle
    state_history = [state[0]]
    observation_history = []
    action_history = []
    
    if verbose:
        print(f"\nStarting simulation (initial state: {state[0]})")
        print("=" * 40)
    
    for t in range(num_steps):
        # Generate observation
        obs_probs = A[0][:, state[0]]
        observation = [sample(obs_probs)]
        observation_history.append(observation[0])
        
        # Agent step
        beliefs, action = run_agent_loop(agent, observation, verbose=verbose)
        action_history.append(action[0])
        
        # Environment step  
        state, _ = simulate_environment_step(state, action, B, A, verbose=verbose)
        state_history.append(state[0])
        
        if verbose:
            print(f"Step {t+1}: Obs={observation[0]}, Action={action[0]}, State={state[0]}")
    
    return {
        'agent': agent,
        'state_history': state_history,
        'observation_history': observation_history, 
        'action_history': action_history,
        'A': A,
        'B': B,
        'C': C
    }


# Compatibility functions for legacy code
def legacy_vfe_inference(A, observation, prior, verbose=False):
    """
    Legacy VFE-based inference for compatibility.
    This wraps PyMDP methods to maintain compatibility with existing textbook code.
    """
    # For now, we'll use manual Bayes rule since PyMDP's algos API has changed
    if verbose:
        print("Using manual Bayes rule for VFE inference compatibility")
    
    # Convert to PyMDP format
    if not utils.is_obj_array(A):
        A = utils.to_obj_array(A)
    if not utils.is_obj_array(prior):
        prior = utils.to_obj_array(prior)
    
    # Manual Bayes rule calculation
    from pymdp.maths import spm_log
    
    # Handle observation format
    if isinstance(observation, int):
        likelihood = A[0][observation, :]
    else:
        # Assume it's already in the right format
        likelihood = A[0][observation, :]
    
    # Bayes rule: P(s|o) = P(o|s) * P(s) / P(o)
    joint = likelihood * prior[0] 
    evidence = np.sum(joint)
    
    if evidence > 1e-16:
        posterior = joint / evidence
    else:
        posterior = prior[0].copy()
    
    if verbose:
        print(f"Manual inference - Prior: {prior[0].round(3)}, Posterior: {posterior.round(3)}")
        
    # Create obj_array directly instead of using to_obj_array with list
    result = utils.obj_array_zeros([[len(posterior)]])
    result[0] = posterior
    return result
