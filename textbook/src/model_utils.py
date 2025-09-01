"""
Model Construction Utilities
============================

Enhanced utilities for building and validating PyMDP models.
Provides convenient functions for creating common model types and
ensuring proper model specification.
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional, Union
from pymdp.utils import obj_array_zeros, obj_array_uniform


def create_gridworld_model(
    height: int = 3, 
    width: int = 3,
    goal_location: Optional[Tuple[int, int]] = None,
    obstacle_locations: Optional[List[Tuple[int, int]]] = None,
    observation_noise: float = 0.0,
    movement_noise: float = 0.1,
    goal_reward: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid world model for active inference.
    
    Parameters
    ----------
    height : int
        Grid height in cells
    width : int  
        Grid width in cells
    goal_location : tuple, optional
        (row, col) position of goal. Defaults to bottom-right corner.
    obstacle_locations : list of tuples, optional
        List of (row, col) obstacle positions
    observation_noise : float
        Probability of observing wrong location (0 = perfect observation)
    movement_noise : float
        Probability of movement failure (stay in place)
    goal_reward : float
        Preference strength for goal location
        
    Returns
    -------
    A : obj_array
        Observation model
    B : obj_array  
        Transition model
    C : obj_array
        Preference model
    D : obj_array
        Prior over initial states
    """
    
    num_states = height * width
    num_actions = 4  # up, down, left, right
    
    # Remove obstacles from valid states
    if obstacle_locations is None:
        obstacle_locations = []
    
    obstacle_indices = [row * width + col for row, col in obstacle_locations]
    valid_states = [i for i in range(num_states) if i not in obstacle_indices]
    
    # Default goal to bottom-right
    if goal_location is None:
        goal_location = (height - 1, width - 1)
    
    goal_index = goal_location[0] * width + goal_location[1]
    
    # Observation model: can observe current position (with noise)
    A = obj_array_zeros([[num_states, num_states]])
    for s in range(num_states):
        if s in valid_states:
            A[0][s, s] = 1.0 - observation_noise
            # Add observation noise uniformly to other states
            if observation_noise > 0:
                noise_per_state = observation_noise / (num_states - 1)
                for other_s in range(num_states):
                    if other_s != s:
                        A[0][other_s, s] = noise_per_state
        else:
            # Obstacle states are unobservable
            A[0][:, s] = 1.0 / num_states
    
    # Transition model: grid movement
    B = obj_array_zeros([[num_states, num_states, num_actions]])
    
    for s in range(num_states):
        if s not in valid_states:
            # Obstacle states: can't be in them
            continue
            
        row, col = divmod(s, width)
        
        for a in range(num_actions):
            # Calculate intended next state
            if a == 0:  # up
                next_row = max(0, row - 1)
                next_col = col
            elif a == 1:  # down  
                next_row = min(height - 1, row + 1)
                next_col = col
            elif a == 2:  # left
                next_row = row
                next_col = max(0, col - 1)
            else:  # right
                next_row = row
                next_col = min(width - 1, col + 1)
            
            intended_next_state = next_row * width + next_col
            
            # Check if intended state is valid
            if intended_next_state in valid_states:
                # Movement succeeds with probability (1 - noise)
                B[0][intended_next_state, s, a] = 1.0 - movement_noise
                # Movement fails (stay in place) with probability noise
                B[0][s, s, a] = movement_noise
            else:
                # Intended state is obstacle or boundary - stay in place
                B[0][s, s, a] = 1.0
    
    # Preferences: strong preference for goal location
    C = obj_array_zeros([num_states])
    C[0] = np.zeros(num_states)
    C[0][goal_index] = goal_reward
    
    # Obstacles have negative preference
    for obstacle_idx in obstacle_indices:
        C[0][obstacle_idx] = -goal_reward
    
    # Prior: uniform over valid starting states
    D = obj_array_zeros([num_states])
    D[0] = np.zeros(num_states)
    for s in valid_states:
        if s != goal_index:  # Don't start at goal
            D[0][s] = 1.0
    D[0] = D[0] / np.sum(D[0])  # Normalize
    
    return A, B, C, D


def create_tmaze_model(
    arm_length: int = 3,
    cue_location: str = "center", 
    reward_side: str = "left",
    observation_noise: float = 0.05,
    movement_noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a T-maze model for active inference.
    
    Parameters
    ----------
    arm_length : int
        Length of T-maze arms
    cue_location : str
        Where cue is presented ("center" or "start")
    reward_side : str
        Which arm has reward ("left" or "right")
    observation_noise : float
        Observation uncertainty
    movement_noise : float
        Movement uncertainty
        
    Returns
    -------
    A, B, C, D : obj_array
        Model components
    """
    
    # State space: [start, center, left_arm_1, ..., left_arm_n, right_arm_1, ..., right_arm_n]
    num_states = 2 + 2 * arm_length  # start + center + left_arm + right_arm
    num_actions = 4  # up, down, left, right
    
    # Observation space: depends on cue configuration
    if cue_location == "center":
        num_obs = 4  # start, center_no_cue, center_left_cue, center_right_cue
    else:
        num_obs = 3  # start, center, arms
    
    # Build observation model
    A = obj_array_zeros([[num_obs, num_states]])
    
    # State indices
    start_idx = 0
    center_idx = 1
    left_arm_start = 2
    left_arm_end = 1 + arm_length
    right_arm_start = 2 + arm_length  
    right_arm_end = 1 + 2 * arm_length
    
    # Build transition model
    B = obj_array_zeros([[num_states, num_states, num_actions]])
    
    # Implement T-maze connectivity
    # (This is a simplified version - full implementation would be more complex)
    
    # Preferences: reward at end of specified arm
    C = obj_array_zeros([num_obs])
    C[0] = np.zeros(num_obs)
    
    if reward_side == "left":
        # Reward observation at left arm end
        C[0][-2] = 2.0  # Assuming left reward obs is second to last
    else:
        # Reward observation at right arm end  
        C[0][-1] = 2.0  # Assuming right reward obs is last
    
    # Prior: start at beginning
    D = obj_array_zeros([num_states])
    D[0] = np.zeros(num_states)
    D[0][start_idx] = 1.0
    
    return A, B, C, D


def create_random_model(
    num_obs: int,
    num_states: int, 
    num_actions: int,
    sparsity: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a random but valid model for testing.
    
    Parameters
    ----------
    num_obs : int
        Number of observations
    num_states : int
        Number of hidden states
    num_actions : int
        Number of actions
    sparsity : float
        Sparsity of transition matrices (0 = dense, 1 = very sparse)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    A, B, C, D : obj_array
        Random model components
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Random observation model
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.random.rand(num_obs, num_states)
    A[0] = A[0] / A[0].sum(axis=0, keepdims=True)  # Normalize columns
    
    # Random transition model with sparsity
    B = obj_array_zeros([[num_states, num_states, num_actions]])
    for a in range(num_actions):
        B[0][:, :, a] = np.random.rand(num_states, num_states)
        
        # Apply sparsity
        if sparsity > 0:
            mask = np.random.rand(num_states, num_states) > sparsity
            B[0][:, :, a] *= mask
        
        # Normalize columns
        B[0][:, :, a] = B[0][:, :, a] / (B[0][:, :, a].sum(axis=0, keepdims=True) + 1e-10)
    
    # Random preferences
    C = obj_array_zeros([num_obs])
    C[0] = np.random.randn(num_obs)
    
    # Random initial state distribution
    D = obj_array_uniform([num_states])
    
    return A, B, C, D


def validate_model(A, B, C, D, verbose: bool = True) -> bool:
    """
    Validate that a model is properly specified.
    
    Parameters
    ----------
    A : obj_array
        Observation model
    B : obj_array  
        Transition model
    C : obj_array
        Preferences
    D : obj_array
        Initial state distribution
    verbose : bool
        Whether to print validation details
        
    Returns
    -------
    is_valid : bool
        Whether model passes all validation checks
    """
    
    is_valid = True
    issues = []
    
    # Check A matrix
    try:
        if not np.allclose(A[0].sum(axis=0), 1.0, rtol=1e-3):
            issues.append("A matrix columns don't sum to 1")
            is_valid = False
        
        if np.any(A[0] < 0):
            issues.append("A matrix has negative values")
            is_valid = False
            
    except Exception as e:
        issues.append(f"Error checking A matrix: {e}")
        is_valid = False
    
    # Check B matrices
    try:
        for a in range(B[0].shape[2]):
            if not np.allclose(B[0][:, :, a].sum(axis=0), 1.0, rtol=1e-3):
                issues.append(f"B matrix for action {a} columns don't sum to 1")
                is_valid = False
                
            if np.any(B[0][:, :, a] < 0):
                issues.append(f"B matrix for action {a} has negative values")
                is_valid = False
                
    except Exception as e:
        issues.append(f"Error checking B matrix: {e}")
        is_valid = False
    
    # Check D (initial state distribution)
    try:
        if not np.allclose(D[0].sum(), 1.0, rtol=1e-3):
            issues.append("D (initial states) doesn't sum to 1")
            is_valid = False
            
        if np.any(D[0] < 0):
            issues.append("D (initial states) has negative values")
            is_valid = False
            
    except Exception as e:
        issues.append(f"Error checking D: {e}")
        is_valid = False
    
    # Check dimension compatibility
    try:
        num_obs, num_states = A[0].shape
        num_states_B = B[0].shape[0]
        num_states_D = len(D[0])
        num_obs_C = len(C[0])
        
        if num_states != num_states_B:
            issues.append(f"Dimension mismatch: A has {num_states} states, B has {num_states_B}")
            is_valid = False
            
        if num_states != num_states_D:
            issues.append(f"Dimension mismatch: A has {num_states} states, D has {num_states_D}")
            is_valid = False
            
        if num_obs != num_obs_C:
            issues.append(f"Dimension mismatch: A has {num_obs} observations, C has {num_obs_C}")
            is_valid = False
            
    except Exception as e:
        issues.append(f"Error checking dimensions: {e}")
        is_valid = False
    
    if verbose:
        if is_valid:
            print("✓ Model validation passed!")
        else:
            print("✗ Model validation failed:")
            for issue in issues:
                print(f"  - {issue}")
    
    return is_valid


def normalize_model(A, B, C, D):
    """
    Normalize model components to ensure they represent valid distributions.
    
    Parameters
    ----------
    A, B, C, D : obj_array
        Model components (will be modified in-place)
        
    Returns
    -------
    A, B, C, D : obj_array
        Normalized model components
    """
    
    # Normalize A matrix (columns should sum to 1)
    A[0] = A[0] / (A[0].sum(axis=0, keepdims=True) + 1e-16)
    
    # Normalize B matrices (columns should sum to 1)
    for a in range(B[0].shape[2]):
        B[0][:, :, a] = B[0][:, :, a] / (B[0][:, :, a].sum(axis=0, keepdims=True) + 1e-16)
    
    # Normalize D (should sum to 1)
    D[0] = D[0] / (D[0].sum() + 1e-16)
    
    # C doesn't need normalization (it's preferences, not probabilities)
    
    return A, B, C, D
