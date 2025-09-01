"""
PyTest Configuration and Fixtures
================================

Common fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymdp
from pymdp.utils import obj_array_zeros, sample
from pymdp.maths import softmax


@pytest.fixture(scope="session")
def pymdp_version():
    """Get PyMDP version for testing compatibility."""
    return pymdp.__version__


@pytest.fixture
def simple_observation_model():
    """Create a simple observation model (A matrix) for testing."""
    # 3 observations, 2 hidden states
    A = obj_array_zeros([[3, 2]])
    A[0] = np.array([[0.9, 0.1],    # obs 0: likely from state 0
                     [0.1, 0.9],    # obs 1: likely from state 1  
                     [0.0, 0.0]])   # obs 2: impossible
    return A


@pytest.fixture
def simple_transition_model():
    """Create a simple transition model (B matrix) for testing."""
    # 2 states, 2 actions
    B = obj_array_zeros([[2, 2, 2]])
    # Action 0: stay in current state
    B[0][:, :, 0] = np.eye(2)
    # Action 1: switch states
    B[0][:, :, 1] = np.array([[0, 1], [1, 0]])
    return B


@pytest.fixture
def simple_preference_model():
    """Create simple preferences (C matrix) for testing."""
    # Prefer observation 0 over observation 1, avoid observation 2
    C = obj_array_zeros([3])
    C[0] = np.array([2.0, 0.0, -4.0])  # Strong preference for obs 0
    return C


@pytest.fixture
def simple_prior():
    """Create simple prior beliefs for testing."""
    # Uniform prior over 2 states
    prior = obj_array_zeros([2])
    prior[0] = np.ones(2) / 2
    return prior


@pytest.fixture
def simple_policy_prior():
    """Create simple policy prior for testing."""
    # 2 policies, uniform prior
    E = np.ones(2) / 2
    return E


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield 42
    # Cleanup: reset to random state after test
    np.random.seed()


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_gridworld_model(height=3, width=3):
        """Create a simple gridworld model for testing."""
        num_states = height * width
        num_actions = 4  # up, down, left, right
        num_obs = num_states  # can observe current position
        
        # Observation model: identity (can perfectly observe state)
        A = obj_array_zeros([[num_obs, num_states]])
        A[0] = np.eye(num_states)
        
        # Transition model: grid movement
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        for s in range(num_states):
            row, col = divmod(s, width)
            for a in range(num_actions):
                if a == 0:  # up
                    next_row = max(0, row - 1)
                    next_state = next_row * width + col
                elif a == 1:  # down
                    next_row = min(height - 1, row + 1)
                    next_state = next_row * width + col
                elif a == 2:  # left
                    next_col = max(0, col - 1)
                    next_state = row * width + next_col
                else:  # right
                    next_col = min(width - 1, col + 1)
                    next_state = row * width + next_col
                
                B[0][next_state, s, a] = 1.0
        
        # Preferences: goal at bottom-right corner
        C = obj_array_zeros([num_obs])
        C[0] = np.zeros(num_obs)
        C[0][-1] = 2.0  # Strong preference for bottom-right
        
        return A, B, C
    
    @staticmethod
    def create_tmaze_model():
        """Create a T-maze model for testing."""
        # States: [start, junction, left_arm, right_arm, left_reward, right_reward]
        num_states = 6
        num_actions = 4  # up, down, left, right
        num_obs = 4      # different observations at key locations
        
        # Observation model
        A = obj_array_zeros([[num_obs, num_states]])
        A[0] = np.array([
            [1, 0, 0, 0, 0, 0],  # obs 0: start location
            [0, 1, 0, 0, 0, 0],  # obs 1: junction
            [0, 0, 1, 0, 1, 0],  # obs 2: left side
            [0, 0, 0, 1, 0, 1],  # obs 3: right side
        ])
        
        return A


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# Custom assertions for testing
def assert_valid_probability_distribution(arr, rtol=1e-5):
    """Assert array is a valid probability distribution."""
    assert np.all(arr >= 0), "Probabilities must be non-negative"
    assert np.isclose(np.sum(arr), 1.0, rtol=rtol), f"Probabilities must sum to 1, got {np.sum(arr)}"


def assert_valid_log_probability_distribution(arr, rtol=1e-5):
    """Assert array is a valid log probability distribution."""
    # Convert from log space
    probs = np.exp(arr)
    assert_valid_probability_distribution(probs, rtol)


def assert_obj_array_structure(obj_arr, expected_shapes):
    """Assert object array has expected structure."""
    assert len(obj_arr) == len(expected_shapes), f"Expected {len(expected_shapes)} arrays, got {len(obj_arr)}"
    for i, expected_shape in enumerate(expected_shapes):
        assert obj_arr[i].shape == expected_shape, f"Array {i} has shape {obj_arr[i].shape}, expected {expected_shape}"


# Add custom assertions to pytest namespace
pytest.assert_valid_probability_distribution = assert_valid_probability_distribution
pytest.assert_valid_log_probability_distribution = assert_valid_log_probability_distribution
pytest.assert_obj_array_structure = assert_obj_array_structure
