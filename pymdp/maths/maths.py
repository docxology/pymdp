import numpy as np
from typing import List, Tuple, Optional, Union

def softmax(x, axis=None):
    """
    Compute softmax values for array x
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def spm_dot(X, x):
    """
    Dot product for multidimensional arrays
    """
    if len(X.shape) == 2:
        return X.dot(x)
    else:
        return np.tensordot(X, x, axes=(-1, 0))

def spm_log(arr):
    """
    Safe log function that returns -32.0 for zero elements
    """
    log_arr = np.log(arr)
    log_arr[np.isneginf(log_arr)] = -32.0
    return log_arr

def spm_log_single(arr):
    """
    Safe log for single array that returns -32.0 for zero elements
    """
    return spm_log(arr)

def spm_log_obj_array(obj_arr):
    """
    Apply safe log to object array
    """
    return np.array([spm_log(arr) for arr in obj_arr])

def dot_likelihood(A: np.ndarray, obs: int) -> np.ndarray:
    """
    Dot product with likelihood array for a specific observation
    
    Parameters
    ----------
    A : np.ndarray
        Likelihood array
    obs : int
        Observation index
        
    Returns
    -------
    np.ndarray
        Result of dot product with likelihood
    """
    return A[obs, ...]

def calc_free_energy(qs: List[np.ndarray], 
                    prior: List[np.ndarray], 
                    likelihood: np.ndarray) -> float:
    """
    Calculate variational free energy
    
    Parameters
    ----------
    qs : List[np.ndarray]
        Current beliefs about hidden states
    prior : List[np.ndarray]
        Prior beliefs about hidden states
    likelihood : np.ndarray
        Likelihood array
        
    Returns
    -------
    float
        Variational free energy
    """
    acc = 0.0
    for i, (q, p) in enumerate(zip(qs, prior)):
        acc += np.sum(q * spm_log(q / p))
    return acc - np.sum(likelihood * np.log(likelihood))

def get_joint_likelihood_seq(A: List[np.ndarray], 
                           obs_seq: List[List[int]], 
                           num_states: List[int]) -> np.ndarray:
    """
    Compute joint likelihood of observation sequence
    
    Parameters
    ----------
    A : List[np.ndarray]
        Sensory likelihood mappings
    obs_seq : List[List[int]]
        Sequence of observations
    num_states : List[int]
        Number of states in each hidden state factor
        
    Returns
    -------
    np.ndarray
        Joint likelihood of the observation sequence
    """
    num_obs = len(obs_seq)
    likelihood_seq = np.ones(tuple(num_states))
    
    for t in range(num_obs):
        likelihood_seq *= get_joint_likelihood(A, obs_seq[t])
        
    return likelihood_seq

def get_joint_likelihood(A: List[np.ndarray], 
                        observation: List[int]) -> np.ndarray:
    """
    Compute joint likelihood of a single observation under the generative model
    
    Parameters
    ----------
    A : List[np.ndarray]
        Sensory likelihood mappings
    observation : List[int]
        Single observation
        
    Returns
    -------
    np.ndarray
        Joint likelihood
    """
    likelihood = np.ones([A_m.shape[1] for A_m in A])
    
    for modality, obs in enumerate(observation):
        likelihood *= get_modality_likelihood(A[modality], obs)
        
    return likelihood

def get_joint_likelihood_seq_by_modality(A: List[np.ndarray], 
                                       obs_seq: List[List[int]], 
                                       num_states: List[int],
                                       modality: int) -> np.ndarray:
    """
    Compute joint likelihood of an observation sequence for a specific modality
    
    Parameters
    ----------
    A : List[np.ndarray]
        Sensory likelihood mappings
    obs_seq : List[List[int]]
        Sequence of observations
    num_states : List[int]
        Number of states in each hidden state factor
    modality : int
        Index of modality to compute likelihood for
        
    Returns
    -------
    np.ndarray
        Joint likelihood for the specified modality
    """
    num_obs = len(obs_seq)
    likelihood_seq = np.ones(tuple(num_states))
    
    for t in range(num_obs):
        likelihood_seq *= get_modality_likelihood(A[modality], obs_seq[t][modality])
        
    return likelihood_seq

def get_modality_likelihood(A_m: np.ndarray, 
                          observation: int) -> np.ndarray:
    """
    Get likelihood for a single modality
    
    Parameters
    ----------
    A_m : np.ndarray
        Likelihood mapping for a single modality
    observation : int
        Observation index for that modality
        
    Returns
    -------
    np.ndarray
        Likelihood for that modality
    """
    return A_m[observation, ...]

def spm_norm(A: np.ndarray) -> np.ndarray:
    """
    Normalize a probability distribution (or array of distributions)
    
    Parameters
    ----------
    A : np.ndarray
        Array to normalize
        
    Returns
    -------
    np.ndarray
        Normalized array
    """
    A = A + 1e-16  # Add small constant for numerical stability
    if len(A.shape) == 1:
        return A / np.sum(A)
    else:
        return A / np.sum(A, axis=0)

def factor_dot_flex(A: np.ndarray, x: np.ndarray, dims_to_omit: Optional[List[int]] = None) -> np.ndarray:
    """
    Flexible dot product of arrays with different shapes
    
    Parameters
    ----------
    A : np.ndarray
        First array
    x : np.ndarray
        Second array
    dims_to_omit : List[int], optional
        Dimensions to omit from dot product
        
    Returns
    -------
    np.ndarray
        Result of dot product
    """
    if dims_to_omit is None:
        dims_to_omit = []
    
    if len(A.shape) == 1 or len(x.shape) == 1:
        return A.dot(x)
    else:
        A_dims = list(range(len(A.shape)))
        x_dims = list(range(len(x.shape)))
        shared_dims = [d for d in A_dims if d not in dims_to_omit]
        return np.tensordot(A, x, axes=(shared_dims, shared_dims))

def softmax_obj_arr(arr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply softmax to each array in an object array
    
    Parameters
    ----------
    arr : List[np.ndarray]
        List of arrays to apply softmax to
        
    Returns
    -------
    List[np.ndarray]
        List of softmaxed arrays
    """
    return [softmax(a) for a in arr]

def spm_wnorm(A: np.ndarray) -> np.ndarray:
    """
    Weighted normalization of an array
    
    Parameters
    ----------
    A : np.ndarray
        Array to normalize
        
    Returns
    -------
    np.ndarray
        Normalized array
    """
    A = A + 1e-16
    return A / np.sum(A, axis=0)

def spm_MDP_G(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute expected free energy for policy evaluation
    
    Parameters
    ----------
    A : np.ndarray
        Likelihood array
    x : np.ndarray
        Hidden state distribution
        
    Returns
    -------
    np.ndarray
        Expected free energy
    """
    return spm_dot(A, x)

def kl_div(q: np.ndarray, p: np.ndarray) -> float:
    """
    Calculate KL divergence between two distributions
    
    Parameters
    ----------
    q : np.ndarray
        First distribution
    p : np.ndarray
        Second distribution
        
    Returns
    -------
    float
        KL divergence
    """
    q = q + 1e-16
    p = p + 1e-16
    return np.sum(q * np.log(q / p))

def entropy(A: np.ndarray) -> float:
    """
    Calculate entropy of a distribution
    
    Parameters
    ----------
    A : np.ndarray
        Probability distribution
        
    Returns
    -------
    float
        Entropy value
    """
    A = A + 1e-16
    return -np.sum(A * np.log(A))