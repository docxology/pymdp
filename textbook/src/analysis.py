"""
Analysis Utilities
==================

Tools for analyzing active inference models and agent behavior.
Provides metrics and diagnostics for understanding model performance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
from pymdp.maths import kl_div


def compute_model_entropy(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """
    Compute entropy measures for model components.
    
    Parameters
    ----------
    A : np.ndarray
        Observation model matrix
    B : np.ndarray  
        Transition model tensor
        
    Returns
    -------
    entropies : dict
        Dictionary of entropy measures
    """
    
    entropies = {}
    
    # Observation model entropy (per state)
    obs_entropies = []
    for s in range(A.shape[1]):
        obs_dist = A[:, s]
        if np.sum(obs_dist) > 0:
            obs_dist = obs_dist / np.sum(obs_dist)  # Normalize
            entropy = -np.sum(obs_dist * np.log(obs_dist + 1e-16))
            obs_entropies.append(entropy)
    
    entropies['obs_entropy_mean'] = np.mean(obs_entropies)
    entropies['obs_entropy_std'] = np.std(obs_entropies)
    entropies['obs_entropy_max'] = np.max(obs_entropies)
    entropies['obs_entropy_min'] = np.min(obs_entropies)
    
    # Transition model entropy (per state-action pair)
    trans_entropies = []
    for s in range(B.shape[1]):
        for a in range(B.shape[2]):
            trans_dist = B[:, s, a]
            if np.sum(trans_dist) > 0:
                trans_dist = trans_dist / np.sum(trans_dist)  # Normalize
                entropy = -np.sum(trans_dist * np.log(trans_dist + 1e-16))
                trans_entropies.append(entropy)
    
    entropies['trans_entropy_mean'] = np.mean(trans_entropies)
    entropies['trans_entropy_std'] = np.std(trans_entropies)
    entropies['trans_entropy_max'] = np.max(trans_entropies)
    entropies['trans_entropy_min'] = np.min(trans_entropies)
    
    return entropies


def analyze_policy_complexity(
    policies: List[List[int]],
    policy_probs: np.ndarray
) -> Dict[str, Union[float, int]]:
    """
    Analyze complexity of policy distribution.
    
    Parameters
    ----------
    policies : list of lists
        Policy sequences
    policy_probs : np.ndarray
        Probability distribution over policies
        
    Returns
    -------
    complexity : dict
        Dictionary of complexity measures
    """
    
    complexity = {}
    
    # Policy entropy
    policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-16))
    complexity['policy_entropy'] = policy_entropy
    complexity['normalized_policy_entropy'] = policy_entropy / np.log(len(policies))
    
    # Effective number of policies
    complexity['effective_policies'] = np.exp(policy_entropy)
    
    # Policy length statistics
    lengths = [len(policy) for policy in policies]
    complexity['mean_policy_length'] = np.mean(lengths)
    complexity['std_policy_length'] = np.std(lengths)
    complexity['max_policy_length'] = np.max(lengths)
    
    # Action diversity (number of unique actions used)
    all_actions = set()
    for policy in policies:
        all_actions.update(policy)
    complexity['action_diversity'] = len(all_actions)
    
    # Most probable policy
    most_probable_idx = np.argmax(policy_probs)
    complexity['max_policy_prob'] = policy_probs[most_probable_idx]
    complexity['most_probable_policy'] = policies[most_probable_idx]
    
    return complexity


def measure_exploration(
    belief_history: List[np.ndarray],
    action_history: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Measure exploration behavior of agent.
    
    Parameters
    ----------
    belief_history : list of np.ndarray
        History of belief distributions
    action_history : list of int, optional
        History of actions taken
        
    Returns
    -------
    exploration : dict
        Dictionary of exploration measures
    """
    
    exploration = {}
    
    if len(belief_history) < 2:
        return exploration
    
    # Belief entropy over time
    entropies = []
    for beliefs in belief_history:
        entropy = -np.sum(beliefs * np.log(beliefs + 1e-16))
        entropies.append(entropy)
    
    exploration['mean_belief_entropy'] = np.mean(entropies)
    exploration['std_belief_entropy'] = np.std(entropies)
    exploration['entropy_trend'] = np.polyfit(range(len(entropies)), entropies, 1)[0]
    
    # Belief change over time (KL divergence between consecutive beliefs)
    belief_changes = []
    for i in range(1, len(belief_history)):
        kl = kl_div(belief_history[i], belief_history[i-1])
        belief_changes.append(kl)
    
    exploration['mean_belief_change'] = np.mean(belief_changes)
    exploration['std_belief_change'] = np.std(belief_changes)
    exploration['total_belief_change'] = np.sum(belief_changes)
    
    # Action diversity (if actions provided)
    if action_history is not None:
        unique_actions = len(set(action_history))
        exploration['action_diversity'] = unique_actions
        exploration['action_diversity_ratio'] = unique_actions / len(action_history)
        
        # Action entropy
        action_counts = np.bincount(action_history)
        action_probs = action_counts / len(action_history)
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-16))
        exploration['action_entropy'] = action_entropy
    
    return exploration


def evaluate_performance(
    observations: List[int],
    preferences: np.ndarray,
    actions: Optional[List[int]] = None,
    action_costs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate agent performance based on preferences and costs.
    
    Parameters
    ----------
    observations : list of int
        Sequence of observations
    preferences : np.ndarray
        Preference values for each observation
    actions : list of int, optional
        Sequence of actions
    action_costs : np.ndarray, optional
        Cost for each action
        
    Returns
    -------
    performance : dict
        Dictionary of performance measures
    """
    
    performance = {}
    
    # Preference-based reward
    rewards = [preferences[obs] for obs in observations]
    performance['total_reward'] = np.sum(rewards)
    performance['mean_reward'] = np.mean(rewards)
    performance['std_reward'] = np.std(rewards)
    performance['cumulative_reward'] = np.cumsum(rewards).tolist()
    
    # Preference achievement rate
    max_preference = np.max(preferences)
    high_pref_obs = np.sum(np.array(rewards) >= 0.8 * max_preference)
    performance['high_preference_rate'] = high_pref_obs / len(observations)
    
    # Action costs (if provided)
    if actions is not None and action_costs is not None:
        costs = [action_costs[action] for action in actions]
        performance['total_cost'] = np.sum(costs)
        performance['mean_cost'] = np.mean(costs)
        performance['net_reward'] = performance['total_reward'] - performance['total_cost']
    
    # Efficiency measures
    performance['reward_per_step'] = performance['mean_reward']
    if len(observations) > 1:
        performance['reward_acceleration'] = (rewards[-1] - rewards[0]) / (len(observations) - 1)
    
    return performance


def compute_information_gain(
    prior_beliefs: np.ndarray,
    posterior_beliefs: np.ndarray
) -> float:
    """
    Compute information gain from prior to posterior.
    
    Parameters
    ----------
    prior_beliefs : np.ndarray
        Prior belief distribution
    posterior_beliefs : np.ndarray
        Posterior belief distribution
        
    Returns
    -------
    info_gain : float
        Information gain (KL divergence from prior to posterior)
    """
    
    # Information gain = KL(posterior || prior)
    info_gain = kl_div(posterior_beliefs, prior_beliefs)
    return info_gain


def analyze_convergence(
    values: List[float],
    window_size: int = 10,
    threshold: float = 1e-3
) -> Dict[str, Union[bool, int, float]]:
    """
    Analyze convergence properties of a time series.
    
    Parameters
    ----------
    values : list of float
        Time series values
    window_size : int
        Window size for stability analysis
    threshold : float
        Convergence threshold
        
    Returns
    -------
    convergence : dict
        Convergence analysis results
    """
    
    convergence = {}
    
    if len(values) < window_size:
        convergence['converged'] = False
        convergence['convergence_step'] = -1
        return convergence
    
    # Check for convergence (small changes in recent window)
    for i in range(window_size, len(values)):
        recent_values = values[i-window_size:i]
        variance = np.var(recent_values)
        
        if variance < threshold:
            convergence['converged'] = True
            convergence['convergence_step'] = i
            convergence['convergence_value'] = np.mean(recent_values)
            convergence['convergence_variance'] = variance
            break
    else:
        convergence['converged'] = False
        convergence['convergence_step'] = -1
    
    # Overall trend
    if len(values) > 2:
        trend_slope = np.polyfit(range(len(values)), values, 1)[0]
        convergence['trend_slope'] = trend_slope
        convergence['is_decreasing'] = trend_slope < -threshold
        convergence['is_increasing'] = trend_slope > threshold
        convergence['is_stable'] = abs(trend_slope) <= threshold
    
    # Final statistics
    convergence['final_value'] = values[-1]
    convergence['min_value'] = np.min(values)
    convergence['max_value'] = np.max(values)
    convergence['mean_value'] = np.mean(values)
    convergence['std_value'] = np.std(values)
    
    return convergence


def compare_models(
    model_results: Dict[str, Dict[str, float]]
) -> Dict[str, Union[str, Dict]]:
    """
    Compare performance of different models.
    
    Parameters
    ----------
    model_results : dict
        Dictionary mapping model names to their performance metrics
        
    Returns
    -------
    comparison : dict
        Model comparison results
    """
    
    comparison = {}
    
    if len(model_results) < 2:
        return comparison
    
    model_names = list(model_results.keys())
    
    # Find common metrics
    common_metrics = set(model_results[model_names[0]].keys())
    for results in model_results.values():
        common_metrics &= set(results.keys())
    
    comparison['common_metrics'] = list(common_metrics)
    
    # Rank models by each metric
    rankings = {}
    for metric in common_metrics:
        values = [(name, results[metric]) for name, results in model_results.items()]
        values.sort(key=lambda x: x[1], reverse=True)  # Higher is better
        rankings[metric] = [name for name, _ in values]
    
    comparison['rankings'] = rankings
    
    # Overall best model (most #1 rankings)
    first_place_counts = {}
    for name in model_names:
        first_place_counts[name] = sum(1 for ranking in rankings.values() 
                                     if ranking[0] == name)
    
    best_model = max(first_place_counts, key=first_place_counts.get)
    comparison['overall_best'] = best_model
    comparison['first_place_counts'] = first_place_counts
    
    # Statistical significance (if enough data)
    comparison['statistical_tests'] = {}
    for metric in common_metrics:
        values_by_model = {name: [results[metric]] for name, results in model_results.items()}
        
        # For now, just record the values (full statistical testing would require multiple runs)
        comparison['statistical_tests'][metric] = values_by_model
    
    return comparison
