"""Analytical processing for Active Inference trajectories."""

from __future__ import annotations

import jax.numpy as jnp


def compute_entropy(probs: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Calculate Shannon entropy for categorical probability distributions.
    
    Args:
        probs: Categorical probabilities (must sum to 1 over `axis`).
        axis: The axis along which to compute entropy.
        
    Returns:
        Shannon entropy of the distribution.
    """
    # Add small constant to avoid log(0)
    eps = 1e-12
    p_safe = jnp.clip(probs, eps, 1.0)
    return -jnp.sum(probs * jnp.log(p_safe), axis=axis)


def trajectory_divergence(qs_seq: list[jnp.ndarray], prior: jnp.ndarray) -> jnp.ndarray:
    """Compute KL-divergence of a belief trajectory against a static prior.
    
    Args:
        qs_seq: List of posterior belief arrays over time.
        prior: A fixed prior belief array (e.g., initial state belief).
        
    Returns:
        Array of KL divergence values mapping to the trajectory steps.
    """
    eps = 1e-12
    p_safe = jnp.clip(prior, eps, 1.0)
    
    divergences = []
    for qs in qs_seq:
        q_safe = jnp.clip(qs, eps, 1.0)
        # KL(q || p) = sum(q * log(q / p))
        kl = jnp.sum(qs * (jnp.log(q_safe) - jnp.log(p_safe)))
        divergences.append(kl)
        
    return jnp.array(divergences)


def marginalize_actions(q_pi: jnp.ndarray, policies: jnp.ndarray) -> jnp.ndarray:
    """Convert policy posteriors into marginal action probabilities.
    
    Args:
        q_pi: Posterior probability over policies, shape (num_policies,).
        policies: Array mapping policies to actions, shape (num_policies, policy_len, num_factors).
        
    Returns:
        Marginal probabilities of the immediate next action for each control factor.
    """
    if len(q_pi.shape) == 1:
        # Standard 1D policy posterior
        immediate_actions = policies[:, 0, :]  # shape: (num_policies, num_factors)
        
        # Max action across any policy to determine required array bounds
        max_action = int(jnp.max(immediate_actions)) + 1
        num_factors = immediate_actions.shape[1]
        
        marginal_probs = jnp.zeros((num_factors, max_action))
        
        # This is a basic NumPy-esque loop; for purely compiled JAX, `jax.lax.fori_loop` or 
        # `jax.ops.segment_sum` could be used, but standard JAX traces standard for loops OK if static.
        for policy_idx in range(len(q_pi)):
            prob = q_pi[policy_idx]
            for factor_idx in range(num_factors):
                action = int(immediate_actions[policy_idx, factor_idx])
                marginal_probs = marginal_probs.at[factor_idx, action].add(prob)
                
        return marginal_probs
    else:
        # If it's factorized or batched differently
        return q_pi

def compute_accuracy_complexity(q_s: jnp.ndarray, p_o_given_s: jnp.ndarray, obs: int, prior: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose Variational Free Energy into Accuracy (Expected Log Likelihood) and Complexity (KL Divergence).
    
    Args:
        q_s: Posterior beliefs over states.
        p_o_given_s: Likelihood mapping P(o|s).
        obs: Current observation index.
        prior: Prior beliefs P(s_t|s_{t-1}, u_{t-1}).
        
    Returns:
        (accuracy, complexity)
    """
    eps = 1e-12
    
    # 1. Accuracy: E_q[log P(o_t | s_t)]
    # Maps the probability of the actual observation given the current hidden state beliefs
    likelihood = jnp.clip(p_o_given_s[obs, :], eps, 1.0)
    accuracy = jnp.sum(q_s * jnp.log(likelihood))
    
    # 2. Complexity: KL( q(s_t) || P(s_t | s_{t-1}, u_{t-1}) )
    # The divergence between posterior and prior predictions
    q_safe = jnp.clip(q_s, eps, 1.0)
    prior_safe = jnp.clip(prior, eps, 1.0)
    complexity = jnp.sum(q_s * (jnp.log(q_safe) - jnp.log(prior_safe)))
    
    return accuracy, complexity
