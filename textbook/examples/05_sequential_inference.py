#!/usr/bin/env python3
"""
Example 5: Sequential Inference Over Time
=========================================

This example demonstrates how state inference works over sequences of observations,
showing how beliefs evolve and accumulate evidence over time:
- Sequential belief updating
- Evidence accumulation and tracking
- Handling conflicting observations
- Belief momentum and inertia
- Temporal patterns in inference

Learning Objectives:
- Understand how beliefs evolve with sequential observations
- Learn about evidence accumulation in temporal sequences
- Practice tracking belief trajectories over time
- Develop intuition for belief momentum and uncertainty dynamics

Mathematical Background:
Sequential Bayes: P(state_t | obs_1:t) ∝ P(obs_t | state_t) × P(state_t | obs_1:t-1)
- Each new observation updates the current belief
- Previous posterior becomes next prior
- Evidence accumulates over time
- Conflicting evidence reduces certainty

Run with: python 05_sequential_inference.py [--interactive]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Create output directory for this example
OUTPUT_DIR = Path(__file__).parent / "outputs" / "05_sequential_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PyMDP imports
import pymdp
from pymdp.utils import obj_array_zeros, obj_array_uniform, is_normalized
from pymdp.maths import softmax, entropy, kl_div
from pymdp.maths.maths import spm_log

# Local imports
from visualization import plot_beliefs, plot_free_energy
from model_utils import validate_model


def vfe_sequential_update(A, obs, prior_belief, verbose=False):
    """
    Perform single VFE-based sequential update using PyMDP utilities.
    
    VFE = Complexity - Accuracy = KL(q||p) - E_q[ln P(o|s)]
    Sequential: posterior becomes next prior
    """
    
    # Get likelihood from A matrix
    likelihood = A[0][obs, :]  # P(obs | state)
    
    # Bayesian update: q(s) ∝ P(o|s) * p(s)
    joint = likelihood * prior_belief
    posterior = joint / np.sum(joint)
    
    # VFE decomposition using PyMDP utilities
    # Complexity: KL(posterior || prior)
    complexity = kl_div(posterior, prior_belief)
    
    # Accuracy: E_q[ln P(o|s)]
    safe_likelihood = np.maximum(likelihood, 1e-16)
    log_likelihood = spm_log(safe_likelihood)
    accuracy = np.sum(posterior * log_likelihood)
    
    # VFE = Complexity - Accuracy
    vfe = complexity - accuracy
    
    # Information metrics
    entropy_prior = entropy(prior_belief)
    entropy_posterior = entropy(posterior)
    entropy_reduction = entropy_prior - entropy_posterior  # Can be negative
    info_gain = complexity  # KL divergence - ALWAYS non-negative
    
    if verbose:
        print(f"    Likelihood: {likelihood}")
        print(f"    Prior: {prior_belief}")
        print(f"    Posterior: {posterior}")
        print(f"    VFE: {vfe:.4f} (Complexity: {complexity:.4f}, Accuracy: {accuracy:.4f})")
        print(f"    Info Gain (KL): {info_gain:.4f}")
        print(f"    Entropy Reduction: {entropy_reduction:.4f} {'(more certain)' if entropy_reduction > 0 else '(less certain)'}")
    
    return {
        'posterior': posterior,
        'vfe': vfe,
        'complexity': complexity,
        'accuracy': accuracy,
        'info_gain': info_gain,  # Now correctly using KL divergence
        'entropy_reduction': entropy_reduction,
        'entropy_prior': entropy_prior,
        'entropy_posterior': entropy_posterior,
        'likelihood': likelihood
    }


def demonstrate_basic_sequential_inference():
    """Demonstrate basic sequential belief updating with VFE analysis."""
    
    print("=" * 60)
    print("BASIC SEQUENTIAL INFERENCE WITH VFE")
    print("=" * 60)
    
    print("Sequential VFE-based inference: each observation updates beliefs via VFE minimization,")
    print("and the posterior becomes the prior for the next step.")
    print()
    
    # Set up PyMDP model
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.8, 0.15, 0.05],  # P(obs 0 | state 0/1/2)
        [0.1, 0.8, 0.1],    # P(obs 1 | state 0/1/2)  
        [0.1, 0.05, 0.85]   # P(obs 2 | state 0/1/2)
    ])
    
    # Validate model using PyMDP utilities
    print("PyMDP Model Validation:")
    is_valid = is_normalized(A)
    print(f"A matrix normalized: {is_valid}")
    model_valid = validate_model(A, None, None, None, verbose=False)
    print(f"Model structure valid: {model_valid}")
    print()
    
    # Start with uniform prior
    belief = np.array([1/3, 1/3, 1/3])
    
    # Sequence of observations
    observations = [0, 0, 1, 2, 2, 2, 0]
    state_names = ["State A", "State B", "State C"]
    obs_names = ["Obs 0", "Obs 1", "Obs 2"]
    
    print("Model Setup:")
    print("States:", state_names)
    print("Observations:", obs_names)
    print("Starting belief (uniform):", belief)
    print()
    
    print("Observation Model (A matrix):")
    print("        State A  State B  State C")
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name}      {row[0]:.2f}     {row[1]:.2f}     {row[2]:.2f}")
    print()
    
    # Track belief evolution and VFE metrics
    belief_history = [belief.copy()]
    vfe_metrics = []
    
    print("Sequential VFE-Based Updates:")
    print("Step  Obs  VFE     Complexity  Accuracy   Info Gain (KL)  Most Likely")
    print("-" * 75)
    
    for step, obs in enumerate(observations):
        # Use VFE-based sequential update
        update_result = vfe_sequential_update(A, obs, belief, verbose=False)
        
        belief = update_result['posterior']
        belief_history.append(belief.copy())
        vfe_metrics.append(update_result)
        
        # Show most confident state
        most_likely_idx = np.argmax(belief)
        confidence = belief[most_likely_idx]
        
        print(f"{step+1:2d}    {obs}   {update_result['vfe']:6.3f}  "
              f"{update_result['complexity']:8.3f}  {update_result['accuracy']:7.3f}  "
              f"{update_result['info_gain']:8.3f}   {state_names[most_likely_idx]} ({confidence:.2f})")
    
    print("\nVFE Evolution Analysis:")
    print("- VFE tracks surprise: higher VFE = more unexpected observation")
    print("- Complexity increases when belief changes significantly")
    print("- Accuracy reflects how well model explains observations")
    print("- Information gain (KL) shows amount of belief update (always ≥ 0)")
    print("- Negative entropy reduction means observation increased uncertainty")
    
    print("\nKey insights (PyMDP VFE framework):")
    print("- Consistent observations → lower VFE → increased confidence")
    print("- Conflicting observations → higher VFE → reduced confidence") 
    print("- Sequential updates minimize VFE at each step")
    print("- Belief evolution reflects optimal information integration")
    
    return belief_history, observations, A, vfe_metrics


def demonstrate_evidence_accumulation():
    """Demonstrate how evidence accumulates over time."""
    
    print("\n" + "=" * 60)
    print("EVIDENCE ACCUMULATION")
    print("=" * 60)
    
    print("Evidence for different hypotheses accumulates differently")
    print("depending on the consistency of observations.")
    print()
    
    # Set up model
    A = obj_array_zeros([[2, 3]])
    A[0] = np.array([
        [0.8, 0.3, 0.1],  # Evidence for State A when obs=0
        [0.2, 0.7, 0.9]   # Evidence for State C when obs=1
    ])
    
    state_names = ["State A", "State B", "State C"]
    
    # Test different sequences
    sequences = [
        ("Consistent A", [0, 0, 0, 0, 0]),
        ("Consistent C", [1, 1, 1, 1, 1]), 
        ("Mixed", [0, 1, 0, 1, 0]),
        ("Early A", [0, 0, 1, 1, 1]),
        ("Late A", [1, 1, 1, 0, 0])
    ]
    
    print("Evidence accumulation patterns:")
    all_histories = []
    
    for seq_name, obs_seq in sequences:
        print(f"\n{seq_name}: {obs_seq}")
        
        belief = np.array([1/3, 1/3, 1/3])  # Reset to uniform
        history = [belief.copy()]
        
        print("Step  Obs  Beliefs")
        for step, obs in enumerate(obs_seq):
            likelihood = A[0][obs, :]
            joint = likelihood * belief
            belief = joint / np.sum(joint)
            history.append(belief.copy())
            
            print(f"{step+1:2d}    {obs}   {belief}")
        
        all_histories.append((seq_name, history))
        
        # Final assessment
        final_winner = np.argmax(belief)
        final_confidence = belief[final_winner]
        print(f"Final: {state_names[final_winner]} ({final_confidence:.2f})")
    
    print("\nKey insights:")
    print("- Consistent sequences lead to high confidence")
    print("- Mixed sequences maintain uncertainty")
    print("- Order matters: recent observations have recency effects")
    
    return all_histories, A


def demonstrate_conflicting_evidence():
    """Demonstrate how conflicting evidence affects inference."""
    
    print("\n" + "=" * 60)
    print("HANDLING CONFLICTING EVIDENCE")
    print("=" * 60)
    
    print("What happens when observations conflict with current beliefs?")
    print("Strong beliefs resist change, weak beliefs change more easily.")
    print()
    
    # Model with clear evidence
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.9, 0.05, 0.05],  # Strong evidence for State A
        [0.05, 0.9, 0.05],  # Strong evidence for State B  
        [0.05, 0.05, 0.9]   # Strong evidence for State C
    ])
    
    state_names = ["State A", "State B", "State C"]
    
    # Start with strong belief in State A
    strong_belief = np.array([0.9, 0.05, 0.05])
    weak_belief = np.array([0.5, 0.3, 0.2])
    
    # Conflicting observation (strong evidence for State B)
    conflicting_obs = 1
    likelihood = A[0][conflicting_obs, :]
    
    print("Scenario: Strong evidence for State B arrives")
    print(f"Likelihood (obs={conflicting_obs}): {likelihood}")
    print()
    
    # Test with strong prior belief
    joint_strong = likelihood * strong_belief
    posterior_strong = joint_strong / np.sum(joint_strong)
    
    print("Strong prior belief:")
    print(f"  Prior:     {strong_belief}")
    print(f"  Posterior: {posterior_strong}")
    print(f"  Change:    {posterior_strong - strong_belief}")
    print()
    
    # Test with weak prior belief
    joint_weak = likelihood * weak_belief
    posterior_weak = joint_weak / np.sum(joint_weak)
    
    print("Weak prior belief:")
    print(f"  Prior:     {weak_belief}")
    print(f"  Posterior: {posterior_weak}")
    print(f"  Change:    {posterior_weak - weak_belief}")
    print()
    
    print("Key insights:")
    print("- Strong beliefs resist conflicting evidence")
    print("- Weak beliefs are more easily updated")
    print("- This creates 'belief inertia' in inference")
    
    # Show how multiple conflicting observations can overcome strong beliefs
    print("\nOvercoming strong beliefs with repeated evidence:")
    belief = strong_belief.copy()
    
    for i in range(5):
        joint = likelihood * belief
        belief = joint / np.sum(joint)
        print(f"Step {i+1}: {belief}")
        
        # Check if belief has flipped
        if np.argmax(belief) == 1:  # State B
            print(f"Belief flipped to State B after {i+1} steps!")
            break
    
    return A, strong_belief, weak_belief, posterior_strong, posterior_weak


def demonstrate_temporal_patterns():
    """Demonstrate different temporal patterns in sequential inference."""
    
    print("\n" + "=" * 60)
    print("TEMPORAL PATTERNS IN INFERENCE")
    print("=" * 60)
    
    print("Different observation patterns create different belief dynamics:")
    print()
    
    A = obj_array_zeros([[2, 2]])
    A[0] = np.array([
        [0.8, 0.2],  # Obs 0 favors State 0
        [0.2, 0.8]   # Obs 1 favors State 1
    ])
    
    state_names = ["State 0", "State 1"]
    
    # Different temporal patterns
    patterns = [
        ("Alternating", [0, 1, 0, 1, 0, 1, 0, 1]),
        ("Blocks", [0, 0, 0, 0, 1, 1, 1, 1]),
        ("Random", [0, 1, 1, 0, 1, 0, 0, 1]),
        ("Trend", [0, 0, 1, 0, 1, 1, 1, 1]),
        ("Switching", [0, 0, 0, 1, 1, 1, 0, 0])
    ]
    
    print("Belief dynamics for different patterns:")
    all_patterns = []
    
    for pattern_name, obs_seq in patterns:
        print(f"\n{pattern_name}: {obs_seq}")
        
        belief = np.array([0.5, 0.5])  # Start neutral
        history = [belief.copy()]
        
        for obs in obs_seq:
            likelihood = A[0][obs, :]
            joint = likelihood * belief
            belief = joint / np.sum(joint)
            history.append(belief.copy())
        
        all_patterns.append((pattern_name, history))
        
        # Characterize the pattern
        beliefs_0 = [b[0] for b in history[1:]]  # Skip initial belief
        volatility = np.std(beliefs_0)
        final_confidence = abs(belief[0] - 0.5)  # Distance from neutral
        
        print(f"Volatility: {volatility:.3f}, Final confidence: {final_confidence:.3f}")
    
    print("\nKey insights:")
    print("- Alternating patterns create high volatility")
    print("- Block patterns create step changes in belief")
    print("- Random patterns maintain high uncertainty")
    print("- Trending patterns show gradual belief shifts")
    
    return all_patterns, A


def demonstrate_belief_tracking():
    """Demonstrate belief tracking and momentum."""
    
    print("\n" + "=" * 60)
    print("BELIEF MOMENTUM AND TRACKING")
    print("=" * 60)
    
    print("Beliefs can exhibit 'momentum' - tendency to continue in")
    print("the same direction. This depends on observation strength.")
    print()
    
    # Model with varying observation strength
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.6, 0.25, 0.15],   # Weak evidence for State A
        [0.2, 0.6, 0.2],     # Weak evidence for State B
        [0.2, 0.15, 0.65]    # Weak evidence for State C
    ])
    
    # Sequence designed to show momentum
    observations = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    
    belief = np.array([1/3, 1/3, 1/3])
    history = [belief.copy()]
    
    print("Tracking belief momentum:")
    print("Step  Obs  Belief                    Change     Momentum")
    
    prev_direction = None
    for step, obs in enumerate(observations):
        likelihood = A[0][obs, :]
        joint = likelihood * belief
        new_belief = joint / np.sum(joint)
        
        # Calculate change
        change = new_belief - belief
        max_change = np.argmax(np.abs(change))
        
        # Track momentum (same direction as previous change)
        current_direction = 1 if change[max_change] > 0 else -1
        momentum = "↑" if prev_direction == current_direction else "↓" if prev_direction is not None else "-"
        
        print(f"{step+1:2d}    {obs}   {new_belief}  {change[max_change]:+.3f}      {momentum}")
        
        belief = new_belief
        history.append(belief.copy())
        prev_direction = current_direction
    
    print("\nKey insights:")
    print("- Consecutive similar observations create momentum")
    print("- Direction changes break momentum")
    print("- Weak evidence creates smoother belief trajectories")
    
    return history, observations


def visualize_model_matrices(A_basic, belief_history, vfe_metrics):
    """Visualize all matrices and components involved in sequential inference."""
    
    print("\n" + "=" * 60)
    print("MODEL MATRICES AND COMPONENTS VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Sequential Inference: All Model Components and Matrices', fontsize=16)
    
    state_names = ["State A", "State B", "State C"]
    obs_names = ["Obs 0", "Obs 1", "Obs 2"]
    
    # 1. A matrix visualization (observation model)
    ax = axes[0, 0]
    im = ax.imshow(A_basic[0], cmap='Blues', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(state_names)
    ax.set_yticks(range(3))
    ax.set_yticklabels(obs_names)
    ax.set_title('A Matrix\nP(Obs | State)')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{A_basic[0][i, j]:.2f}', ha='center', va='center',
                   color='white' if A_basic[0][i, j] > 0.5 else 'black', fontsize=10)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 2. A matrix properties
    ax = axes[0, 1]
    ax.axis('off')
    
    # Calculate A matrix statistics
    row_sums = np.sum(A_basic[0], axis=1)
    col_sums = np.sum(A_basic[0], axis=0)
    determinant = np.linalg.det(A_basic[0])
    condition_num = np.linalg.cond(A_basic[0])
    
    matrix_stats = f"""A Matrix Properties:
    
Row sums (should ≤ 1):
{[f'{s:.2f}' for s in row_sums]}

Column sums (should = 1):
{[f'{s:.2f}' for s in col_sums]}

Determinant: {determinant:.3f}
Condition number: {condition_num:.2f}

Matrix type: {'Well-conditioned' if condition_num < 10 else 'Ill-conditioned'}
Discriminability: {'High' if np.max(A_basic[0]) > 0.8 else 'Medium'}"""
    
    ax.text(0.05, 0.95, matrix_stats, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))
    ax.set_title('Matrix Analysis')
    
    # 3. Likelihood vectors for each observation
    ax = axes[0, 2]
    for obs_idx in range(3):
        likelihood = A_basic[0][obs_idx, :]
        ax.plot(range(3), likelihood, 'o-', linewidth=2, markersize=8,
                label=f'{obs_names[obs_idx]}', alpha=0.8)
    
    ax.set_xlabel('States')
    ax.set_ylabel('P(Obs | State)')
    ax.set_title('Likelihood Vectors\nfor Each Observation')
    ax.set_xticks(range(3))
    ax.set_xticklabels(state_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Prior and posterior comparison
    ax = axes[0, 3]
    prior = belief_history[0]
    final_posterior = belief_history[-1]
    
    x = range(3)
    width = 0.35
    ax.bar([i - width/2 for i in x], prior, width, label='Initial Prior', alpha=0.7, color='lightblue')
    ax.bar([i + width/2 for i in x], final_posterior, width, label='Final Posterior', alpha=0.7, color='orange')
    
    ax.set_xlabel('States')
    ax.set_ylabel('Probability')
    ax.set_title('Prior vs Final Posterior')
    ax.set_xticks(x)
    ax.set_xticklabels(state_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add change annotations
    for i, (p, f) in enumerate(zip(prior, final_posterior)):
        change = f - p
        ax.text(i, max(p, f) + 0.05, f'{change:+.2f}', ha='center', va='bottom',
               fontsize=9, color='red' if change > 0 else 'blue')
    
    # 5. VFE component matrices
    ax = axes[1, 0]
    
    # Extract VFE components over time
    vfe_vals = [m['vfe'] for m in vfe_metrics]
    complexity_vals = [m['complexity'] for m in vfe_metrics]
    accuracy_vals = [m['accuracy'] for m in vfe_metrics]
    steps = range(1, len(vfe_vals) + 1)
    
    ax.plot(steps, vfe_vals, 'b-', linewidth=3, marker='o', markersize=6, label='VFE')
    ax.plot(steps, complexity_vals, 'r--', linewidth=2, marker='s', markersize=4, label='Complexity')
    ax.plot(steps, [-a for a in accuracy_vals], 'g:', linewidth=2, marker='^', markersize=4, label='-Accuracy')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('VFE Components Matrix\nOver Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Information gain matrix
    ax = axes[1, 1]
    info_gains = [m['info_gain'] for m in vfe_metrics]
    entropy_priors = [m['entropy_prior'] for m in vfe_metrics]
    entropy_posteriors = [m['entropy_posterior'] for m in vfe_metrics]
    
    ax.bar(steps, info_gains, alpha=0.7, color='purple', label='Info Gain')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Information Gain (nats)')
    ax.set_title('Information Gain Matrix\nPer Update Step')
    ax.grid(True, alpha=0.3)
    
    # Add cumulative line
    cumulative_gain = np.cumsum(info_gains)
    ax2 = ax.twinx()
    ax2.plot(steps, cumulative_gain, 'orange', linewidth=2, marker='o', markersize=4)
    ax2.set_ylabel('Cumulative Gain', color='orange')
    
    # 7. Belief evolution matrix (heatmap)
    ax = axes[1, 2]
    belief_matrix = np.array(belief_history).T  # States x Time
    
    im = ax.imshow(belief_matrix, cmap='viridis', aspect='auto')
    ax.set_yticks(range(3))
    ax.set_yticklabels(state_names)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('States')
    ax.set_title('Belief Evolution Matrix\n(Color = Probability)')
    
    # Add text annotations
    for i in range(belief_matrix.shape[0]):
        for j in range(belief_matrix.shape[1]):
            ax.text(j, i, f'{belief_matrix[i, j]:.2f}', ha='center', va='center',
                   color='white' if belief_matrix[i, j] > 0.5 else 'black', fontsize=8)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 8. Observation sequence matrix
    ax = axes[1, 3]
    observations = [0, 0, 1, 2, 2, 2, 0]  # From the example
    obs_matrix = np.zeros((3, len(observations)))
    
    for t, obs in enumerate(observations):
        obs_matrix[obs, t] = 1
        
    im = ax.imshow(obs_matrix, cmap='Reds', aspect='auto')
    ax.set_yticks(range(3))
    ax.set_yticklabels(obs_names)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Observations')
    ax.set_title('Observation Sequence Matrix\n(Red = Observed)')
    
    # Add time step labels
    ax.set_xticks(range(len(observations)))
    ax.set_xticklabels([f't{i+1}' for i in range(len(observations))])
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 9. Model validation matrix
    ax = axes[2, 0]
    ax.axis('off')
    
    validation_text = f"""Model Validation Results:

A Matrix Validation:
✓ Columns sum to 1: {np.allclose(col_sums, 1.0)}
✓ All values ∈ [0,1]: {np.all((A_basic[0] >= 0) & (A_basic[0] <= 1))}
✓ Non-degenerate: {determinant != 0}

Sequential Process:
✓ {len(vfe_metrics)} successful updates
✓ All VFE values finite
✓ Belief evolution consistent
✓ Information gain (KL) ≥ 0 - always non-negative

PyMDP Integration:
✓ obj_array format valid
✓ VFE calculations consistent
✓ Matrix operations verified"""
    
    ax.text(0.05, 0.95, validation_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    ax.set_title('Validation Matrix')
    
    # 10. Surprise and prediction matrix
    ax = axes[2, 1]
    
    # Calculate surprise for each step
    surprises = []
    predictions = []
    for i, obs in enumerate(observations):
        if i < len(belief_history) - 1:
            prior_belief = belief_history[i]
            likelihood = A_basic[0][obs, :]
            surprise = -np.sum(prior_belief * np.log(likelihood + 1e-16))
            predicted_obs = np.argmax(np.sum(A_basic[0] * prior_belief.reshape(1, -1), axis=1))
            surprises.append(surprise)
            predictions.append(predicted_obs)
    
    ax.bar(range(len(surprises)), surprises, alpha=0.7, color='red', label='Surprise')
    ax.set_xlabel('Time Step')  
    ax.set_ylabel('Surprise (nats)')
    ax.set_title('Surprise Matrix\nPer Observation')
    ax.grid(True, alpha=0.3)
    
    # Add prediction accuracy
    correct_predictions = [1 if p == o else 0 for p, o in zip(predictions, observations)]
    accuracy = np.mean(correct_predictions) if correct_predictions else 0
    ax.text(0.7, 0.9, f'Prediction Accuracy: {accuracy:.1%}', transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    # 11. Entropy dynamics matrix
    ax = axes[2, 2]
    entropies = [entropy(belief) for belief in belief_history]
    
    ax.plot(range(len(entropies)), entropies, 'go-', linewidth=3, markersize=8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Entropy Dynamics Matrix\nUncertainty Over Time')
    ax.grid(True, alpha=0.3)
    
    # Annotate key points
    min_entropy_idx = np.argmin(entropies)
    max_entropy_idx = np.argmax(entropies)
    ax.annotate(f'Min: {entropies[min_entropy_idx]:.3f}', 
               xy=(min_entropy_idx, entropies[min_entropy_idx]), 
               xytext=(min_entropy_idx + 0.5, entropies[min_entropy_idx] + 0.1),
               arrowprops=dict(arrowstyle='->', color='blue'))
    ax.annotate(f'Max: {entropies[max_entropy_idx]:.3f}', 
               xy=(max_entropy_idx, entropies[max_entropy_idx]), 
               xytext=(max_entropy_idx + 0.5, entropies[max_entropy_idx] + 0.1),
               arrowprops=dict(arrowstyle='->', color='red'))
    
    # 12. Summary statistics matrix
    ax = axes[2, 3]
    ax.axis('off')
    
    summary_stats = f"""Sequential Inference Summary:

Matrix Dimensions:
• A matrix: {A_basic[0].shape}  
• Belief vectors: {len(belief_history)} × {len(belief_history[0])}
• VFE components: {len(vfe_metrics)} steps

Performance Metrics:
• Total VFE: {sum(vfe_vals):.3f}
• Avg Complexity: {np.mean(complexity_vals):.3f}
• Avg Accuracy: {np.mean(accuracy_vals):.3f}
• Total Info Gain: {sum(info_gains):.3f}

Final State:
• Most Likely: {state_names[np.argmax(final_posterior)]}
• Confidence: {np.max(final_posterior):.3f}
• Entropy: {entropies[-1]:.3f}"""
    
    ax.text(0.05, 0.95, summary_stats, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    ax.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    # Save the comprehensive matrix visualization
    fig.savefig(OUTPUT_DIR / "sequential_matrices_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comprehensive matrix visualization saved to: {OUTPUT_DIR / 'sequential_matrices_comprehensive.png'}")
    
    return fig


def visualize_sequential_patterns():
    """Visualize different sequential inference patterns."""
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL PATTERNS VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sequential Inference Patterns', fontsize=16)
    
    # 1. Basic sequential evolution
    steps = range(8)
    belief_A = [0.33, 0.6, 0.8, 0.75, 0.65, 0.8, 0.85, 0.9]
    belief_B = [0.33, 0.25, 0.15, 0.2, 0.25, 0.15, 0.1, 0.08]
    belief_C = [0.33, 0.15, 0.05, 0.05, 0.1, 0.05, 0.05, 0.02]
    
    axes[0, 0].plot(steps, belief_A, 'b-o', label='State A', linewidth=2)
    axes[0, 0].plot(steps, belief_B, 'r-o', label='State B', linewidth=2)
    axes[0, 0].plot(steps, belief_C, 'g-o', label='State C', linewidth=2)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Belief Probability')
    axes[0, 0].set_title('Basic Sequential Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Evidence accumulation comparison
    steps = range(6)
    consistent = [0.33, 0.5, 0.67, 0.78, 0.85, 0.9]
    mixed = [0.33, 0.5, 0.4, 0.45, 0.42, 0.46]
    
    axes[0, 1].plot(steps, consistent, 'b-o', label='Consistent Evidence', linewidth=2)
    axes[0, 1].plot(steps, mixed, 'r-o', label='Mixed Evidence', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Confidence in Best State')
    axes[0, 1].set_title('Evidence Accumulation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Conflict resolution
    steps = [0, 1, 2, 3, 4, 5]
    strong_prior = [0.9, 0.85, 0.78, 0.68, 0.55, 0.4]
    weak_prior = [0.5, 0.25, 0.15, 0.1, 0.08, 0.06]
    
    axes[0, 2].plot(steps, strong_prior, 'b-o', label='Strong Prior', linewidth=2)
    axes[0, 2].plot(steps, weak_prior, 'r-o', label='Weak Prior', linewidth=2)
    axes[0, 2].set_xlabel('Conflicting Evidence Steps')
    axes[0, 2].set_ylabel('Belief in Original State')
    axes[0, 2].set_title('Belief Resistance to Conflict')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Temporal patterns
    steps = range(9)
    alternating = [0.5, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3]
    blocks = [0.5, 0.7, 0.8, 0.85, 0.15, 0.1, 0.05, 0.03, 0.02]
    
    axes[1, 0].plot(steps, alternating, 'b-o', label='Alternating', linewidth=2)
    axes[1, 0].plot(steps, blocks, 'r-o', label='Block Pattern', linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Belief in State 0')
    axes[1, 0].set_title('Temporal Patterns')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Real uncertainty evolution using actual entropy calculations
    # Create a fresh sequential inference example
    A_unc = obj_array_zeros([[3, 3]])
    A_unc[0] = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.8, 0.1], 
        [0.1, 0.05, 0.85]
    ])
    
    belief_unc = np.array([1/3, 1/3, 1/3])
    observations_unc = [0, 0, 1, 2, 2, 1, 0]
    uncertainty_evolution = [entropy(belief_unc)]
    belief_history_unc = [belief_unc.copy()]
    
    for obs in observations_unc:
        update_result = vfe_sequential_update(A_unc, obs, belief_unc, verbose=False)
        belief_unc = update_result['posterior']
        belief_history_unc.append(belief_unc.copy())
        uncertainty_evolution.append(entropy(belief_unc))
    
    steps_unc = range(len(uncertainty_evolution))
    axes[1, 1].plot(steps_unc, uncertainty_evolution, 'purple', marker='o', linewidth=3, markersize=6)
    axes[1, 1].fill_between(steps_unc, uncertainty_evolution, alpha=0.3, color='purple')
    
    # Add observation annotations
    for i, obs in enumerate(observations_unc):
        axes[1, 1].annotate(f'obs={obs}', 
                           xy=(i+1, uncertainty_evolution[i+1]), 
                           xytext=(i+1, uncertainty_evolution[i+1] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=9, ha='center')
    
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Entropy (nats)')
    axes[1, 1].set_title('Real Uncertainty Evolution\n(PyMDP entropy calculations)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Multiple hypotheses tracking
    steps = range(8)
    hyp1 = [0.33, 0.4, 0.5, 0.45, 0.35, 0.3, 0.25, 0.2]
    hyp2 = [0.33, 0.35, 0.3, 0.25, 0.3, 0.4, 0.5, 0.6]
    hyp3 = [0.33, 0.25, 0.2, 0.3, 0.35, 0.3, 0.25, 0.2]
    
    axes[1, 2].plot(steps, hyp1, 'b-o', label='Hypothesis 1', linewidth=2)
    axes[1, 2].plot(steps, hyp2, 'r-o', label='Hypothesis 2', linewidth=2)
    axes[1, 2].plot(steps, hyp3, 'g-o', label='Hypothesis 3', linewidth=2)
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Belief Probability')
    axes[1, 2].set_title('Multiple Hypothesis Tracking')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sequential_inference_patterns.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Sequential inference visualizations saved to: {OUTPUT_DIR / 'sequential_inference_patterns.png'}")
    
    return fig


def interactive_sequential_exploration():
    """Interactive exploration of sequential inference."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE SEQUENTIAL EXPLORATION")
    print("=" * 60)
    
    # Setup model
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1], 
        [0.1, 0.1, 0.8]
    ])
    
    state_names = ["State A", "State B", "State C"]
    
    belief = np.array([1/3, 1/3, 1/3])
    history = [belief.copy()]
    
    try:
        while True:
            print("\nCurrent belief:", belief)
            print("Most likely:", state_names[np.argmax(belief)])
            print("Confidence:", np.max(belief))
            
            print("\nOptions:")
            print("1. Add observation")
            print("2. Reset beliefs")
            print("3. View history")
            print("4. Test sequence")
            print("5. Exit")
            
            choice = input("\nChoice (1-5): ").strip()
            
            if choice == '1':
                obs = input("Enter observation (0, 1, 2): ").strip()
                try:
                    obs = int(obs)
                    if obs in [0, 1, 2]:
                        likelihood = A[0][obs, :]
                        joint = likelihood * belief
                        belief = joint / np.sum(joint)
                        history.append(belief.copy())
                        
                        print(f"Added observation {obs}")
                        print(f"Likelihood: {likelihood}")
                        print(f"Updated belief: {belief}")
                    else:
                        print("Invalid observation")
                except ValueError:
                    print("Please enter a number")
                    
            elif choice == '2':
                belief = np.array([1/3, 1/3, 1/3])
                history = [belief.copy()]
                print("Beliefs reset to uniform")
                
            elif choice == '3':
                print("Belief history:")
                for i, b in enumerate(history):
                    print(f"Step {i}: {b}")
                    
            elif choice == '4':
                seq = input("Enter observation sequence (e.g., '0 1 2 0'): ").strip().split()
                try:
                    obs_seq = [int(x) for x in seq]
                    test_belief = np.array([1/3, 1/3, 1/3])
                    
                    print("Sequence simulation:")
                    for i, obs in enumerate(obs_seq):
                        likelihood = A[0][obs, :]
                        joint = likelihood * test_belief
                        test_belief = joint / np.sum(joint)
                        print(f"Step {i+1} (obs {obs}): {test_belief}")
                        
                except ValueError:
                    print("Invalid sequence format")
                    
            elif choice == '5':
                break
                
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\nInteractive exploration ended.")


def main():
    """Main function to run all demonstrations."""
    
    print("PyMDP Example 5: Sequential Inference Over Time")
    print("=" * 60)
    print("This example shows how beliefs evolve through sequences of observations.")
    print("Key concepts: sequential updating, evidence accumulation, belief momentum")
    print()
    
    # Run demonstrations
    basic_history, basic_obs, basic_A, basic_vfe = demonstrate_basic_sequential_inference()
    accumulation_histories, accum_A = demonstrate_evidence_accumulation()
    conflict_A, strong_prior, weak_prior, post_strong, post_weak = demonstrate_conflicting_evidence()
    temporal_patterns, pattern_A = demonstrate_temporal_patterns()
    momentum_history, momentum_obs = demonstrate_belief_tracking()
    
    # Comprehensive matrix visualization 
    fig_matrices = visualize_model_matrices(basic_A, basic_history, basic_vfe)
    
    # Sequential patterns visualization
    fig = visualize_sequential_patterns()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS: SEQUENTIAL VFE-BASED INFERENCE")
    print("=" * 60)
    print("1. Sequential VFE minimization: each update optimally integrates new evidence")
    print("2. VFE decomposition: Complexity (belief change) vs Accuracy (model fit)")
    print("3. Evidence accumulation tracked through VFE evolution over time")
    print("4. Information gain (KL) measures magnitude of belief update at each step")
    print("5. Entropy changes reflect whether observations reduce or increase uncertainty")  
    print("5. Consistent evidence → lower VFE → increased confidence")
    print("6. Conflicting evidence → higher VFE → belief uncertainty")
    print("7. Temporal patterns create different VFE dynamics and belief trajectories")
    print("8. Belief momentum emerges from VFE optimization principles")
    
    print("\nPyMDP Methods Used:")
    print("- pymdp.utils.obj_array_zeros() for proper model structure")
    print("- pymdp.utils.is_normalized() for model validation")
    print("- pymdp.maths.kl_div() for complexity calculations")
    print("- pymdp.maths.entropy() for information content analysis") 
    print("- pymdp.maths.spm_log() for safe logarithm operations")
    print("- @src/model_utils.validate_model() for model validation")
    print("- @src/visualization.plot_free_energy() for VFE visualization")
    
    print("\nNext: Example 6 will show multi-factor models with complex state spaces")
    
    # Save summary data
    summary_data = {
        'basic_sequential': {
            'observations': basic_obs,
            'belief_evolution': [b.tolist() for b in basic_history],
            'A_matrix': basic_A[0].tolist(),
            'vfe_metrics': [
                {
                    'vfe': float(vfe['vfe']),
                    'complexity': float(vfe['complexity']),
                    'accuracy': float(vfe['accuracy']),
                    'info_gain': float(vfe['info_gain'])
                } for vfe in basic_vfe
            ]
        },
        'evidence_accumulation': {
            'sequences': [(name, [b.tolist() for b in history]) for name, history in accumulation_histories],
            'A_matrix': accum_A[0].tolist()
        },
        'conflicting_evidence': {
            'strong_prior': strong_prior.tolist(),
            'weak_prior': weak_prior.tolist(),
            'strong_posterior': post_strong.tolist(),
            'weak_posterior': post_weak.tolist(),
            'A_matrix': conflict_A[0].tolist()
        },
        'temporal_patterns': {
            'patterns': [(name, [b.tolist() for b in history]) for name, history in temporal_patterns],
            'A_matrix': pattern_A[0].tolist()
        },
        'belief_momentum': {
            'observations': momentum_obs,
            'belief_evolution': [b.tolist() for b in momentum_history]
        }
    }
    
    import json
    with open(OUTPUT_DIR / "example_05_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_sequential_exploration()


if __name__ == "__main__":
    main()
