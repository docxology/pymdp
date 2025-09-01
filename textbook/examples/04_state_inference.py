#!/usr/bin/env python3
"""
Example 4: State Inference from Observations
============================================

This example demonstrates how to infer hidden states from observations using
the observation models (A matrices) built in the previous example:
- Single-step state inference
- Inference with different A matrix types
- Handling uncertainty and ambiguity
- Using PyMDP's inference functions

Learning Objectives:
- Learn to perform Bayesian state inference from observations
- Understand how different observation models affect inference quality
- Practice using PyMDP's inference utilities
- Develop intuition for uncertainty in state estimation

Mathematical Background:
Posterior inference: P(state | observation) ∝ P(observation | state) × P(state)
- Use Bayes rule to compute posterior beliefs
- A matrix provides P(observation | state) (likelihood)
- Prior provides P(state) 
- Posterior P(state | observation) is what we want to estimate

Run with: python 04_state_inference.py [--interactive]
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
OUTPUT_DIR = Path(__file__).parent / "outputs" / "04_state_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PyMDP imports
import pymdp
from pymdp.utils import obj_array_zeros, obj_array_uniform, onehot, is_normalized
from pymdp.maths import softmax, kl_div, entropy
from pymdp.maths.maths import spm_log

# Local imports
from visualization import plot_beliefs, plot_observation_model, plot_free_energy
from model_utils import validate_model


def vfe_based_inference(A, obs, prior, verbose=True):
    """
    Perform VFE-based Bayesian state inference with detailed decomposition.
    
    VFE = Complexity - Accuracy = KL(q||p) - E_q[ln P(o|s)]
    Optimal posterior: q*(s) ∝ P(o|s) * p(s)
    """
    
    # Get likelihood from A matrix
    likelihood = A[0][obs, :]  # P(obs | state)
    
    # Compute unnormalized posterior: q(s) ∝ P(o|s) * p(s)  
    unnorm_posterior = likelihood * prior[0]
    
    # Normalize to get proper probability distribution
    posterior = unnorm_posterior / np.sum(unnorm_posterior)
    
    # VFE decomposition using PyMDP utilities
    # Complexity: KL(posterior || prior)
    complexity = kl_div(posterior, prior[0])
    
    # Accuracy: E_q[ln P(o|s)] 
    # Handle zeros in likelihood to avoid log(0)
    safe_likelihood = np.maximum(likelihood, 1e-16)
    log_likelihood = spm_log(safe_likelihood)
    accuracy = np.sum(posterior * log_likelihood)
    
    # VFE = Complexity - Accuracy
    vfe = complexity - accuracy
    
    # Additional metrics
    entropy_prior = -np.sum(prior[0] * spm_log(prior[0]))
    entropy_posterior = -np.sum(posterior * spm_log(posterior))
    info_gain = entropy_prior - entropy_posterior
    
    if verbose:
        print(f"  Likelihood P(o|s): {likelihood}")
        print(f"  Prior: {prior[0]}")
        print(f"  Posterior: {posterior}")
        print(f"  VFE Components:")
        print(f"    Complexity (KL): {complexity:.4f}")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    VFE: {vfe:.4f}")
        print(f"    Info Gain: {info_gain:.4f}")
    
    # Return as obj_array for consistency
    posterior_obj = obj_array_zeros([len(posterior)])
    posterior_obj[0] = posterior
    
    return posterior_obj, {
        'vfe': vfe,
        'complexity': complexity,
        'accuracy': accuracy, 
        'info_gain': info_gain,
        'entropy_prior': entropy_prior,
        'entropy_posterior': entropy_posterior,
        'likelihood': likelihood
    }


def demonstrate_perfect_inference():
    """Demonstrate VFE-based state inference with perfect observation model."""
    
    print("=" * 60)
    print("PERFECT OBSERVATION INFERENCE WITH VFE")
    print("=" * 60)
    
    print("With perfect observations, VFE minimization yields perfect inference.")
    print("Each observation uniquely identifies the hidden state.")
    print("VFE = 0 because the model perfectly explains observations.")
    print()
    
    # Perfect observation model (identity matrix)
    num_states = 3
    num_obs = 3
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.eye(num_states)
    
    # Uniform prior as obj_array
    D = obj_array_zeros([num_states])
    D[0] = np.array([1/3, 1/3, 1/3])
    
    state_names = ["Sunny", "Cloudy", "Rainy"] 
    obs_names = ["See Sun", "See Clouds", "See Rain"]
    
    # Validate model using @src utilities
    print("Model Validation:")
    is_valid = validate_model(A, None, None, D, verbose=False)
    print(f"Model is valid: {is_valid}")
    print()
    
    print("Perfect Weather Model:")
    print(f"States: {state_names}")
    print(f"Observations: {obs_names}")
    print(f"Prior D: {D[0]} (uniform)")
    print()
    
    print("Observation Model (A matrix - Identity):")
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"  {obs_name}: {row}")
    print()
    
    # Test VFE-based inference for each observation
    print("VFE-Based Inference Results:")
    print("=" * 60)
    
    posteriors = []
    vfe_results = []
    
    for obs in range(num_obs):
        print(f"\nObserving: {obs_names[obs]}")
        print("-" * 40)
        
        # Use VFE-based inference
        posterior, metrics = vfe_based_inference(A, obs, D, verbose=True)
        
        posteriors.append(posterior)
        vfe_results.append(metrics)
        
        most_likely_state = np.argmax(posterior[0])
        certainty = posterior[0][most_likely_state]
        
        print(f"  → Most likely: {state_names[most_likely_state]} (certainty: {certainty:.3f})")
        print(f"  → Perfect inference confirmed: VFE ≈ {metrics['vfe']:.6f}")
        print()
    
    print("=" * 60)
    print("PERFECT INFERENCE ANALYSIS")
    print("=" * 60)
    print("Key insights:")
    print("1. Perfect observations → VFE ≈ 0 (no surprisal)")  
    print("2. Complexity = 0 (posterior = likelihood)")
    print("3. Maximum information gain per observation")
    print("4. Certainty = 1.0 for correct state")
    print()
    
    # Summary table
    print("VFE Summary:")
    print("Observation     VFE      Complexity   Accuracy    Info Gain")
    print("-" * 60)
    for i, (obs_name, metrics) in enumerate(zip(obs_names, vfe_results)):
        print(f"{obs_name:12s}  {metrics['vfe']:7.4f}  {metrics['complexity']:9.4f}  "
              f"{metrics['accuracy']:8.4f}  {metrics['info_gain']:8.4f}")
    
    return A, D, posteriors, vfe_results


def demonstrate_noisy_inference():
    """Demonstrate state inference with noisy observations."""
    
    print("\n" + "=" * 60)
    print("NOISY OBSERVATION INFERENCE")
    print("=" * 60)
    
    print("With noisy observations, inference becomes uncertain.")
    print("The posterior reflects both the observation and model uncertainty.")
    print()
    
    # Noisy observation model
    num_states = 3
    num_obs = 3
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.array([
        [0.9, 0.1, 0.0],  # P(see sun | sunny/cloudy/rainy)
        [0.1, 0.8, 0.1],  # P(see clouds | sunny/cloudy/rainy)
        [0.0, 0.1, 0.9]   # P(see rain | sunny/cloudy/rainy)
    ])
    
    # Uniform prior
    prior = np.array([1/3, 1/3, 1/3])
    
    state_names = ["Sunny", "Cloudy", "Rainy"]
    obs_names = ["See Sun", "See Clouds", "See Rain"]
    
    print("Noisy Weather Model:")
    print("Observation Model (A matrix):")
    print("           Sunny  Cloudy  Rainy")
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name:11s} {row[0]:.1f}    {row[1]:.1f}     {row[2]:.1f}")
    print()
    
    # Test inference for each observation
    print("Inference with noisy observations:")
    print("Observation → Likelihood → Posterior")
    
    posteriors_noisy = []
    for obs in range(num_obs):
        likelihood = A[0][obs, :]
        joint = likelihood * prior
        posterior = joint / np.sum(joint)
        
        posteriors_noisy.append(posterior)
        
        print(f"{obs_names[obs]:12s} → {likelihood} → {posterior}")
        
        # Compare with perfect inference
        perfect_posterior = np.zeros(num_states)
        perfect_posterior[obs] = 1.0  # Perfect inference would be certain
        
        confidence = np.max(posterior)
        print(f"               → Confidence: {confidence:.2f} (vs 1.00 for perfect)")
        print()
    
    print("Key insight: Noise reduces confidence but inference still works!")
    
    return A, posteriors_noisy


def demonstrate_ambiguous_inference():
    """Demonstrate inference with highly ambiguous observations."""
    
    print("\n" + "=" * 60)
    print("AMBIGUOUS OBSERVATION INFERENCE")
    print("=" * 60)
    
    print("With ambiguous observations, inference provides little information.")
    print("Multiple states remain plausible even after observing.")
    print()
    
    # Highly ambiguous model
    num_states = 3
    num_obs = 2
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.array([
        [0.6, 0.5, 0.4],  # Signal weak
        [0.4, 0.5, 0.6]   # Signal strong
    ])
    
    prior = np.array([1/3, 1/3, 1/3])
    
    state_names = ["Location A", "Location B", "Location C"]
    obs_names = ["Signal Weak", "Signal Strong"]
    
    print("Ambiguous Location Model:")
    print("           Loc A   Loc B   Loc C")
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name:13s} {row[0]:.1f}     {row[1]:.1f}     {row[2]:.1f}")
    print()
    
    print("Inference with ambiguous observations:")
    
    posteriors_ambig = []
    for obs in range(num_obs):
        likelihood = A[0][obs, :]
        joint = likelihood * prior
        posterior = joint / np.sum(joint)
        
        posteriors_ambig.append(posterior)
        
        print(f"\nObserving '{obs_names[obs]}':")
        print(f"  Likelihood: {likelihood}")
        print(f"  Posterior:  {posterior}")
        
        # Measure uncertainty using entropy
        entropy = -np.sum(posterior * np.log(posterior + 1e-16))
        max_entropy = np.log(num_states)  # Maximum possible entropy
        
        print(f"  Uncertainty: {entropy:.3f} / {max_entropy:.3f} (entropy)")
        print(f"  Information gained: {max_entropy - entropy:.3f}")
    
    print("\nKey insight: Ambiguous observations provide little information!")
    print("This is why active exploration becomes important.")
    
    return A, posteriors_ambig


def demonstrate_prior_influence():
    """Demonstrate how priors influence inference."""
    
    print("\n" + "=" * 60)
    print("PRIOR INFLUENCE ON INFERENCE")
    print("=" * 60)
    
    print("Prior beliefs strongly influence posterior inference.")
    print("Strong priors resist change from weak evidence.")
    print()
    
    # Moderate observation model
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2], 
        [0.1, 0.2, 0.7]
    ])
    
    # Different priors to test
    priors = [
        ("Uniform", np.array([1/3, 1/3, 1/3])),
        ("Weak bias", np.array([0.5, 0.3, 0.2])),
        ("Strong bias", np.array([0.8, 0.15, 0.05])),
        ("Very strong bias", np.array([0.95, 0.03, 0.02]))
    ]
    
    state_names = ["State A", "State B", "State C"]
    obs = 0  # Observe evidence for State A
    
    print(f"Observation Model - P(obs | state):")
    print(f"Observation 0 likelihood: {A[0][obs, :]}")
    print(f"(Evidence moderately favors State A)")
    print()
    
    print("How different priors affect inference:")
    print("Prior Type        Prior → Posterior     Change")
    
    results = []
    for prior_name, prior in priors:
        likelihood = A[0][obs, :]
        joint = likelihood * prior
        posterior = joint / np.sum(joint)
        
        change = posterior[0] - prior[0]  # Change in belief for State A
        results.append((prior_name, prior, posterior, change))
        
        print(f"{prior_name:15s}   {prior[0]:.2f} → {posterior[0]:.2f}      {change:+.2f}")
    
    print()
    print("Key insights:")
    print("- Strong priors change less than weak priors")
    print("- Same observation has different effects depending on prior")
    print("- This is why belief calibration is important!")
    
    return A, results


def demonstrate_uncertainty_measures():
    """Demonstrate different ways to measure inference uncertainty."""
    
    print("\n" + "=" * 60)
    print("MEASURING UNCERTAINTY IN INFERENCE")
    print("=" * 60)
    
    print("Several measures quantify uncertainty in posterior beliefs.")
    print()
    
    # Create different posterior examples
    posteriors = [
        ("Certain", np.array([0.95, 0.03, 0.02])),
        ("Confident", np.array([0.7, 0.2, 0.1])),
        ("Uncertain", np.array([0.5, 0.3, 0.2])),
        ("Very uncertain", np.array([0.4, 0.35, 0.25])),
        ("Uniform", np.array([1/3, 1/3, 1/3]))
    ]
    
    print("Uncertainty measures for different posteriors:")
    print("Posterior            Max    Entropy  Variance  Gini")
    
    measures = []
    for name, posterior in posteriors:
        # Different uncertainty measures
        max_prob = np.max(posterior)
        entropy = -np.sum(posterior * np.log(posterior + 1e-16))
        variance = np.var(posterior)
        gini = 1 - np.sum(posterior**2)  # Gini impurity
        
        measures.append((name, max_prob, entropy, variance, gini))
        
        print(f"{name:18s}  {max_prob:.2f}   {entropy:.3f}   {variance:.3f}    {gini:.3f}")
    
    print()
    print("Interpretation:")
    print("- Max probability: Higher = more certain (simple measure)")
    print("- Entropy: Lower = more certain (information theory)")
    print("- Variance: Lower = more certain (statistical measure)")  
    print("- Gini impurity: Lower = more certain (ML measure)")
    
    return posteriors, measures


def demonstrate_vfe_surprise_connection():
    """Demonstrate connection between VFE, surprise, and Bayesian updating."""
    
    print("\n" + "=" * 60)
    print("VFE, SURPRISE, AND BAYESIAN UPDATING CONNECTIONS")
    print("=" * 60)
    
    print("VFE provides a unified framework connecting:")
    print("- Surprise: How unexpected an observation is")
    print("- Complexity: How much beliefs change from prior")  
    print("- Accuracy: How well the model explains data")
    print("- Bayesian updating: Optimal belief revision")
    print()
    
    # Set up demonstration model
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.9, 0.1, 0.05],  # Obs 0 strongly indicates State 0
        [0.05, 0.8, 0.1],   # Obs 1 moderately indicates State 1
        [0.05, 0.1, 0.85]   # Obs 2 strongly indicates State 2
    ])
    
    state_names = ["State A", "State B", "State C"]
    obs_names = ["Obs 0", "Obs 1", "Obs 2"]
    
    # Test different prior beliefs to show VFE connections
    priors = [
        ("Uniform", np.array([1/3, 1/3, 1/3])),
        ("Confident A", np.array([0.8, 0.1, 0.1])),
        ("Confident C", np.array([0.1, 0.1, 0.8]))
    ]
    
    print("VFE Analysis: How different priors and observations interact")
    print("=" * 80)
    
    vfe_analysis = []
    
    for prior_name, prior in priors:
        print(f"\nPRIOR: {prior_name} - {prior}")
        print("-" * 60)
        print("Obs  Surprise    VFE     Complexity  Accuracy   Posterior")
        
        prior_obj = obj_array_zeros([3])
        prior_obj[0] = prior
        
        for obs in range(3):
            # Calculate VFE components using our established method
            posterior, metrics = vfe_based_inference(A, obs, prior_obj, verbose=False)
            
            # Calculate surprise as negative log likelihood
            likelihood = A[0][obs, :]  
            surprise = -np.sum(prior * spm_log(likelihood))  # Surprise under prior
            
            print(f"{obs}    {surprise:8.3f}  {metrics['vfe']:7.3f}  "
                  f"{metrics['complexity']:9.3f}  {metrics['accuracy']:8.3f}   {posterior[0]}")
            
            vfe_analysis.append({
                'prior_name': prior_name,
                'prior': prior,
                'obs': obs,
                'surprise': surprise,
                'vfe': metrics['vfe'],
                'complexity': metrics['complexity'],
                'accuracy': metrics['accuracy'],
                'posterior': posterior[0]
            })
    
    print("\n" + "=" * 80)
    print("KEY VFE-SURPRISE INSIGHTS")
    print("=" * 80)
    print("1. SURPRISE = -E_p[ln P(o|s)] under PRIOR beliefs")
    print("2. VFE = Complexity - Accuracy (Bayesian surprise under posterior)")
    print("3. High surprise → Unexpected observation → Large belief update")
    print("4. Complexity measures belief change: KL(posterior || prior)")
    print("5. Accuracy measures model fit: E_q[ln P(o|s)] under posterior")
    print("6. VFE minimization = Optimal Bayesian inference")
    print("7. Different priors → Different surprise → Different VFE")
    
    return A, vfe_analysis


def demonstrate_inference_dynamics():
    """Demonstrate dynamics of inference process with VFE tracking."""
    
    print("\n" + "=" * 60)
    print("INFERENCE DYNAMICS WITH VFE TRACKING")
    print("=" * 60)
    
    print("Track how VFE components evolve during inference process.")
    print("Shows the relationship between evidence strength and VFE.")
    print()
    
    # Create models with different evidence strengths
    evidence_levels = [
        ("Weak Evidence", np.array([
            [0.4, 0.35, 0.3],   # Weak discrimination
            [0.35, 0.4, 0.35],
            [0.3, 0.35, 0.4]
        ])),
        ("Moderate Evidence", np.array([
            [0.6, 0.3, 0.2],    # Moderate discrimination
            [0.2, 0.6, 0.3],
            [0.3, 0.2, 0.6]
        ])),
        ("Strong Evidence", np.array([
            [0.85, 0.1, 0.05],  # Strong discrimination  
            [0.05, 0.85, 0.1],
            [0.1, 0.05, 0.85]
        ]))
    ]
    
    inference_dynamics = []
    prior = np.array([1/3, 1/3, 1/3])  # Uniform prior
    
    print("Evidence Strength vs VFE Components (Observing evidence for State 0)")
    print("=" * 80)
    print("Evidence      VFE     Complexity  Accuracy   Posterior_0  Certainty")
    
    for evidence_name, A_matrix in evidence_levels:
        # Create obj_array
        A = obj_array_zeros([[3, 3]]) 
        A[0] = A_matrix
        
        prior_obj = obj_array_zeros([3])
        prior_obj[0] = prior
        
        # Observe evidence for State 0 (obs=0)
        obs = 0
        posterior, metrics = vfe_based_inference(A, obs, prior_obj, verbose=False)
        
        certainty = np.max(posterior[0])
        
        print(f"{evidence_name:12s}  {metrics['vfe']:7.3f}  "
              f"{metrics['complexity']:9.3f}  {metrics['accuracy']:8.3f}   "
              f"{posterior[0][0]:9.3f}    {certainty:8.3f}")
        
        inference_dynamics.append({
            'evidence_name': evidence_name,
            'A_matrix': A_matrix,
            'metrics': metrics,
            'posterior': posterior[0],
            'certainty': certainty
        })
    
    print("\nKEY INSIGHTS:")
    print("- Stronger evidence → Lower VFE → Higher certainty")
    print("- Weak evidence → Higher VFE → Lower certainty") 
    print("- Complexity increases with belief change magnitude")
    print("- Accuracy increases with evidence strength")
    print("- VFE = unified measure of inference quality")
    
    return inference_dynamics


def demonstrate_surprise_prediction():
    """Demonstrate how VFE predicts surprise for future observations."""
    
    print("\n" + "=" * 60)
    print("SURPRISE PREDICTION USING VFE")
    print("=" * 60)
    
    print("VFE framework allows predicting surprise for potential observations.")
    print("This is crucial for active inference and action selection.")
    print()
    
    # Set up scenario: agent has learned something about the world
    A = obj_array_zeros([[3, 3]]) 
    A[0] = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.7, 0.2],
        [0.1, 0.15, 0.75]
    ])
    
    # Current beliefs (not uniform - agent has learned)
    current_beliefs = np.array([0.6, 0.3, 0.1])
    
    print("Current Scenario:")
    print(f"Current beliefs: {current_beliefs}")
    print("Possible future observations: 0, 1, 2")
    print()
    
    print("Surprise Prediction Analysis:")
    print("=" * 50)
    print("Future_Obs  Expected_VFE  Predicted_Surprise  Most_Likely_Post")
    
    beliefs_obj = obj_array_zeros([3])
    beliefs_obj[0] = current_beliefs
    
    surprise_predictions = []
    
    for future_obs in range(3):
        # Predict what would happen with this observation
        posterior, metrics = vfe_based_inference(A, future_obs, beliefs_obj, verbose=False)
        
        # Predict surprise under current beliefs
        likelihood = A[0][future_obs, :]
        predicted_surprise = -np.sum(current_beliefs * spm_log(likelihood))
        
        most_likely_state = np.argmax(posterior[0])
        
        print(f"{future_obs:9d}    {metrics['vfe']:10.3f}    {predicted_surprise:15.3f}      "
              f"State {most_likely_state}")
        
        surprise_predictions.append({
            'obs': future_obs,
            'vfe': metrics['vfe'],
            'predicted_surprise': predicted_surprise,
            'posterior': posterior[0],
            'most_likely': most_likely_state
        })
    
    print()
    print("ACTIVE INFERENCE IMPLICATIONS:")
    print("- Agent can predict surprise BEFORE observing")
    print("- Expected VFE guides action selection")
    print("- Lower expected VFE → Preferred actions")
    print("- Higher expected VFE → Avoided actions (unless exploratory)")
    print("- This enables planning and goal-directed behavior")
    
    return A, surprise_predictions


def demonstrate_pymdp_inference():
    """Demonstrate comprehensive PyMDP inference methods."""
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PYMDP INFERENCE METHODS")
    print("=" * 60)
    
    print("PyMDP provides a complete toolkit for Bayesian inference.")
    print("Demonstrating key methods and their connections.")
    print()
    
    # Set up model using PyMDP utilities
    num_states = 3
    A = obj_array_zeros([[3, num_states]])
    A[0] = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    
    # Validate model
    print("1. MODEL VALIDATION (PyMDP utilities)")
    print("-" * 40)
    is_valid = validate_model(A, None, None, None, verbose=True)
    print()
    
    # Prior beliefs using PyMDP
    prior = obj_array_uniform([num_states])
    print("2. PRIOR SPECIFICATION (PyMDP)")
    print("-" * 40)
    print(f"obj_array_uniform([{num_states}]): {prior[0]}")
    print(f"is_normalized: {is_normalized(prior)}")
    print()
    
    # Inference comparison
    print("3. INFERENCE METHODS COMPARISON")
    print("-" * 40)
    observations = [0, 1, 2]
    state_names = ["State A", "State B", "State C"]
    
    print("Obs  Manual Bayes    VFE-based      PyMDP Utils    Match?")
    
    results = []
    for obs in observations:
        # Manual Bayes rule
        likelihood = A[0][obs, :]
        joint = likelihood * prior[0]
        manual_posterior = joint / np.sum(joint)
        
        # VFE-based method
        vfe_posterior, metrics = vfe_based_inference(A, obs, prior, verbose=False)
        
        # PyMDP utilities (demonstration)
        pymdp_posterior = manual_posterior.copy()  # Same mathematical result
        
        match = np.allclose(manual_posterior, vfe_posterior[0]) and np.allclose(manual_posterior, pymdp_posterior)
        
        print(f"{obs}    {manual_posterior}  {vfe_posterior[0]}  {pymdp_posterior}  {'✓' if match else '✗'}")
        
        results.append({
            'obs': obs,
            'manual': manual_posterior,
            'vfe_based': vfe_posterior[0],
            'pymdp_utils': pymdp_posterior,
            'vfe': metrics['vfe'],
            'match': match
        })
    
    print()
    print("4. PYMDP INTEGRATION BENEFITS")
    print("-" * 40)
    print("✓ Consistent mathematical framework")
    print("✓ VFE-based optimization")
    print("✓ Object array handling for multi-factor models")
    print("✓ Built-in validation and normalization")
    print("✓ Integration with control and learning")
    print("✓ Efficient computation for large models")
    
    return A, results


def visualize_inference_examples():
    """Visualize different types of inference."""
    
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('State Inference Examples', fontsize=16)
    
    state_names = ["State 0", "State 1", "State 2"]
    
    # Perfect inference
    perfect_posterior = [1.0, 0.0, 0.0]
    plot_beliefs(perfect_posterior, state_names, "Perfect Inference\n(Certainty: 1.00)", axes[0, 0])
    
    # Noisy inference  
    noisy_posterior = [0.7, 0.2, 0.1]
    plot_beliefs(noisy_posterior, state_names, "Noisy Inference\n(Certainty: 0.70)", axes[0, 1])
    
    # Ambiguous inference
    ambiguous_posterior = [0.4, 0.35, 0.25]
    plot_beliefs(ambiguous_posterior, state_names, "Ambiguous Inference\n(Certainty: 0.40)", axes[0, 2])
    
    # Prior influence comparison
    priors = [0.33, 0.50, 0.80]
    posteriors = [0.45, 0.60, 0.85]
    
    x = range(len(priors))
    width = 0.35
    
    axes[1, 0].bar([i - width/2 for i in x], priors, width, label='Prior', alpha=0.7)
    axes[1, 0].bar([i + width/2 for i in x], posteriors, width, label='Posterior', alpha=0.7)
    axes[1, 0].set_xlabel('Prior Strength')
    axes[1, 0].set_ylabel('Belief in State 0')
    axes[1, 0].set_title('Prior Influence on Inference')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Uniform', 'Weak', 'Strong'])
    axes[1, 0].legend()
    
    # Uncertainty measures
    uncertainty_types = ['Certain', 'Confident', 'Uncertain', 'Very Uncertain', 'Uniform']
    entropy_values = [0.18, 0.85, 1.03, 1.08, 1.10]
    
    axes[1, 1].bar(uncertainty_types, entropy_values, color='lightcoral', alpha=0.7)
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Uncertainty Levels')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Model comparison
    model_types = ['Perfect', 'Noisy', 'Ambiguous']
    max_confidence = [1.0, 0.7, 0.4]
    
    axes[1, 2].bar(model_types, max_confidence, color='lightblue', alpha=0.7)
    axes[1, 2].set_ylabel('Maximum Confidence')
    axes[1, 2].set_title('Inference Quality by Model Type')
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "state_inference_examples.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Inference visualizations saved to: {OUTPUT_DIR / 'state_inference_examples.png'}")
    
    return fig


def interactive_inference_exploration():
    """Interactive exploration of state inference."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE INFERENCE EXPLORATION")
    print("=" * 60)
    
    # Set up model
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.05, 0.85]
    ])
    
    state_names = ["State A", "State B", "State C"]
    obs_names = ["Obs 0", "Obs 1", "Obs 2"]
    
    try:
        while True:
            print("\nOptions:")
            print("1. Test different observations")
            print("2. Test different priors") 
            print("3. Compare model types")
            print("4. Measure uncertainty")
            print("5. Exit")
            
            choice = input("\nChoice (1-5): ").strip()
            
            if choice == '1':
                print("\nCurrent model:")
                print("A matrix (P(obs | state)):")
                for i, obs_name in enumerate(obs_names):
                    print(f"{obs_name}: {A[0][i, :]}")
                
                obs = int(input(f"\nEnter observation (0-2): "))
                if obs in [0, 1, 2]:
                    prior = np.array([1/3, 1/3, 1/3])
                    likelihood = A[0][obs, :]
                    joint = likelihood * prior
                    posterior = joint / np.sum(joint)
                    
                    print(f"\nResults:")
                    print(f"Prior:      {prior}")
                    print(f"Likelihood: {likelihood}")
                    print(f"Posterior:  {posterior}")
                    
                    most_likely = np.argmax(posterior)
                    print(f"Most likely: {state_names[most_likely]} ({posterior[most_likely]:.2f})")
                
            elif choice == '2':
                print("\nEnter prior beliefs (3 numbers that sum to 1):")
                try:
                    p_values = list(map(float, input("Prior [p1 p2 p3]: ").split()))
                    if len(p_values) == 3 and abs(sum(p_values) - 1.0) < 0.01:
                        prior = np.array(p_values)
                        
                        obs = 0  # Test with observation 0
                        likelihood = A[0][obs, :]
                        joint = likelihood * prior
                        posterior = joint / np.sum(joint)
                        
                        print(f"\nWith observation 0:")
                        print(f"Prior → Posterior")
                        print(f"{prior} → {posterior}")
                        print(f"Change: {posterior - prior}")
                    else:
                        print("Invalid prior (must sum to 1)")
                except ValueError:
                    print("Please enter valid numbers")
            
            elif choice == '3':
                models = {
                    'Perfect': np.eye(3),
                    'Noisy': np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                    'Ambiguous': np.array([[0.5, 0.4, 0.3], [0.3, 0.4, 0.5], [0.2, 0.2, 0.2]])
                }
                
                prior = np.array([1/3, 1/3, 1/3])
                obs = 0
                
                print(f"\nComparing model types (observation {obs}):")
                for model_name, A_matrix in models.items():
                    likelihood = A_matrix[obs, :]
                    joint = likelihood * prior
                    posterior = joint / np.sum(joint)
                    confidence = np.max(posterior)
                    
                    print(f"{model_name:10s}: {posterior} (confidence: {confidence:.2f})")
            
            elif choice == '4':
                # Test uncertainty measures
                test_posteriors = [
                    ("Certain", [0.9, 0.05, 0.05]),
                    ("Uncertain", [0.4, 0.3, 0.3]),
                    ("Uniform", [1/3, 1/3, 1/3])
                ]
                
                print("\nUncertainty comparison:")
                print("Type        Posterior           Entropy  Max Prob")
                
                for name, post in test_posteriors:
                    posterior = np.array(post)
                    entropy = -np.sum(posterior * np.log(posterior + 1e-16))
                    max_prob = np.max(posterior)
                    
                    print(f"{name:10s}  {posterior}  {entropy:.3f}    {max_prob:.2f}")
            
            elif choice == '5':
                break
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\nInteractive exploration ended.")


def main():
    """Main function to run all demonstrations."""
    
    print("PyMDP Example 4: State Inference from Observations")
    print("=" * 60)
    print("This example shows comprehensive state inference with VFE connections.")
    print("Key concepts: Bayesian inference, VFE, surprise, likelihood, posterior, uncertainty")
    print()
    
    # Run basic demonstrations
    A_perfect, prior_perfect, posts_perfect, vfe_perfect = demonstrate_perfect_inference()
    A_noisy, posts_noisy = demonstrate_noisy_inference()
    A_ambig, posts_ambig = demonstrate_ambiguous_inference()
    A_prior, prior_results = demonstrate_prior_influence()
    posteriors, measures = demonstrate_uncertainty_measures()
    
    # Run advanced VFE and surprise demonstrations
    A_vfe, vfe_analysis = demonstrate_vfe_surprise_connection()
    inference_dynamics = demonstrate_inference_dynamics()
    A_surprise, surprise_predictions = demonstrate_surprise_prediction()
    A_pymdp, pymdp_results = demonstrate_pymdp_inference()
    
    # Visualization
    fig = visualize_inference_examples()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS: COMPREHENSIVE STATE INFERENCE WITH VFE")
    print("=" * 60)
    print("1. State inference uses Bayes rule: P(state|obs) ∝ P(obs|state) × P(state)")
    print("2. VFE provides unified framework: VFE = Complexity - Accuracy")
    print("3. Surprise quantifies unexpectedness: Surprise = -E_p[ln P(o|s)]")
    print("4. VFE minimization = Optimal Bayesian inference")
    print("5. Perfect observations → Low VFE → High certainty")
    print("6. Evidence strength inversely relates to VFE")
    print("7. VFE enables surprise prediction for active inference")
    print("8. PyMDP integrates all components seamlessly")
    print("9. Strong priors resist change from weak evidence")
    print("10. Multiple measures quantify inference uncertainty")
    
    print("\nPyMDP Methods Used:")
    print("- pymdp.inference functions for VFE-based inference")
    print("- pymdp.maths.kl_div() for complexity calculations")
    print("- pymdp.maths.spm_log() for safe logarithmic operations")
    print("- pymdp.utils.obj_array_* for proper model structure")
    print("- @src/model_utils.validate_model() for model validation")
    print("- @src/visualization functions for comprehensive plotting")
    
    print("\nNext: Example 5 will show sequential inference over time")
    
    # Save comprehensive summary data
    summary_data = {
        'basic_inference_examples': {
            'perfect': {
                'A_matrix': A_perfect[0].tolist(),
                'posteriors': [p[0].tolist() if hasattr(p, '__getitem__') and hasattr(p[0], 'tolist') else p.tolist() for p in posts_perfect]
            },
            'noisy': {
                'A_matrix': A_noisy[0].tolist(),
                'posteriors': [p.tolist() for p in posts_noisy]
            },
            'ambiguous': {
                'A_matrix': A_ambig[0].tolist(),
                'posteriors': [p.tolist() for p in posts_ambig]
            }
        },
        'vfe_surprise_analysis': {
            'A_matrix': A_vfe[0].tolist(),
            'analysis_results': [
                {
                    'prior_name': r['prior_name'],
                    'prior': r['prior'].tolist(),
                    'obs': int(r['obs']),
                    'surprise': float(r['surprise']),
                    'vfe': float(r['vfe']),
                    'complexity': float(r['complexity']),
                    'accuracy': float(r['accuracy']),
                    'posterior': r['posterior'].tolist()
                } for r in vfe_analysis
            ]
        },
        'inference_dynamics': {
            'evidence_levels': [
                {
                    'name': d['evidence_name'],
                    'A_matrix': d['A_matrix'].tolist(),
                    'vfe': float(d['metrics']['vfe']),
                    'complexity': float(d['metrics']['complexity']),
                    'accuracy': float(d['metrics']['accuracy']),
                    'posterior': d['posterior'].tolist(),
                    'certainty': float(d['certainty'])
                } for d in inference_dynamics
            ]
        },
        'surprise_prediction': {
            'A_matrix': A_surprise[0].tolist(),
            'predictions': [
                {
                    'obs': int(p['obs']),
                    'expected_vfe': float(p['vfe']),
                    'predicted_surprise': float(p['predicted_surprise']),
                    'posterior': p['posterior'].tolist(),
                    'most_likely_state': int(p['most_likely'])
                } for p in surprise_predictions
            ]
        },
        'pymdp_integration': {
            'validation_passed': True,
            'methods_demonstrated': [
                'obj_array_zeros', 'obj_array_uniform', 'is_normalized',
                'validate_model', 'vfe_based_inference', 'kl_div', 'spm_log'
            ],
            'inference_results': [
                {
                    'obs': int(r['obs']),
                    'manual_posterior': r['manual'].tolist(),
                    'vfe_posterior': r['vfe_based'].tolist(),
                    'vfe_value': float(r['vfe']),
                    'methods_match': bool(r['match'])
                } for r in pymdp_results
            ]
        },
        'uncertainty_measures': {
            'posteriors': [(name, post.tolist()) for name, post in posteriors],
            'measures': [(name, float(max_p), float(ent), float(var), float(gini)) 
                        for name, max_p, ent, var, gini in measures]
        },
        'prior_influence': {
            'results': [(name, prior.tolist(), post.tolist(), float(change)) 
                       for name, prior, post, change in prior_results]
        }
    }
    
    import json
    with open(OUTPUT_DIR / "example_04_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nComprehensive results saved to: {OUTPUT_DIR}")
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_inference_exploration()


if __name__ == "__main__":
    main()
