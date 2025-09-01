#!/usr/bin/env python3
"""
Example 6: Multi-Factor Models
==============================

This example demonstrates models with multiple state factors, where the
hidden state space has multiple independent or semi-independent dimensions:
- Factorized state spaces
- Multi-factor observation models
- Joint vs independent inference
- Handling state space complexity

Learning Objectives:
- Understand factorized state representations
- Learn to build multi-factor observation models
- Practice inference in complex state spaces
- Develop intuition for state space dimensionality

Mathematical Background:
Factorized states: s = (s1, s2, ..., sF) where each si is a factor
Joint observations: P(o | s1, s2, ...) may depend on multiple factors
Inference: P(s1, s2 | o) = P(s1 | o, s2) P(s2 | o, s1) (if dependent)
                        = P(s1 | o) P(s2 | o) (if independent)

Run with: python 06_multi_factor_models.py [--interactive]
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
OUTPUT_DIR = Path(__file__).parent / "outputs" / "06_multi_factor_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PyMDP imports
import pymdp
from pymdp.utils import obj_array_zeros, obj_array_uniform
from pymdp.maths import softmax

# Local imports
from visualization import plot_beliefs


def demonstrate_factorized_states():
    """Demonstrate basic factorized state representation."""
    
    print("=" * 60)
    print("FACTORIZED STATE SPACES")
    print("=" * 60)
    
    print("Complex environments often have multiple independent aspects.")
    print("Example: Robot location (3 rooms) × Time of day (2 periods)")
    print()
    
    # Define factors
    locations = ["Kitchen", "Living Room", "Bedroom"]
    times = ["Day", "Night"]
    
    # Joint state space
    joint_states = []
    for loc in locations:
        for time in times:
            joint_states.append(f"{loc}-{time}")
    
    print("Factor 1 - Location:")
    for i, loc in enumerate(locations):
        print(f"  {i}: {loc}")
    
    print("\nFactor 2 - Time:")
    for i, time in enumerate(times):
        print(f"  {i}: {time}")
    
    print(f"\nJoint State Space ({len(locations)} × {len(times)} = {len(joint_states)} states):")
    for i, state in enumerate(joint_states):
        print(f"  {i}: {state}")
    
    print("\nKey concepts:")
    print("- Factorized representation is more efficient")
    print("- Can model dependencies between factors")
    print("- Enables structured inference")
    
    return locations, times, joint_states


def demonstrate_independent_factors():
    """Demonstrate inference with independent factors."""
    
    print("\n" + "=" * 60)
    print("INDEPENDENT FACTOR INFERENCE")
    print("=" * 60)
    
    print("When factors are independent, we can infer them separately.")
    print("This is computationally efficient and often realistic.")
    print()
    
    # Independent factors: Location and Weather
    num_locations = 3  # Home, Work, Store
    num_weather = 2    # Sunny, Rainy
    
    # Separate observation models for each factor
    # Location observations: GPS signal
    A_location = obj_array_zeros([[3, num_locations]])  # 3 GPS readings, 3 locations
    A_location[0] = np.array([
        [0.8, 0.1, 0.1],  # GPS reading 0 → Home likely
        [0.1, 0.8, 0.1],  # GPS reading 1 → Work likely
        [0.1, 0.1, 0.8]   # GPS reading 2 → Store likely
    ])
    
    # Weather observations: Temperature sensor
    A_weather = obj_array_zeros([[2, num_weather]])  # 2 temp readings, 2 weather states
    A_weather[0] = np.array([
        [0.2, 0.8],  # Cold reading → Rainy likely  
        [0.8, 0.2]   # Warm reading → Sunny likely
    ])
    
    # Independent priors
    prior_location = np.array([0.5, 0.3, 0.2])  # Usually home
    prior_weather = np.array([0.7, 0.3])        # Usually sunny
    
    location_names = ["Home", "Work", "Store"]
    weather_names = ["Sunny", "Rainy"]
    
    print("Independent Factor Models:")
    print("\nLocation Model (GPS):")
    print("        Home  Work  Store")
    gps_names = ["GPS 0", "GPS 1", "GPS 2"]
    for i, name in enumerate(gps_names):
        row = A_location[0][i, :]
        print(f"{name}  {row[0]:.1f}   {row[1]:.1f}   {row[2]:.1f}")
    
    print("\nWeather Model (Temperature):")
    print("        Sunny  Rainy")
    temp_names = ["Cold", "Warm"]
    for i, name in enumerate(temp_names):
        row = A_weather[0][i, :]
        print(f"{name}   {row[0]:.1f}    {row[1]:.1f}")
    
    print(f"\nPriors:")
    print(f"Location: {prior_location} → {location_names}")
    print(f"Weather:  {prior_weather} → {weather_names}")
    
    # Test independent inference
    print("\nIndependent Inference Example:")
    print("Observations: GPS=1 (Work signal), Temperature=Cold")
    
    # Location inference
    location_obs = 1  # GPS reading for work
    location_likelihood = A_location[0][location_obs, :]
    location_joint = location_likelihood * prior_location
    location_posterior = location_joint / np.sum(location_joint)
    
    # Weather inference
    weather_obs = 0  # Cold temperature
    weather_likelihood = A_weather[0][weather_obs, :]
    weather_joint = weather_likelihood * prior_weather
    weather_posterior = weather_joint / np.sum(weather_joint)
    
    print(f"\nLocation inference:")
    print(f"  Likelihood: {location_likelihood}")
    print(f"  Posterior:  {location_posterior}")
    print(f"  → Most likely: {location_names[np.argmax(location_posterior)]}")
    
    print(f"\nWeather inference:")
    print(f"  Likelihood: {weather_likelihood}")
    print(f"  Posterior:  {weather_posterior}")
    print(f"  → Most likely: {weather_names[np.argmax(weather_posterior)]}")
    
    print("\nKey insight: Factors inferred independently, very efficient!")
    
    return A_location, A_weather, location_posterior, weather_posterior


def demonstrate_dependent_factors():
    """Demonstrate inference with dependent factors."""
    
    print("\n" + "=" * 60)
    print("DEPENDENT FACTOR INFERENCE")
    print("=" * 60)
    
    print("When factors interact, observations may depend on multiple factors.")
    print("This requires joint inference but provides richer models.")
    print()
    
    # Dependent factors: Location × Time → Activity level
    num_locations = 2  # Home, Work
    num_times = 2      # Day, Night
    num_activities = 3 # Low, Medium, High
    
    # Joint observation model: Activity depends on both location and time
    # State ordering: [(Home,Day), (Work,Day), (Home,Night), (Work,Night)]
    A_joint = obj_array_zeros([[num_activities, num_locations * num_times]])
    
    # Activity patterns:
    # Home-Day: Medium activity (home chores)
    # Work-Day: High activity (work tasks)
    # Home-Night: Low activity (relaxing)
    # Work-Night: Low activity (minimal night shift)
    A_joint[0] = np.array([
        [0.1, 0.1, 0.8, 0.7],  # Low activity
        [0.7, 0.2, 0.2, 0.3],  # Medium activity
        [0.2, 0.7, 0.0, 0.0]   # High activity
    ])
    
    # Joint prior: P(location, time)
    joint_prior = np.array([0.3, 0.2, 0.4, 0.1])  # Mostly home, prefer day
    
    state_names = ["Home-Day", "Work-Day", "Home-Night", "Work-Night"]
    activity_names = ["Low Activity", "Medium Activity", "High Activity"]
    
    print("Dependent Factor Model:")
    print("Joint states:", state_names)
    print("\nActivity Model (depends on Location × Time):")
    print("              Home-Day  Work-Day  Home-Night  Work-Night")
    
    for i, activity in enumerate(activity_names):
        row = A_joint[0][i, :]
        print(f"{activity:13s}    {row[0]:.1f}      {row[1]:.1f}       {row[2]:.1f}        {row[3]:.1f}")
    
    print(f"\nJoint prior: {joint_prior}")
    print(f"            {state_names}")
    
    # Test dependent inference
    print("\nDependent Inference Example:")
    print("Observation: High Activity")
    
    activity_obs = 2  # High activity
    likelihood = A_joint[0][activity_obs, :]
    joint_posterior_unnorm = likelihood * joint_prior
    joint_posterior = joint_posterior_unnorm / np.sum(joint_posterior_unnorm)
    
    print(f"\nJoint inference:")
    print(f"  Likelihood: {likelihood}")
    print(f"  Joint posterior: {joint_posterior}")
    
    # Marginal inference: What's the probability of each location?
    location_marginal = np.array([
        joint_posterior[0] + joint_posterior[2],  # Home (day + night)
        joint_posterior[1] + joint_posterior[3]   # Work (day + night)
    ])
    
    time_marginal = np.array([
        joint_posterior[0] + joint_posterior[1],  # Day (home + work)
        joint_posterior[2] + joint_posterior[3]   # Night (home + work)
    ])
    
    print(f"\nMarginal posteriors:")
    print(f"  Location: {location_marginal} → ['Home', 'Work']")
    print(f"  Time: {time_marginal} → ['Day', 'Night']")
    
    most_likely_joint = np.argmax(joint_posterior)
    print(f"\nMost likely joint state: {state_names[most_likely_joint]}")
    
    print("\nKey insight: Joint inference captures factor interactions!")
    
    return A_joint, joint_posterior, location_marginal, time_marginal


def demonstrate_hierarchical_factors():
    """Demonstrate hierarchical factor models."""
    
    print("\n" + "=" * 60)
    print("HIERARCHICAL FACTOR MODELS")
    print("=" * 60)
    
    print("Factors can be organized hierarchically: higher-level factors")
    print("influence lower-level factors in a structured way.")
    print()
    
    # Hierarchical example: Context → Task → Action
    contexts = ["Work Mode", "Home Mode"]
    tasks = ["Focus", "Social", "Rest"]  
    actions = ["Type", "Talk", "Read", "Sleep"]
    
    print("Hierarchical Structure:")
    print("Context (Level 2):", contexts)
    print("Task (Level 1):", tasks)  
    print("Action (Level 0):", actions)
    print()
    
    # Context influences task probabilities
    # P(task | context)
    P_task_context = np.array([
        [0.7, 0.1],  # Focus: likely at work, unlikely at home
        [0.2, 0.4],  # Social: possible everywhere
        [0.1, 0.5]   # Rest: unlikely at work, likely at home
    ])
    
    # Task influences action probabilities  
    # P(action | task) - rows are actions, columns are tasks
    P_action_task = np.array([
        [0.8, 0.1, 0.1],  # Type: focus→likely, social→unlikely, rest→unlikely
        [0.1, 0.7, 0.1],  # Talk: focus→unlikely, social→likely, rest→unlikely
        [0.1, 0.1, 0.3],  # Read: possible for all but more for rest
        [0.0, 0.1, 0.5]   # Sleep: mainly during rest
    ])
    
    print("Context → Task probabilities:")
    print("         Work  Home")
    for i, task in enumerate(tasks):
        row = P_task_context[i, :]
        print(f"{task:7s}   {row[0]:.1f}   {row[1]:.1f}")
    
    print("\nTask → Action probabilities:")
    print("         Focus  Social  Rest")
    for i, action in enumerate(actions):
        row = P_action_task[i, :]
        print(f"{action:5s}     {row[0]:.1f}    {row[1]:.1f}   {row[2]:.1f}")
    
    # Hierarchical inference example
    print("\nHierarchical Inference Example:")
    print("Observation: Agent is typing")
    
    observed_action = 0  # Typing
    
    # Bottom-up inference: Action → Task
    # P_action_task is (actions x tasks), so P(task | action) ∝ P(action | task) * P(task)
    task_prior = np.array([0.33, 0.33, 0.34])  # Uniform task prior
    action_likelihoods = P_action_task[observed_action, :]  # P(type | task) for each task
    task_posterior_unnorm = action_likelihoods * task_prior
    task_posterior = task_posterior_unnorm / np.sum(task_posterior_unnorm)
    
    print(f"\nStep 1: Action → Task")
    print(f"  P(type | task): {action_likelihoods}")
    print(f"  Task posterior: {task_posterior}")
    print(f"  → Most likely task: {tasks[np.argmax(task_posterior)]}")
    
    # Top-down inference: Task → Context  
    # P(context | task) ∝ P(task | context) × P(context)
    context_prior = np.array([0.5, 0.5])  # Uniform context prior
    
    # For each context, compute P(context | task_posterior)
    context_posteriors = []
    for ctx in range(len(contexts)):
        # Weight by how much this context predicts the inferred tasks
        context_score = np.sum(task_posterior * P_task_context[:, ctx])
        context_posteriors.append(context_score)
    
    context_posterior = np.array(context_posteriors)
    context_posterior = context_posterior / np.sum(context_posterior)
    
    print(f"\nStep 2: Task → Context")
    print(f"  Context posterior: {context_posterior}")
    print(f"  → Most likely context: {contexts[np.argmax(context_posterior)]}")
    
    print("\nKey insight: Hierarchy enables rich structured inference!")
    
    return P_task_context, P_action_task, task_posterior, context_posterior


def visualize_multi_factor_models():
    """Visualize multi-factor model concepts."""
    
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Factor Model Examples', fontsize=16)
    
    # 1. Factor independence
    location_beliefs = [0.7, 0.2, 0.1]
    weather_beliefs = [0.8, 0.2]
    
    plot_beliefs(location_beliefs, ["Home", "Work", "Store"], 
                "Independent Factor: Location", axes[0, 0])
    
    ax = axes[0, 1]
    ax.bar(["Sunny", "Rainy"], weather_beliefs, color=['gold', 'blue'], alpha=0.7)
    ax.set_title("Independent Factor: Weather")
    ax.set_ylabel("Belief Probability")
    
    # 2. Joint state beliefs
    joint_beliefs = [0.3, 0.5, 0.15, 0.05]
    joint_states = ["Home-Day", "Work-Day", "Home-Night", "Work-Night"]
    
    ax = axes[1, 0]
    ax.bar(joint_states, joint_beliefs, color='lightcoral', alpha=0.7)
    ax.set_title("Joint State Beliefs")
    ax.set_ylabel("Belief Probability")
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Hierarchical structure
    hierarchy_levels = ["Context", "Task", "Action"]
    level_uncertainties = [0.3, 0.6, 0.9]
    
    ax = axes[1, 1]
    ax.bar(hierarchy_levels, level_uncertainties, color='lightgreen', alpha=0.7)
    ax.set_title("Hierarchical Uncertainty")
    ax.set_ylabel("Uncertainty (Entropy)")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "multi_factor_examples.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Multi-factor visualizations saved to: {OUTPUT_DIR / 'multi_factor_examples.png'}")
    
    return fig


def main():
    """Main function to run all demonstrations."""
    
    print("PyMDP Example 6: Multi-Factor Models")
    print("=" * 60)
    print("This example shows how to work with complex, factorized state spaces.")
    print("Key concepts: factor independence, joint inference, hierarchical structure")
    print()
    
    # Run demonstrations
    locations, times, joint_states = demonstrate_factorized_states()
    A_loc, A_weather, loc_post, weather_post = demonstrate_independent_factors()
    A_joint, joint_post, loc_marg, time_marg = demonstrate_dependent_factors()
    P_task_ctx, P_act_task, task_post, ctx_post = demonstrate_hierarchical_factors()
    
    # Visualization
    fig = visualize_multi_factor_models()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Factorized states decompose complex environments into manageable factors")
    print("2. Independent factors can be inferred separately (efficient)")
    print("3. Dependent factors require joint inference (more complex but richer)")
    print("4. Hierarchical factors enable structured, multi-level reasoning")
    print("5. Marginal inference extracts single-factor beliefs from joint posteriors")
    print("6. Factor interactions capture realistic environmental dependencies")
    
    print("\nNext: Example 7 will show how to build transition models (B matrices)")
    
    # Save summary data
    summary_data = {
        'factorized_representation': {
            'factors': [locations, times],
            'joint_states': joint_states
        },
        'independent_factors': {
            'location_A_matrix': A_loc[0].tolist(),
            'weather_A_matrix': A_weather[0].tolist(),
            'location_posterior': loc_post.tolist(),
            'weather_posterior': weather_post.tolist()
        },
        'dependent_factors': {
            'joint_A_matrix': A_joint[0].tolist(),
            'joint_posterior': joint_post.tolist(),
            'location_marginal': loc_marg.tolist(),
            'time_marginal': time_marg.tolist()
        },
        'hierarchical_factors': {
            'task_given_context': P_task_ctx.tolist(),
            'action_given_task': P_act_task.tolist(),
            'task_posterior': task_post.tolist(),
            'context_posterior': ctx_post.tolist()
        }
    }
    
    import json
    with open(OUTPUT_DIR / "example_06_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("Interactive mode not implemented for this example")


if __name__ == "__main__":
    main()
