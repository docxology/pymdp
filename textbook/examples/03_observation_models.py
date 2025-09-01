#!/usr/bin/env python3
"""
Example 3: Building Observation Models (A Matrices)
===================================================

This example focuses on constructing observation models - the A matrices that
define the relationship between hidden states and observations in active inference:
- Understanding observation models conceptually
- Building A matrices for different scenarios
- Working with PyMDP's object array structure
- Handling multiple observation modalities

Learning Objectives:
- Learn to construct observation models systematically
- Understand the relationship between hidden states and observations
- Practice building A matrices for realistic scenarios
- Work with PyMDP's multi-modal observation framework

Mathematical Background:
A matrix: P(observation | state)
- Rows represent possible observations
- Columns represent hidden states
- Each column must sum to 1 (probability distribution)
- A[o, s] = probability of observing o given state s

Run with: python 03_observation_models.py
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
OUTPUT_DIR = Path(__file__).parent / "outputs" / "03_observation_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PyMDP imports
import pymdp
from pymdp.utils import obj_array_zeros, obj_array_uniform, is_normalized
from pymdp.maths import softmax, kl_div, entropy
from pymdp.maths.maths import spm_log

# Local imports
from visualization import plot_observation_model
from model_utils import validate_model


def demonstrate_perfect_observation():
    """Demonstrate perfect observation models with VFE analysis."""
    
    print("=" * 60)
    print("PERFECT OBSERVATION MODELS WITH VFE")
    print("=" * 60)
    
    print("Perfect observation means each state produces a unique observation")
    print("with 100% reliability. This maximizes information content and minimizes VFE.")
    print()
    
    # Simple 3-state, 3-observation perfect model
    num_states = 3
    num_obs = 3
    
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.eye(num_states)  # Identity matrix = perfect observation
    
    # Validate using PyMDP utilities
    print("PyMDP Model Validation:")
    is_valid = is_normalized(A)
    print(f"A matrix is normalized: {is_valid}")
    model_valid = validate_model(A, None, None, None, verbose=False)
    print(f"Model structure valid: {model_valid}")
    print()
    
    print("3-State Perfect Observation Model:")
    print("States: [Sunny, Cloudy, Rainy]")
    print("Observations: [See Sun, See Clouds, See Rain]")
    print()
    print("A matrix (P(observation | state)):")
    print("           Sunny  Cloudy  Rainy")
    
    state_names = ["Sunny", "Cloudy", "Rainy"]
    obs_names = ["See Sun", "See Clouds", "See Rain"]
    
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name:11s} {row[0]:.1f}    {row[1]:.1f}     {row[2]:.1f}")
    
    # Analyze information content using PyMDP utilities
    print("\nInformation Analysis (using PyMDP utilities):")
    total_entropy = 0
    for s in range(num_states):
        state_entropy = entropy(A[0][:, s])
        total_entropy += state_entropy
        print(f"State {state_names[s]} entropy: {state_entropy:.6f} (perfect = 0.0)")
    
    print(f"Total model entropy: {total_entropy:.6f}")
    print(f"Average entropy per state: {total_entropy/num_states:.6f}")
    
    print("\nInterpretation:")
    print("- Each state produces unique observation (entropy = 0)")
    print("- No ambiguity → maximum information transfer")
    print("- VFE minimized because model perfectly explains observations")
    print("- Identity matrix A = perfect observation model")
    
    return A, state_names, obs_names


def demonstrate_noisy_observation():
    """Demonstrate observation models with noise and VFE analysis."""
    
    print("\n" + "=" * 60)
    print("NOISY OBSERVATION MODELS WITH VFE")
    print("=" * 60)
    
    print("Realistic observations include noise, which increases VFE.")
    print("The A matrix captures this uncertainty using PyMDP utilities.")
    print()
    
    # Noisy weather observation
    num_states = 3  # [Sunny, Cloudy, Rainy]
    num_obs = 3     # [See Sun, See Clouds, See Rain]
    
    A = obj_array_zeros([[num_obs, num_states]])
    
    # Build noisy observation model using PyMDP best practices
    A[0] = np.array([
        [0.9, 0.1, 0.0],  # P(see sun | sunny/cloudy/rainy)
        [0.1, 0.8, 0.1],  # P(see clouds | sunny/cloudy/rainy) 
        [0.0, 0.1, 0.9]   # P(see rain | sunny/cloudy/rainy)
    ])
    
    state_names = ["Sunny", "Cloudy", "Rainy"]
    obs_names = ["See Sun", "See Clouds", "See Rain"]
    
    # Validate using PyMDP utilities
    print("PyMDP Model Validation:")
    is_valid = is_normalized(A)
    print(f"A matrix is normalized: {is_valid}")
    model_valid = validate_model(A, None, None, None, verbose=False)
    print(f"Model structure valid: {model_valid}")
    print()
    
    print("Noisy Weather Observation Model:")
    print("A matrix (P(observation | state)):")
    print("           Sunny  Cloudy  Rainy")
    
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name:11s} {row[0]:.1f}    {row[1]:.1f}     {row[2]:.1f}")
    
    # Information analysis using PyMDP utilities
    print("\nInformation Analysis (using PyMDP utilities):")
    total_entropy = 0
    max_info_loss = 0
    
    for s in range(num_states):
        state_entropy = entropy(A[0][:, s])
        total_entropy += state_entropy
        info_loss = state_entropy  # Information lost due to noise
        max_info_loss = max(max_info_loss, info_loss)
        print(f"State {state_names[s]} entropy: {state_entropy:.4f} (noise = {info_loss:.4f})")
    
    avg_entropy = total_entropy / num_states
    print(f"Average entropy per state: {avg_entropy:.4f} (vs 0.0 for perfect)")
    
    # Compare observation discriminability 
    print("\nObservation Discriminability:")
    for obs_idx in range(num_obs):
        obs_probs = A[0][obs_idx, :]
        # KL divergence from uniform (measure of discriminability)
        uniform_dist = np.ones(num_states) / num_states
        discriminability = kl_div(obs_probs, uniform_dist)
        print(f"{obs_names[obs_idx]:12s}: discriminability = {discriminability:.4f}")
    
    print("\nInterpretation:")
    print("- Higher entropy = more uncertainty = higher VFE")
    print("- Off-diagonal entries create observation ambiguity")
    print("- Discriminability measures how well observations distinguish states")
    print("- Noise increases computational cost of inference")
    
    return A, state_names, obs_names


def demonstrate_ambiguous_observation():
    """Demonstrate highly ambiguous observation models."""
    
    print("\n" + "=" * 60)
    print("AMBIGUOUS OBSERVATION MODELS")
    print("=" * 60)
    
    print("Sometimes observations provide little information about states.")
    print("This creates high uncertainty and makes inference challenging.")
    print()
    
    # Highly ambiguous model - observations barely distinguish states
    num_states = 3
    num_obs = 2  # Fewer observations than states!
    
    A = obj_array_zeros([[num_obs, num_states]])
    A[0] = np.array([
        [0.6, 0.5, 0.4],  # P(obs 0 | state 0/1/2) - barely informative
        [0.4, 0.5, 0.6]   # P(obs 1 | state 0/1/2) - barely informative
    ])
    
    print("Ambiguous Observation Model:")
    print("States: [Location A, Location B, Location C]")
    print("Observations: [Signal Weak, Signal Strong]")
    print()
    print("A matrix (P(observation | state)):")
    print("              Loc A   Loc B   Loc C")
    
    obs_names = ["Signal Weak", "Signal Strong"]
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name:13s} {row[0]:.1f}     {row[1]:.1f}     {row[2]:.1f}")
    
    print("\nInterpretation:")
    print("- Observations barely distinguish between locations")
    print("- Even 'Signal Strong' only slightly favors Location C")
    print("- This leads to high uncertainty in state inference")
    
    # Calculate mutual information to show how little info observations provide
    # I(S;O) measures how much observations tell us about states
    print("\nInformation Analysis:")
    print("- Small differences between columns = low information")
    print("- This makes active exploration more important!")
    
    return A


def demonstrate_multi_modal_observations():
    """Demonstrate models with multiple observation modalities."""
    
    print("\n" + "=" * 60)
    print("MULTI-MODAL OBSERVATION MODELS") 
    print("=" * 60)
    
    print("Agents often have multiple ways to observe the world.")
    print("Each modality provides different information about states.")
    print()
    
    num_states = 3  # [At Home, At Work, At Store]
    
    # Modality 1: Visual observations
    num_obs_visual = 4  # [See House, See Office, See Products, See Street]
    A_visual = obj_array_zeros([[num_obs_visual, num_states]])
    A_visual[0] = np.array([
        [0.8, 0.0, 0.1],  # P(see house | home/work/store)
        [0.0, 0.9, 0.0],  # P(see office | home/work/store)
        [0.1, 0.0, 0.8],  # P(see products | home/work/store)
        [0.1, 0.1, 0.1]   # P(see street | home/work/store)
    ])
    
    # Modality 2: Auditory observations
    num_obs_audio = 3   # [Hear Quiet, Hear Chatter, Hear Music]
    A_audio = obj_array_zeros([[num_obs_audio, num_states]])
    A_audio[0] = np.array([
        [0.7, 0.1, 0.2],  # P(hear quiet | home/work/store)
        [0.1, 0.8, 0.6],  # P(hear chatter | home/work/store)
        [0.2, 0.1, 0.2]   # P(hear music | home/work/store)
    ])
    
    # Combine into multi-modal model
    A_multi = [A_visual[0], A_audio[0]]
    
    print("Multi-Modal Location Model:")
    print("States: [At Home, At Work, At Store]")
    print()
    
    print("Visual Modality - P(visual observation | state):")
    print("              Home   Work   Store")
    visual_names = ["See House", "See Office", "See Products", "See Street"]
    for i, name in enumerate(visual_names):
        row = A_visual[0][i, :]
        print(f"{name:12s}  {row[0]:.1f}    {row[1]:.1f}     {row[2]:.1f}")
    
    print("\nAuditory Modality - P(audio observation | state):")
    print("              Home   Work   Store")
    audio_names = ["Hear Quiet", "Hear Chatter", "Hear Music"]
    for i, name in enumerate(audio_names):
        row = A_audio[0][i, :]
        print(f"{name:12s}  {row[0]:.1f}    {row[1]:.1f}     {row[2]:.1f}")
    
    print("\nInterpretation:")
    print("- Visual: Houses at home, offices at work, products at store")
    print("- Audio: Quiet at home, chatter at work, mixed sounds at store")
    print("- Combining both modalities gives more reliable state inference")
    
    return A_visual, A_audio


def demonstrate_partial_observability():
    """Demonstrate partial observability scenarios."""
    
    print("\n" + "=" * 60)
    print("PARTIAL OBSERVABILITY")
    print("=" * 60)
    
    print("Often we can only observe some aspects of the world state.")
    print("Hidden states may have multiple dimensions we can't fully observe.")
    print()
    
    # Hidden state has two factors: [Location, Time of Day]
    # But we can only observe indirect cues
    locations = 3  # [Home, Work, Store]  
    times = 2      # [Day, Night]
    num_states = locations * times  # Joint state space: 6 states
    
    # Observations: [Bright, Dim, Crowded, Empty]
    num_obs = 4
    
    A = obj_array_zeros([[num_obs, num_states]])
    
    # Build observation model for factored states
    # States ordered as: [(Home,Day), (Work,Day), (Store,Day), (Home,Night), (Work,Night), (Store,Night)]
    state_names = ["Home-Day", "Work-Day", "Store-Day", "Home-Night", "Work-Night", "Store-Night"]
    
    A[0] = np.array([
        # Bright: More likely during day, varies by location
        [0.6, 0.8, 0.7, 0.1, 0.3, 0.2],  
        # Dim: More likely at night, varies by location  
        [0.2, 0.1, 0.1, 0.7, 0.4, 0.5],  
        # Crowded: More likely at work/store during day
        [0.1, 0.8, 0.6, 0.1, 0.2, 0.1],  
        # Empty: More likely at night, especially at home
        [0.1, 0.1, 0.2, 0.1, 0.1, 0.2]   
    ])
    
    print("Partial Observability Model:")
    print("Hidden States (Location × Time):")
    for i, name in enumerate(state_names):
        print(f"  State {i}: {name}")
    
    print("\nObservations indirectly reveal both location and time:")
    print("A matrix (P(observation | joint state)):")
    print("         ", end="")
    for name in state_names:
        print(f" {name:>9s}", end="")
    print()
    
    obs_names = ["Bright", "Dim", "Crowded", "Empty"] 
    for i, obs_name in enumerate(obs_names):
        print(f"{obs_name:8s} ", end="")
        for j in range(num_states):
            print(f"   {A[0][i, j]:.1f}   ", end="")
        print()
    
    print("\nInterpretation:")
    print("- Bright/Dim cues reveal time of day")
    print("- Crowded/Empty cues reveal location type")
    print("- Agent must combine cues to infer full state")
    print("- Partial observability makes inference more challenging")
    
    return A


def visualize_observation_models():
    """Create visualizations of different observation model types."""
    
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Types of Observation Models', fontsize=16)
    
    # Perfect observation model
    A_perfect = np.eye(3)
    plot_observation_model(
        A_perfect,
        obs_names=["Obs 0", "Obs 1", "Obs 2"],
        state_names=["State 0", "State 1", "State 2"],
        title="Perfect Observation",
        ax=axes[0, 0]
    )
    
    # Noisy observation model
    A_noisy = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9]
    ])
    plot_observation_model(
        A_noisy,
        obs_names=["See Sun", "See Clouds", "See Rain"],
        state_names=["Sunny", "Cloudy", "Rainy"],
        title="Noisy Observation",
        ax=axes[0, 1]
    )
    
    # Ambiguous observation model  
    A_ambiguous = np.array([
        [0.6, 0.5, 0.4],
        [0.4, 0.5, 0.6]
    ])
    plot_observation_model(
        A_ambiguous,
        obs_names=["Signal Weak", "Signal Strong"],
        state_names=["Loc A", "Loc B", "Loc C"],
        title="Ambiguous Observation",
        ax=axes[1, 0]
    )
    
    # Multi-modal visual component
    A_visual = np.array([
        [0.8, 0.0, 0.1],
        [0.0, 0.9, 0.0],
        [0.1, 0.0, 0.8],
        [0.1, 0.1, 0.1]
    ])
    plot_observation_model(
        A_visual,
        obs_names=["House", "Office", "Products", "Street"],
        state_names=["Home", "Work", "Store"],
        title="Multi-Modal (Visual)",
        ax=axes[1, 1]
    )
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "observation_model_types.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Observation model visualizations saved to: {OUTPUT_DIR / 'observation_model_types.png'}")
    
    return fig


def demonstrate_model_construction():
    """Demonstrate systematic construction of observation models."""
    
    print("\n" + "=" * 60)
    print("SYSTEMATIC MODEL CONSTRUCTION")
    print("=" * 60)
    
    print("Step-by-step process for building observation models:")
    print()
    
    # Example: Robot navigation with sensors
    print("Example: Robot with noisy sensors")
    print("1. Define hidden states (robot locations)")
    print("2. Define observations (sensor readings)")
    print("3. Specify sensor noise/reliability")
    print("4. Build A matrix")
    print("5. Validate model")
    print()
    
    # Step 1: Define states
    num_states = 4  # [Room1, Room2, Corridor, Outside]
    state_names = ["Room1", "Room2", "Corridor", "Outside"]
    print(f"Step 1 - States: {state_names}")
    
    # Step 2: Define observations
    num_obs = 4     # [GPS_Indoor, GPS_Outdoor, Walls_Close, Walls_Far]
    obs_names = ["GPS_Indoor", "GPS_Outdoor", "Walls_Close", "Walls_Far"]
    print(f"Step 2 - Observations: {obs_names}")
    
    # Step 3: Specify sensor characteristics
    print("Step 3 - Sensor characteristics:")
    print("  - GPS works better outdoors")
    print("  - Wall sensor detects proximity")
    print("  - Both sensors have some noise")
    
    # Step 4: Build A matrix
    print("Step 4 - Build A matrix:")
    A = obj_array_zeros([[num_obs, num_states]])
    
    A[0] = np.array([
        [0.7, 0.1, 0.1, 0.1],  # GPS_Indoor: likely at indoor locations
        [0.1, 0.1, 0.1, 0.7],  # GPS_Outdoor: likely outside
        [0.1, 0.7, 0.7, 0.1],  # Walls_Close: rooms and corridor have close walls
        [0.1, 0.1, 0.1, 0.1]   # Walls_Far: rare everywhere (for normalization)
    ])
    
    # Each column must sum to 1, so normalize
    for j in range(num_states):
        A[0][:, j] = A[0][:, j] / np.sum(A[0][:, j])
    
    print("A matrix (P(observation | state)):")
    print("              Room1  Room2  Corridor Outside")
    for i, obs_name in enumerate(obs_names):
        row = A[0][i, :]
        print(f"{obs_name:12s}  {row[0]:.1f}   {row[1]:.1f}    {row[2]:.1f}     {row[3]:.1f}")
    
    # Step 5: Validate
    print("\nStep 5 - Validation:")
    valid = validate_model(A, None, None, None, verbose=False)
    if valid:
        print("✓ Model is valid (columns sum to 1)")
    else:
        print("✗ Model has validation errors")
    
    # Show column sums
    print("Column sums (should be 1.0):")
    for i, state_name in enumerate(state_names):
        col_sum = np.sum(A[0][:, i])
        print(f"  {state_name}: {col_sum:.3f}")
    
    return A


def main():
    """Main function to run all demonstrations."""
    
    print("PyMDP Example 3: Building Observation Models (A Matrices)")
    print("=" * 60)
    print("This example shows how to construct observation models systematically.")
    print("Key concepts: A matrices, partial observability, multi-modal sensing")
    print()
    
    # Run demonstrations
    A_perfect, perfect_states, perfect_obs = demonstrate_perfect_observation()
    A_noisy, noisy_states, noisy_obs = demonstrate_noisy_observation()
    A_ambiguous = demonstrate_ambiguous_observation()
    A_visual, A_audio = demonstrate_multi_modal_observations()
    A_partial = demonstrate_partial_observability()
    A_robot = demonstrate_model_construction()
    
    # Visualization
    fig = visualize_observation_models()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS: OBSERVATION MODELS & VFE")
    print("=" * 60)
    print("1. A matrices encode P(observation | state) using PyMDP obj_array structure")
    print("2. Perfect observation: identity matrix (entropy=0, minimal VFE)")
    print("3. Noise/ambiguity: off-diagonal entries increase entropy and VFE")
    print("4. Information content: lower entropy = better state discrimination")
    print("5. Multi-modal: separate A matrices provide complementary information")
    print("6. Partial observability: indirect observation requires inference")
    print("7. PyMDP validation: is_normalized() and validate_model() ensure correctness")
    print("8. Discriminability: KL divergence measures observation informativeness")
    
    print("\nPyMDP Methods Used:")
    print("- pymdp.utils.obj_array_zeros() for proper A matrix structure")
    print("- pymdp.utils.is_normalized() for validation")
    print("- pymdp.maths.entropy() for information analysis")
    print("- pymdp.maths.kl_div() for discriminability measurement")
    print("- @src/model_utils.validate_model() for comprehensive validation")
    
    print("\nNext: Example 4 will show how to use these A matrices for state inference")
    
    # Save summary data
    summary_data = {
        'perfect_observation': A_perfect[0].tolist(),
        'noisy_observation': A_noisy[0].tolist(),
        'ambiguous_observation': A_ambiguous[0].tolist(),
        'multi_modal_visual': A_visual[0].tolist(),
        'multi_modal_audio': A_audio[0].tolist(),
        'partial_observability': A_partial[0].tolist(),
        'robot_navigation': A_robot[0].tolist(),
        'model_types': {
            'perfect': 'Identity matrix - each state produces unique observation',
            'noisy': 'High diagonal, low off-diagonal - realistic sensor noise',
            'ambiguous': 'Small differences - observations barely distinguish states',
            'multi_modal': 'Multiple observation channels provide different info',
            'partial': 'Joint states but indirect observation cues'
        }
    }
    
    import json
    with open(OUTPUT_DIR / "example_03_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
