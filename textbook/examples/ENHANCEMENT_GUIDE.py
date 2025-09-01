#!/usr/bin/env python3
"""
PyMDP Integration Enhancement Guide
===================================

This guide provides specific code patterns for maximizing PyMDP integration
across all textbook examples. Use these patterns to enhance existing examples.

Key Enhancement Areas:
1. PyMDP Agent class integration
2. Core PyMDP inference functions  
3. PyMDP control and policy functions
4. PyMDP learning functions
5. Improved visualizations
"""

import numpy as np
import matplotlib.pyplot as plt

# PyMDP imports that should be used throughout examples
import pymdp
from pymdp.agent import Agent
from pymdp.inference import update_posterior_states
from pymdp.control import sample_action, update_posterior_policies_full
from pymdp.learning import update_obs_likelihood_dirichlet, update_state_likelihood_dirichlet
from pymdp.utils import obj_array_zeros, obj_array_uniform, is_normalized
from pymdp.maths import softmax, entropy, kl_div, spm_log
from pymdp.algos import run_vanilla_fpi


def enhancement_pattern_dual_implementation():
    """
    Enhancement Pattern 1: Dual Implementation
    ==========================================
    
    Always provide both educational and PyMDP implementations for comparison.
    This validates the educational implementation and shows PyMDP usage.
    """
    
    # Example: State inference (for Examples 01-06)
    def educational_inference(A, obs, prior, verbose=True):
        """Educational step-by-step implementation."""
        likelihood = A[0][obs, :]
        joint = likelihood * prior[0] 
        posterior = joint / np.sum(joint)
        
        # VFE decomposition for education
        complexity = kl_div(posterior, prior[0])
        accuracy = np.sum(posterior * smp_log(likelihood))
        vfe = complexity - accuracy
        
        if verbose:
            print(f"Educational VFE: {vfe:.4f}")
            print(f"  Complexity: {complexity:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        return posterior, vfe
    
    def pymdp_inference(A, obs, prior, verbose=True):
        """PyMDP implementation for comparison."""
        # Use actual PyMDP function
        qs = update_posterior_states(A, obs, prior)
        
        if verbose:
            print(f"PyMDP result: {qs[0] if hasattr(qs, '__iter__') and hasattr(qs[0], '__iter__') else qs}")
        
        return qs
    
    # Validation pattern
    print("=== DUAL IMPLEMENTATION PATTERN ===")
    
    # Setup
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    prior = obj_array_zeros([3])
    prior[0] = np.array([1/3, 1/3, 1/3])
    obs = 0
    
    # Educational implementation
    print("\n1. Educational Implementation:")
    edu_result, edu_vfe = educational_inference(A, obs, prior)
    
    # PyMDP implementation  
    print("\n2. PyMDP Implementation:")
    pymdp_result = pymdp_inference(A, obs, prior)
    
    # Validation
    print("\n3. Validation:")
    # Extract result properly (PyMDP may return different formats)
    if hasattr(pymdp_result, '__iter__') and len(pymdp_result) > 0:
        if hasattr(pymdp_result[0], '__iter__'):
            pymdp_posterior = pymdp_result[0]
        else:
            pymdp_posterior = pymdp_result
    else:
        pymdp_posterior = pymdp_result
        
    match = np.allclose(edu_result, pymdp_posterior, atol=1e-6)
    print(f"Results match: {match}")
    if match:
        print("✅ Educational implementation validated against PyMDP!")
    else:
        print("❌ Discrepancy found - check implementation")
        print(f"Educational: {edu_result}")
        print(f"PyMDP: {pymdp_posterior}")


def enhancement_pattern_agent_class():
    """
    Enhancement Pattern 2: PyMDP Agent Class Integration
    ===================================================
    
    Examples 07-12 should use the PyMDP Agent class as the primary implementation.
    This shows real-world usage patterns.
    """
    
    print("\n=== AGENT CLASS PATTERN ===")
    
    # Setup complete POMDP model
    num_obs = [3]     # 3 observations  
    num_states = [3]  # 3 states
    num_controls = [3] # 3 actions
    
    # A matrix (observation model)
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    
    # B matrix (transition model)  
    B = obj_array_zeros([[3, 3, 3]])  # states x states x actions
    B[0][:, :, 0] = np.eye(3)  # Action 0: no change
    B[0][:, :, 1] = np.roll(np.eye(3), 1, axis=0)  # Action 1: forward
    B[0][:, :, 2] = np.roll(np.eye(3), -1, axis=0)  # Action 2: backward
    
    # C vector (preferences)
    C = obj_array_zeros([3])
    C[0] = np.array([0.0, 0.0, 1.0])  # Prefer state 2
    
    # D vector (prior)
    D = obj_array_uniform(num_states)
    
    # Create PyMDP Agent
    print("1. Creating PyMDP Agent:")
    agent = Agent(
        A=A, B=B, C=C, D=D,
        inference_algo='VANILLA',
        policy_len=1,
        control_fac_idx=[0]
    )
    print("✅ Agent created successfully")
    
    # Demonstration of agent usage
    print("\n2. Agent Usage Pattern:")
    
    # Step 1: Observe
    observation = [1]  # Observe something
    print(f"Observation: {observation}")
    
    # Step 2: Infer states
    qs = agent.infer_states(observation)
    print(f"Inferred states: {qs[0]}")
    print(f"Most likely state: {np.argmax(qs[0])}")
    
    # Step 3: Infer policies
    q_pi, G = agent.infer_policies()
    print(f"Policy probabilities: {q_pi}")
    print(f"Expected free energies: {G}")
    
    # Step 4: Sample action
    action = agent.sample_action()
    print(f"Selected action: {action}")
    
    print("\n✅ Complete Agent workflow demonstrated")
    
    return agent


def enhancement_pattern_learning():
    """
    Enhancement Pattern 3: PyMDP Learning Integration
    ================================================
    
    Examples should show how to update model parameters using PyMDP learning functions.
    This demonstrates adaptive behavior.
    """
    
    print("\n=== LEARNING PATTERN ===")
    
    # Setup for learning demonstration
    A = obj_array_zeros([[3, 3]])
    A[0] = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])  # Noisy initial model
    
    # Prior Dirichlet parameters (for learning)
    pA = obj_array_zeros([[3, 3]])
    pA[0] = np.ones((3, 3))  # Uniform prior
    
    print("1. Initial A matrix (noisy):")
    print(A[0])
    
    # Simulate some observations and state beliefs for learning
    observations = [0, 0, 1, 1, 2, 2]  # Sequence of observations
    state_beliefs = []  # We'll generate these
    
    # For demonstration, create some state beliefs
    current_belief = np.array([1/3, 1/3, 1/3])
    
    print("\n2. Learning from observations:")
    for i, obs in enumerate(observations):
        # Infer states (simplified)
        likelihood = A[0][obs, :]
        joint = likelihood * current_belief
        new_belief = joint / np.sum(joint)
        state_beliefs.append(new_belief.copy())
        
        # Update A matrix using PyMDP learning
        qs = obj_array_zeros([3])  
        qs[0] = new_belief
        
        pA = update_obs_likelihood_dirichlet(pA, A, [obs], qs)
        A = pymdp.utils.norm_dist_obj_arr(pA)  # Convert to probabilities
        
        print(f"Step {i+1}: obs={obs}, belief={new_belief.round(3)}")
        
        current_belief = new_belief
    
    print("\n3. Learned A matrix:")
    print(A[0])
    
    print("\n✅ Learning integration demonstrated")


def enhancement_pattern_comprehensive_visualization():
    """
    Enhancement Pattern 4: Comprehensive Accessible Visualization  
    ============================================================
    
    Enhanced visualization patterns for better accessibility and understanding.
    """
    
    print("\n=== VISUALIZATION ENHANCEMENT PATTERN ===")
    
    # Create sample data for visualization
    belief_evolution = np.array([
        [0.33, 0.33, 0.34],  # t=0
        [0.6, 0.3, 0.1],     # t=1  
        [0.8, 0.15, 0.05],   # t=2
        [0.85, 0.1, 0.05],   # t=3
        [0.9, 0.08, 0.02]    # t=4
    ])
    
    vfe_evolution = [1.2, 0.8, 0.5, 0.3, 0.2]
    observations = [0, 0, 1, 0]
    
    # Enhanced visualization with accessibility features
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced PyMDP Visualization Pattern', fontsize=16, fontweight='bold')
    
    # 1. Belief evolution with enhanced accessibility
    ax = axes[0, 0]
    times = range(len(belief_evolution))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colorblind-friendly
    state_names = ['State A', 'State B', 'State C']
    
    for i, (color, name) in enumerate(zip(colors, state_names)):
        ax.plot(times, belief_evolution[:, i], 'o-', color=color, 
                linewidth=3, markersize=8, label=name)
    
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Belief Probability', fontsize=12, fontweight='bold')
    ax.set_title('Belief Evolution Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Add observation annotations
    for i, obs in enumerate(observations):
        ax.annotate(f'obs={obs}', xy=(i+1, 0.9), xytext=(i+1, 0.95),
                   fontsize=10, ha='center', 
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # 2. VFE evolution with enhanced features
    ax = axes[0, 1]
    ax.bar(range(1, len(vfe_evolution)+1), vfe_evolution, 
           color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('VFE (Surprise)', fontsize=12, fontweight='bold')  
    ax.set_title('VFE Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Add value labels on bars
    for i, v in enumerate(vfe_evolution):
        ax.text(i+1, v + 0.02, f'{v:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 3. Model matrix visualization
    ax = axes[1, 0]
    A_matrix = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    
    im = ax.imshow(A_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(state_names, fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Obs 0', 'Obs 1', 'Obs 2'], fontsize=11)
    ax.set_title('Observation Model (A Matrix)', fontsize=14, fontweight='bold')
    
    # Add text annotations with larger, bold font
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{A_matrix[i, j]:.1f}', ha='center', va='center',
                   color='white' if A_matrix[i, j] > 0.5 else 'black', 
                   fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 4. Information summary panel
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""PyMDP Integration Summary:
    
Key Methods Used:
• pymdp.Agent() - Main agent class
• agent.infer_states() - State inference  
• agent.infer_policies() - Policy inference
• agent.sample_action() - Action selection
• pymdp.learning functions - Parameter updates
    
Results:
• Final belief certainty: {np.max(belief_evolution[-1]):.1%}
• Total VFE reduction: {vfe_evolution[0] - vfe_evolution[-1]:.1f}
• Most likely final state: {state_names[np.argmax(belief_evolution[-1])]}
    
Educational Value:
✅ PyMDP methods demonstrated
✅ VFE tracking throughout  
✅ Accessible visualization
✅ Real-world applicability"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))
    ax.set_title('Integration Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with high quality for accessibility
    plt.savefig('/home/trim/Documents/GitHub/pymdp/textbook/examples/outputs/enhanced_visualization_pattern.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Enhanced visualization pattern demonstrated")
    print("📊 Saved to: outputs/enhanced_visualization_pattern.png")


def enhancement_pattern_interactive():
    """
    Enhancement Pattern 5: Interactive Exploration
    ==============================================
    
    Pattern for adding interactive exploration to examples.
    """
    
    print("\n=== INTERACTIVE PATTERN ===")
    
    def interactive_pymdp_demo():
        """Interactive demonstration with PyMDP Agent."""
        
        # Setup
        print("Setting up PyMDP Agent for interactive demo...")
        
        # Simple navigation environment
        A = obj_array_zeros([[4, 4]])  # 4 observations, 4 states
        A[0] = np.array([
            [0.8, 0.1, 0.05, 0.05],  # Obs 0 → State 0 likely
            [0.1, 0.8, 0.05, 0.05],  # Obs 1 → State 1 likely
            [0.05, 0.05, 0.8, 0.1],  # Obs 2 → State 2 likely  
            [0.05, 0.05, 0.1, 0.8]   # Obs 3 → State 3 likely
        ])
        
        B = obj_array_zeros([[4, 4, 2]])  # states x states x actions
        B[0][:, :, 0] = np.eye(4)  # Action 0: stay
        B[0][:, :, 1] = np.roll(np.eye(4), 1, axis=0)  # Action 1: move
        
        C = obj_array_zeros([4])
        C[0] = np.array([0, 0, 0, 1])  # Prefer state 3 (goal)
        
        D = obj_array_uniform([4])
        
        agent = Agent(A=A, B=B, C=C, D=D, policy_len=1, control_fac_idx=[0])
        
        print("🤖 PyMDP Agent ready for interaction!")
        print("\nAvailable commands:")
        print("- Enter observation (0-3)")
        print("- 'q' to quit")
        print("- 'status' for agent status")
        print("- 'help' for this message")
        
        step = 0
        while True:
            try:
                print(f"\n--- Step {step} ---")
                user_input = input("Enter observation or command: ").strip().lower()
                
                if user_input == 'q':
                    print("👋 Goodbye!")
                    break
                elif user_input == 'help':
                    print("\nCommands: obs(0-3), 'status', 'q'uit")
                    continue
                elif user_input == 'status':
                    print(f"Current beliefs: {agent.qs[0].round(3)}")
                    print(f"Most likely state: {np.argmax(agent.qs[0])}")
                    continue
                
                # Try to parse as observation
                obs = int(user_input)
                if obs not in [0, 1, 2, 3]:
                    print("❌ Observation must be 0, 1, 2, or 3")
                    continue
                
                # Process with PyMDP Agent
                qs = agent.infer_states([obs])
                q_pi, G = agent.infer_policies()
                action = agent.sample_action()
                
                print(f"📡 Observed: {obs}")
                print(f"🧠 State beliefs: {qs[0].round(3)}")
                print(f"📈 Policy probs: {q_pi.round(3)}")
                print(f"🎯 Selected action: {int(action[0])}")
                print(f"⚡ Expected free energies: {G.round(3)}")
                
                step += 1
                
            except ValueError:
                print("❌ Please enter a valid observation (0-3) or command")
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
    
    # For demonstration, just show the pattern
    print("Interactive pattern demonstrated (would run interactive_pymdp_demo())")
    print("✅ Interactive exploration pattern ready")


def main():
    """
    Run all enhancement patterns to demonstrate maximum PyMDP integration.
    """
    
    print("🚀 PyMDP Integration Enhancement Patterns")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs('/home/trim/Documents/GitHub/pymdp/textbook/examples/outputs', exist_ok=True)
    
    # Run all enhancement patterns
    enhancement_pattern_dual_implementation()
    agent = enhancement_pattern_agent_class()  
    enhancement_pattern_learning()
    enhancement_pattern_comprehensive_visualization()
    enhancement_pattern_interactive()
    
    print("\n" + "=" * 60)
    print("🎉 ALL ENHANCEMENT PATTERNS DEMONSTRATED")
    print("=" * 60)
    print("These patterns should be integrated into textbook examples 01-12")
    print("for maximum PyMDP integration and educational value.")
    
    print("\n📋 Integration Checklist:")
    print("✅ Dual implementation pattern (educational + PyMDP)")
    print("✅ PyMDP Agent class integration")  
    print("✅ PyMDP learning functions")
    print("✅ Enhanced accessible visualization")
    print("✅ Interactive exploration patterns")
    
    print("\n🎯 Next Steps:")
    print("1. Apply these patterns to examples 01-06 (add PyMDP comparisons)")
    print("2. Apply these patterns to examples 07-12 (full PyMDP integration)")
    print("3. Enhance all visualizations with accessibility features")
    print("4. Add interactive modes to all examples")
    
    return agent


if __name__ == "__main__":
    # Run demonstration
    main()
