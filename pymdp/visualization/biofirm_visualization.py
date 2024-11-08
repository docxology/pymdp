"""
Biofirm Visualization Module
==========================

Visualization utilities for Biofirm active inference experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Union, List

logger = logging.getLogger(__name__)

def plot_active_inference_summary(histories: Dict, output_dir: Path):
    """Plot comprehensive active inference summary"""
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot belief dynamics
        plot_belief_dynamics(histories['beliefs'], output_dir / "belief_dynamics.png")
        
        # Plot action frequencies
        plot_action_frequencies(histories['actions'], output_dir / "action_frequencies.png")
        
        # Plot state transitions
        plot_state_transitions(histories['true_states'], output_dir / "state_transitions.png")
        
        # Plot policy probabilities
        plot_policy_probabilities(histories['policy_probs'], output_dir / "policy_probabilities.png")
        
        # Verify files were created
        required_plots = [
            'belief_dynamics.png',
            'action_frequencies.png',
            'state_transitions.png',
            'policy_probabilities.png'
        ]
        
        missing_plots = []
        for plot in required_plots:
            if not (output_dir / plot).exists():
                missing_plots.append(plot)
                
        if missing_plots:
            raise RuntimeError(f"Failed to create plots: {', '.join(missing_plots)}")
            
        logger.info(f"Generated all plots in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error plotting active inference summary: {str(e)}")
        raise

def plot_belief_dynamics(beliefs: np.ndarray, output_file: Path):
    """Plot belief dynamics over time"""
    plt.figure(figsize=(12, 6))
    labels = ['LOW', 'HOMEO', 'HIGH']
    for i in range(beliefs.shape[1]):
        plt.plot(beliefs[:, i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Belief Probability')
    plt.title('Belief Dynamics')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_action_frequencies(actions: np.ndarray, output_file: Path):
    """Plot action frequency histogram"""
    plt.figure(figsize=(10, 6))
    counts = np.bincount(actions, minlength=3)
    plt.bar(['DECREASE', 'MAINTAIN', 'INCREASE'], counts / len(actions))
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title('Action Selection Distribution')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_transitions(states: np.ndarray, output_file: Path):
    """Plot state transitions over time"""
    plt.figure(figsize=(12, 6))
    plt.plot(states, 'b-', alpha=0.7)
    plt.axhline(y=1, color='g', linestyle='--', label='Homeostasis')
    plt.fill_between(range(len(states)), states == 1, alpha=0.2, color='g')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('State Transitions')
    plt.yticks([0, 1, 2], ['LOW', 'HOMEO', 'HIGH'])
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_policy_probabilities(policy_probs: np.ndarray, output_file: Path):
    """Plot policy probability dynamics"""
    plt.figure(figsize=(12, 6))
    labels = ['DECREASE', 'MAINTAIN', 'INCREASE']
    for i in range(policy_probs.shape[1]):
        plt.plot(policy_probs[:, i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Policy Probability')
    plt.title('Policy Selection Probabilities')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()