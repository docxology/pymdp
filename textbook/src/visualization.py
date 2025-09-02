"""
Visualization Utilities
=======================

Tools for visualizing active inference models, agent behavior, and learning dynamics.
Provides intuitive plots and animations to understand PyMDP models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple, Dict, Any
import seaborn as sns


def plot_beliefs(
    beliefs: np.ndarray,
    state_names: Optional[List[str]] = None,
    title: str = "Posterior Beliefs",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot posterior beliefs over states.
    
    Parameters
    ----------
    beliefs : np.ndarray
        Probability distribution over states
    state_names : list of str, optional
        Names for states (default: State 0, State 1, ...)
    title : str
        Plot title
    ax : matplotlib.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.Axes
        The plot axes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    num_states = len(beliefs)
    if state_names is None:
        state_names = [f"State {i}" for i in range(num_states)]
    
    # Create bar plot
    bars = ax.bar(range(num_states), beliefs, **kwargs)
    ax.set_xlabel("States")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_xticks(range(num_states))
    ax.set_xticklabels(state_names, rotation=45 if len(state_names) > 5 else 0)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, beliefs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return ax


def plot_policy_tree(
    policies: List[List[int]],
    policy_probs: np.ndarray,
    action_names: Optional[List[str]] = None,
    max_depth: int = 3,
    title: str = "Policy Tree",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize policy tree with probabilities.
    
    Parameters
    ----------
    policies : list of lists
        Policy sequences (action sequences)
    policy_probs : np.ndarray
        Probability of each policy
    action_names : list of str, optional
        Names for actions
    max_depth : int
        Maximum depth to visualize
    title : str
        Plot title
    ax : matplotlib.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.Axes
        The plot axes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if action_names is None:
        action_names = [f"Action {i}" for i in range(max(max(p) for p in policies) + 1)]
    
    # Simple tree visualization
    y_positions = np.linspace(0, 1, len(policies))
    
    for i, (policy, prob) in enumerate(zip(policies, policy_probs)):
        y_pos = y_positions[i]
        
        # Draw policy path
        x_positions = np.arange(min(len(policy), max_depth))
        
        # Line thickness proportional to probability
        linewidth = 1 + 5 * prob
        
        for j, action in enumerate(policy[:max_depth]):
            if j < max_depth - 1:
                # Draw connection to next action
                ax.plot([j, j+1], [y_pos, y_pos], 'b-', linewidth=linewidth, alpha=0.7)
            
            # Draw action node
            circle = patches.Circle((j, y_pos), 0.03, facecolor='lightblue', 
                                   edgecolor='blue', linewidth=2)
            ax.add_patch(circle)
            
            # Add action label
            ax.text(j, y_pos - 0.08, action_names[action], ha='center', va='top', fontsize=8)
        
        # Add probability label
        ax.text(-0.15, y_pos, f'π{i}: {prob:.3f}', ha='right', va='center', fontsize=9)
    
    ax.set_xlim([-0.5, max_depth - 0.5])
    ax.set_ylim([-0.2, 1.2])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Policies")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return ax


def plot_free_energy(
    free_energy_history: List[float],
    title: str = "Free Energy Over Time",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot free energy evolution over time.
    
    Parameters
    ----------
    free_energy_history : list of float
        Free energy values over time
    title : str
        Plot title
    ax : matplotlib.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.Axes
        The plot axes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(free_energy_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Free Energy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add trend line if enough data points
    if len(free_energy_history) > 3:
        z = np.polyfit(range(len(free_energy_history)), free_energy_history, 1)
        p = np.poly1d(z)
        ax.plot(range(len(free_energy_history)), p(range(len(free_energy_history))), 
                'r--', alpha=0.7, label=f'Trend (slope: {z[0]:.3f})')
        ax.legend()
    
    plt.tight_layout()
    return ax


def plot_gridworld(
    height: int,
    width: int,
    agent_pos: Optional[Tuple[int, int]] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
    obstacle_pos: Optional[List[Tuple[int, int]]] = None,
    beliefs: Optional[np.ndarray] = None,
    title: str = "Grid World",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize a grid world environment.
    
    Parameters
    ----------
    height, width : int
        Grid dimensions
    agent_pos : tuple, optional
        (row, col) position of agent
    goal_pos : tuple, optional
        (row, col) position of goal
    obstacle_pos : list of tuples, optional
        List of (row, col) obstacle positions
    beliefs : np.ndarray, optional
        Beliefs over grid positions (for visualization)
    title : str
        Plot title
    ax : matplotlib.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.Axes
        The plot axes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(width+1, height+1))
    
    # Create grid
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Draw grid lines
    for i in range(height + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(width + 1):
        ax.axvline(j, color='gray', linewidth=0.5)
    
    # Fill cells with belief intensities if provided
    if beliefs is not None:
        beliefs_2d = beliefs.reshape((height, width))
        im = ax.imshow(beliefs_2d, cmap='Blues', alpha=0.7, 
                      extent=[0, width, 0, height], origin='lower')
        plt.colorbar(im, ax=ax, label='Belief Probability')
    
    # Draw obstacles
    if obstacle_pos is not None:
        for row, col in obstacle_pos:
            rect = patches.Rectangle((col, height-row-1), 1, 1, 
                                   facecolor='black', alpha=0.8)
            ax.add_patch(rect)
            ax.text(col + 0.5, height-row-0.5, 'X', ha='center', va='center',
                   color='white', fontsize=16, fontweight='bold')
    
    # Draw goal
    if goal_pos is not None:
        row, col = goal_pos
        circle = patches.Circle((col + 0.5, height-row-0.5), 0.3, 
                               facecolor='gold', edgecolor='orange', linewidth=3)
        ax.add_patch(circle)
        ax.text(col + 0.5, height-row-0.5, 'G', ha='center', va='center',
               color='black', fontsize=12, fontweight='bold')
    
    # Draw agent
    if agent_pos is not None:
        row, col = agent_pos
        circle = patches.Circle((col + 0.5, height-row-0.5), 0.25, 
                               facecolor='red', edgecolor='darkred', linewidth=2)
        ax.add_patch(circle)
        ax.text(col + 0.5, height-row-0.5, 'A', ha='center', va='center',
               color='white', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.set_title(title)
    ax.invert_yaxis()  # Make (0,0) top-left
    
    plt.tight_layout()
    return ax


def plot_observation_model(
    A: np.ndarray,
    obs_names: Optional[List[str]] = None,
    state_names: Optional[List[str]] = None,
    title: str = "Observation Model (A Matrix)",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize observation model as heatmap.
    
    Parameters
    ----------
    A : np.ndarray
        Observation model matrix
    obs_names : list of str, optional
        Names for observations
    state_names : list of str, optional
        Names for states
    title : str
        Plot title
    ax : matplotlib.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.Axes
        The plot axes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    num_obs, num_states = A.shape
    
    if obs_names is None:
        obs_names = [f"Obs {i}" for i in range(num_obs)]
    if state_names is None:
        state_names = [f"State {i}" for i in range(num_states)]
    
    # Create heatmap
    im = ax.imshow(A, cmap='Blues', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='P(observation | state)')
    
    # Add labels
    ax.set_xticks(range(num_states))
    ax.set_xticklabels(state_names, rotation=45)
    ax.set_yticks(range(num_obs))
    ax.set_yticklabels(obs_names)
    ax.set_xlabel('Hidden States')
    ax.set_ylabel('Observations')
    ax.set_title(title)
    
    # Add text annotations
    for i in range(num_obs):
        for j in range(num_states):
            ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center',
                   color='white' if A[i, j] > 0.5 else 'black', fontsize=9)
    
    plt.tight_layout()
    return ax


def animate_agent(
    trajectory: List[Tuple[int, int]],
    height: int,
    width: int,
    goal_pos: Optional[Tuple[int, int]] = None,
    obstacle_pos: Optional[List[Tuple[int, int]]] = None,
    title: str = "Agent Trajectory",
    interval: int = 500,
    save_path: Optional[str] = None
) -> FuncAnimation:
    """
    Create animation of agent moving through environment.
    
    Parameters
    ----------
    trajectory : list of tuples
        Sequence of (row, col) positions
    height, width : int
        Grid dimensions
    goal_pos : tuple, optional
        Goal position
    obstacle_pos : list of tuples, optional
        Obstacle positions
    title : str
        Animation title
    interval : int
        Milliseconds between frames
    save_path : str, optional
        Path to save animation
        
    Returns
    -------
    anim : FuncAnimation
        The animation object
    """
    
    fig, ax = plt.subplots(figsize=(width+1, height+1))
    
    def animate(frame):
        ax.clear()
        
        # Get current position
        if frame < len(trajectory):
            agent_pos = trajectory[frame]
        else:
            agent_pos = trajectory[-1]
        
        # Draw environment
        plot_gridworld(height, width, agent_pos, goal_pos, obstacle_pos,
                      title=f"{title} (Step {frame})", ax=ax)
        
        # Draw trajectory so far
        if frame > 0:
            past_trajectory = trajectory[:frame]
            for i, pos in enumerate(past_trajectory[:-1]):
                row, col = pos
                ax.plot(col + 0.5, height-row-0.5, 'o', color='lightcoral', 
                       markersize=5, alpha=0.7)
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory)+5, 
                        interval=interval, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    return anim


def plot_learning_curves(
    metrics: Dict[str, List[float]],
    title: str = "Learning Curves",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot multiple learning metrics over time.
    
    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names to lists of values
    title : str
        Plot title
    ax : matplotlib.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.Axes
        The plot axes
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.plot(values, label=metric_name, color=colors[i], 
               linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Time Step / Episode')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return ax


def plot_observation_model(A_matrix, obs_names=None, state_names=None, 
                          title="Observation Model", ax=None):
    """
    Plot observation model (A matrix) as a heatmap.
    
    Parameters
    ----------
    A_matrix : np.ndarray
        Observation model matrix (observations x states)
    obs_names : list, optional
        Names for observations
    state_names : list, optional  
        Names for states
    title : str
        Title for the plot
    ax : matplotlib axis, optional
        Axis to plot on
        
    Returns
    -------
    ax : matplotlib axis
        The axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(A_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Set labels
    if state_names is not None:
        ax.set_xticks(range(len(state_names)))
        ax.set_xticklabels(state_names)
    if obs_names is not None:
        ax.set_yticks(range(len(obs_names)))
        ax.set_yticklabels(obs_names)
    
    ax.set_xlabel('Hidden States')
    ax.set_ylabel('Observations')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='P(obs | state)')
    
    # Add text annotations
    for i in range(A_matrix.shape[0]):
        for j in range(A_matrix.shape[1]):
            text = ax.text(j, i, f'{A_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white" if A_matrix[i, j] > 0.5 else "black")
    
    return ax


def apply_accessibility_enhancements():
    """
    Apply consistent, accessible matplotlib styling across examples.

    Returns
    -------
    colors : list
        Colorblind-friendly categorical palette used by examples
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.weight': 'normal',
        'axes.titleweight': 'bold'
    })

    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]

    return colors
