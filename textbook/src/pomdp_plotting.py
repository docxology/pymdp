"""
POMDP Plotting Helpers
======================

Reusable plotting functions for textbook POMDP examples. These helpers keep
visualization concerns separate from control/inference logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def _add_heatmap_annotations(matrix: np.ndarray, ax: plt.Axes, fmt: str = '%.2f') -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, fmt % val, ha='center', va='center', color=color, fontsize=10, fontweight='bold')


def plot_A_matrix(A: np.ndarray, ax: Optional[plt.Axes] = None,
                  state_labels: Optional[List[str]] = None,
                  obs_labels: Optional[List[str]] = None,
                  title: str = 'A Matrix\nP(Observation | State)') -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    if state_labels is None:
        state_labels = [f'State {i}' for i in range(A.shape[1])]
    if obs_labels is None:
        obs_labels = [f'Obs {i}' for i in range(A.shape[0])]

    im = ax.imshow(A, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(state_labels)))
    ax.set_xticklabels(state_labels, fontsize=11)
    ax.set_yticks(range(len(obs_labels)))
    ax.set_yticklabels(obs_labels, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    _add_heatmap_annotations(A, ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_B_matrix_slice(B_slice: np.ndarray, ax: Optional[plt.Axes] = None,
                        state_labels: Optional[List[str]] = None,
                        next_state_labels: Optional[List[str]] = None,
                        title: str = 'B Matrix Slice\nP(Next | Current, Action)',
                        cmap: str = 'Reds') -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    if state_labels is None:
        state_labels = [f'State {i}' for i in range(B_slice.shape[1])]
    if next_state_labels is None:
        next_state_labels = [f'State {i}' for i in range(B_slice.shape[0])]

    im = ax.imshow(B_slice, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(state_labels)))
    ax.set_xticklabels(state_labels, fontsize=11)
    ax.set_yticks(range(len(next_state_labels)))
    ax.set_yticklabels(next_state_labels, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    _add_heatmap_annotations(B_slice, ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_C_vector_bar(C: np.ndarray, ax: Optional[plt.Axes] = None,
                      obs_labels: Optional[List[str]] = None,
                      title: str = 'C Vector\nPreferences') -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    if obs_labels is None:
        obs_labels = [f'Obs {i}' for i in range(len(C))]

    bars = ax.bar(obs_labels, C, color=['lightcoral', 'gold', 'lightgreen'][:len(C)], alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Log Preference', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)

    for bar, val in zip(bars, C):
        height = bar.get_height()
        y_pos = height + 0.05 if height >= 0 else height - 0.15
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos, f'{val:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=12, fontweight='bold')
    return ax


def plot_D_vector_bar(D: np.ndarray, ax: Optional[plt.Axes] = None,
                      state_labels: Optional[List[str]] = None,
                      title: str = 'D Vector\nPrior Beliefs') -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    if state_labels is None:
        state_labels = [f'State {i}' for i in range(len(D))]

    bars = ax.bar(state_labels, D, color='lightblue', alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    if len(D) > 0:
        ax.set_ylim(0, max(D) * 1.2)

    for bar, val in zip(bars, D):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    return ax


