"""Optional figure export (headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


def _apply_aesthetics():
    """Apply a premium, publication-ready visual style globally."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        pass # Fallback gracefully if seaborn styles are missing
        
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.facecolor': '#f8f9fa',
        'axes.facecolor': '#ffffff',
        'grid.alpha': 0.6,
        'grid.color': '#cccccc',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2
    })

_apply_aesthetics()


def save_current_figure(path: Path | None, *, stem: str, suffix: str = ".png") -> Path | None:
    if path is None:
        plt.close()
        return None
        
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{stem}{suffix}"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_beliefs_heatmap(qs_seq: list[jnp.ndarray], title: str = "Posterior Beliefs") -> None:
    """Plot a heatmap of states over time for a single factor."""
    if not qs_seq:
        return
        
    qs_stack = jnp.stack(qs_seq, axis=-1)
    qs_2d = jnp.squeeze(qs_stack)
    if qs_2d.ndim == 1:
        qs_2d = jnp.expand_dims(qs_2d, axis=-1)
    elif qs_2d.ndim > 2:
        qs_2d = qs_2d.reshape(-1, qs_2d.shape[-1])
        
    plt.figure(figsize=(max(5, qs_2d.shape[1] * 0.6), max(4, qs_2d.shape[0] * 0.6)))
    plt.imshow(qs_2d, aspect='auto', cmap='magma', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('Probability', rotation=270, labelpad=15)
    plt.title(title, pad=15)
    plt.xlabel("Timestep")
    plt.ylabel("Hidden State")
    plt.tight_layout()


def plot_free_energy(F_seq: list[float] | jnp.ndarray, title: str = "Variational Free Energy") -> None:
    """Line plot of variational free energy over time."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(F_seq, marker='o', markersize=6, linestyle='-', color='#d62728', linewidth=2)
    plt.title(title, pad=15)
    plt.xlabel("Timestep")
    plt.ylabel("Free Energy (F)")
    plt.tight_layout()


def plot_action_probabilities(action_probs: jnp.ndarray, title: str = "Action Probabilities") -> None:
    """Bar plot of marginal action probabilities for a single control factor."""
    plt.figure(figsize=(7, 4.5))
    x_pos = np.arange(len(action_probs))
    plt.bar(x_pos, action_probs, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1)
    plt.title(title, pad=15)
    plt.xlabel("Action (Control State Index)")
    plt.ylabel("Probability")
    plt.xticks(x_pos)
    plt.ylim(0, 1.05)
    plt.tight_layout()


def plot_policy_posterior(q_pi: jnp.ndarray, title: str = "Policy Posterior (q_pi)") -> None:
    """Bar plot of the inferred probability distribution over policies."""
    plt.figure(figsize=(max(7, len(q_pi) * 0.5), 4.5))
    x_pos = np.arange(len(q_pi))
    plt.bar(x_pos, q_pi, color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=1)
    plt.title(title, pad=15)
    plt.xlabel("Policy Index")
    plt.ylabel("Posterior Probability")
    plt.xticks(x_pos, rotation=45 if len(q_pi) > 10 else 0)
    plt.ylim(0, 1.05)
    plt.tight_layout()


def plot_likelihood_matrix(A_matrix, title: str = "Likelihood Matrix (A) - Modality 0") -> None:
    """Heatmap plot of observation likelihoods P(o|s)."""
    A = A_matrix[0] if isinstance(A_matrix, (list, tuple)) else A_matrix
    A = np.asarray(A)
    if A.ndim > 2:
        A = A.reshape(A.shape[0], -1)
        
    plt.figure(figsize=(max(5, A.shape[1] * 0.6), max(4, A.shape[0] * 0.6)))
    plt.imshow(A, aspect='auto', cmap='Blues', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('P(o|s)', rotation=270, labelpad=15)
    plt.title(title, pad=15)
    plt.xlabel("Hidden State")
    plt.ylabel("Observation")
    plt.tight_layout()


def plot_transition_matrix(B_matrix, title: str = "Transition Matrix (B) - Factor 0, Control 0") -> None:
    """Heatmap plot of state transitions P(s_t|s_{t-1}, u)."""
    B = B_matrix[0] if isinstance(B_matrix, (list, tuple)) else B_matrix
    B = np.asarray(B)
    if B.ndim >= 3:
        B = B[..., 0]
        
    plt.figure(figsize=(max(5, B.shape[1] * 0.6), max(4, B.shape[0] * 0.6)))
    plt.imshow(B, aspect='auto', cmap='Purples', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('P(s_t|s_{t-1})', rotation=270, labelpad=15)
    plt.title(title, pad=15)
    plt.xlabel("Previous State ($s_{t-1}$)")
    plt.ylabel("Next State ($s_t$)")
    plt.tight_layout()


def plot_empirical_prior(D_matrix, title: str = "Empirical Prior (D) - Factor 0") -> None:
    """Bar plot of prior beliefs over hidden states P(s)."""
    D = D_matrix[0] if isinstance(D_matrix, (list, tuple)) else D_matrix
    D = np.asarray(D).flatten()
    
    plt.figure(figsize=(7, 4.5))
    x_pos = np.arange(len(D))
    plt.bar(x_pos, D, color='#8c564b', alpha=0.85, edgecolor='black', linewidth=1)
    plt.title(title, pad=15)
    plt.xlabel("Hidden State Index")
    plt.ylabel("Prior Probability")
    plt.xticks(x_pos)
    plt.ylim(0, max(1.05, float(np.max(D)) * 1.1))
    plt.tight_layout()


def plot_prior_preferences(C_matrix, title: str = "Prior Preferences (C) - Modality 0") -> None:
    """Bar plot of prior preferences over observations P(o)."""
    C = C_matrix[0] if isinstance(C_matrix, (list, tuple)) else C_matrix
    C = np.asarray(C).flatten()
    
    plt.figure(figsize=(7, 4.5))
    x_pos = np.arange(len(C))
    # Diverging color map logic based on positive/negative preferences
    colors = ['#2ca02c' if val >= 0 else '#d62728' for val in C]
    
    plt.bar(x_pos, C, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    plt.title(title, pad=15)
    plt.xlabel("Observation Index")
    plt.ylabel("Utility (Log Probability)")
    plt.xticks(x_pos)
    
    # Add horizontal zero line for clarity
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    
    # Dynamic y-limits preserving zero-crossing
    c_max, c_min = float(np.max(C)), float(np.min(C))
    pad = max(abs(c_max), abs(c_min)) * 0.15
    plt.ylim(min(0, c_min) - pad, max(0, c_max) + pad)
    plt.tight_layout()


def plot_efe_components(g_epistemic: jnp.ndarray, g_pragmatic: jnp.ndarray, title: str = "Expected Free Energy Breakdown") -> None:
    """Stacked bar plot breaking down EFE into Ambiguity (Epistemic) and Risk (Pragmatic) components."""
    # Active Inference typically splits G = Epistemic Value + Pragmatic Value (or Ambiguity + Risk)
    e_vals = np.asarray(g_epistemic).flatten()
    p_vals = np.asarray(g_pragmatic).flatten()
    
    if len(e_vals) != len(p_vals):
        return # Skip if dimension mismatch
        
    x_pos = np.arange(len(e_vals))
    width = 0.35
    
    plt.figure(figsize=(max(8, len(e_vals) * 0.6), 5))
    
    # Grouped bars to show comparison clearly
    plt.bar(x_pos - width/2, e_vals, width, label='Epistemic Value (Info Gain)', color='#ff7f0e', alpha=0.85, edgecolor='black')
    plt.bar(x_pos + width/2, p_vals, width, label='Pragmatic Value (Utility)', color='#1f77b4', alpha=0.85, edgecolor='black')
    
    plt.title(title, pad=15)
    plt.xlabel("Policy Index")
    plt.ylabel("Expected Free Energy (G) Magnitude")
    plt.xticks(x_pos, rotation=45 if len(e_vals) > 10 else 0)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.tight_layout()


def plot_entropy_trajectory(qs_seq: list, title: str = "Belief Entropy Over Time") -> None:
    """Line plot of Shannon entropy of beliefs at each timestep."""
    if not qs_seq or len(qs_seq) < 2:
        return
    
    entropies = []
    for qs in qs_seq:
        q = np.asarray(qs).flatten()
        q = np.clip(q, 1e-12, 1.0)
        h = -np.sum(q * np.log(q))
        entropies.append(float(h))
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(len(entropies)), entropies, marker='s', markersize=5, 
             linestyle='-', color='#9467bd', linewidth=2, label='H(q)')
    plt.fill_between(range(len(entropies)), entropies, alpha=0.15, color='#9467bd')
    plt.title(title, pad=15)
    plt.xlabel("Timestep")
    plt.ylabel("Shannon Entropy (nats)")
    plt.legend(loc='best')
    plt.tight_layout()


def plot_kl_divergence_trajectory(qs_seq: list, prior, title: str = "KL Divergence from Prior") -> None:
    """Line plot of KL(q || prior) at each timestep."""
    if not qs_seq or len(qs_seq) < 2:
        return
    
    prior_flat = np.asarray(prior).flatten()
    prior_flat = np.clip(prior_flat, 1e-12, 1.0)
    
    kl_vals = []
    for qs in qs_seq:
        q = np.asarray(qs).flatten()
        if q.shape != prior_flat.shape:
            continue
        q = np.clip(q, 1e-12, 1.0)
        kl = float(np.sum(q * (np.log(q) - np.log(prior_flat))))
        kl_vals.append(kl)
    
    if len(kl_vals) < 2:
        return
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(len(kl_vals)), kl_vals, marker='D', markersize=5, 
             linestyle='-', color='#e377c2', linewidth=2, label='KL(q || p)')
    plt.fill_between(range(len(kl_vals)), kl_vals, alpha=0.15, color='#e377c2')
    plt.title(title, pad=15)
    plt.xlabel("Timestep")
    plt.ylabel("KL Divergence (nats)")
    plt.legend(loc='best')
    plt.tight_layout()


def plot_belief_trajectory_animation(qs_seq: list, output_dir: Path | None, stem: str, 
                                      title: str = "Belief Evolution", fps: int = 4) -> Path | None:
    """Generate animated GIF of belief distribution evolving over time."""
    if not qs_seq or len(qs_seq) < 3 or output_dir is None:
        return None
    
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return None
    
    # Normalize beliefs to 1D arrays
    frames = []
    for qs in qs_seq:
        q = np.asarray(qs).flatten()
        frames.append(q)
    
    num_states = len(frames[0])
    x_pos = np.arange(num_states)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x_pos, frames[0], color='#17becf', alpha=0.85, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hidden State")
    ax.set_ylabel("Posterior Probability")
    ax.set_title(f"{title} (t=0)", pad=15)
    ax.set_xticks(x_pos)
    
    def update(frame_idx):
        for bar, val in zip(bars, frames[frame_idx]):
            bar.set_height(float(val))
        ax.set_title(f"{title} (t={frame_idx})", pad=15)
        return bars
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_beliefs_anim.gif"
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def plot_efe_trajectory(neg_efe_seq: list, title: str = "Expected Free Energy Over Time") -> None:
    """Line plot of mean and best (max) neg-EFE across policies per timestep."""
    if not neg_efe_seq or len(neg_efe_seq) < 2:
        return
    
    means = []
    bests = []
    for step_g in neg_efe_seq:
        arr = np.asarray(step_g).flatten()
        means.append(float(np.mean(arr)))
        bests.append(float(np.max(arr)))
    
    plt.figure(figsize=(8, 4.5))
    t = range(len(means))
    plt.plot(t, bests, marker='o', markersize=4, linestyle='-', color='#2ca02c', linewidth=2, label='Best policy (-G)')
    plt.plot(t, means, marker='s', markersize=3, linestyle='--', color='#7f7f7f', linewidth=1.5, label='Mean across policies')
    plt.fill_between(t, means, bests, alpha=0.12, color='#2ca02c')
    plt.title(title, pad=15)
    plt.xlabel("Timestep")
    plt.ylabel("Negative EFE (−G)")
    plt.legend(loc='best')
    plt.tight_layout()


def plot_policy_posterior_heatmap(q_pi_seq: list, title: str = "Policy Posterior Evolution") -> None:
    """Heatmap showing how q(π) evolves over timesteps (rows=time, cols=policies)."""
    if not q_pi_seq or len(q_pi_seq) < 2:
        return
    
    # Build matrix: rows=timesteps, cols=policies
    rows = []
    for qp in q_pi_seq:
        rows.append(np.asarray(qp).flatten())
    
    min_len = min(len(r) for r in rows)
    mat = np.array([r[:min_len] for r in rows])
    
    if mat.shape[1] > 30:
        # Truncate to top-30 policies by max probability for readability
        max_probs = mat.max(axis=0)
        top_idx = np.argsort(max_probs)[-30:]
        mat = mat[:, top_idx]
    
    plt.figure(figsize=(max(8, mat.shape[1] * 0.3), max(5, mat.shape[0] * 0.2)))
    plt.imshow(mat, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(label='q(π)')
    plt.title(title, pad=15)
    plt.xlabel("Policy Index")
    plt.ylabel("Timestep")
    plt.tight_layout()


def plot_action_frequency_donut(actions: list, title: str = "Action Distribution") -> None:
    """Donut chart showing action frequency across the simulation."""
    if not actions:
        return
    
    flat = np.asarray(actions).flatten().astype(int)
    unique_acts, counts = np.unique(flat, return_counts=True)
    
    if len(unique_acts) < 2:
        return  # Not interesting with only one action
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_acts)))
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=[f"Action {a}" for a in unique_acts],
        autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        textprops={'fontsize': 10}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color('white')
        t.set_fontweight('bold')
    ax.set_title(title, pad=20, fontsize=13, fontweight='bold')
    plt.tight_layout()


def plot_neg_efe_heatmap(neg_efe_seq: list, title: str = "Neg-EFE Landscape") -> None:
    """Heatmap of neg-EFE values: rows=timesteps, cols=policies. Shows which policies become favoured."""
    if not neg_efe_seq or len(neg_efe_seq) < 2:
        return
    
    rows = [np.asarray(g).flatten() for g in neg_efe_seq]
    min_len = min(len(r) for r in rows)
    mat = np.array([r[:min_len] for r in rows])
    
    if mat.shape[1] > 30:
        max_vals = mat.max(axis=0)
        top_idx = np.argsort(max_vals)[-30:]
        mat = mat[:, top_idx]
    
    plt.figure(figsize=(max(8, mat.shape[1] * 0.3), max(5, mat.shape[0] * 0.2)))
    plt.imshow(mat, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='−G (neg EFE)')
    plt.title(title, pad=15)
    plt.xlabel("Policy Index")
    plt.ylabel("Timestep")
    plt.tight_layout()


def plot_reachability_matrix(I_matrix, title: str = "Reachability Matrix (I)") -> None:
    """Heatmap of the inductive-inference reachability matrix I.

    I[i,j] indicates whether state j is reachable from state i within the
    planning horizon given the goal defined by H.
    """
    I = I_matrix[0] if isinstance(I_matrix, (list, tuple)) else I_matrix
    I = np.asarray(I)
    if I.ndim > 2:
        I = I.reshape(I.shape[0], -1)

    plt.figure(figsize=(max(5, I.shape[1] * 0.8), max(4, I.shape[0] * 0.8)))
    plt.imshow(I, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('Reachability', rotation=270, labelpad=15)
    plt.title(title, pad=15)
    plt.xlabel("Target State")
    plt.ylabel("Source State")

    # Annotate cells with values
    for r in range(I.shape[0]):
        for c in range(I.shape[1]):
            val = float(I[r, c])
            color = 'white' if val > 0.5 else 'black'
            plt.text(c, r, f"{val:.2f}", ha='center', va='center', fontsize=10, color=color)
    plt.tight_layout()


def plot_spm_log_comparison(x: "np.ndarray", log_x: "np.ndarray",
                            title: str = "SPM Log Transform") -> None:
    """Side-by-side bar chart comparing a distribution with its spm_log values.

    Useful for visualising the legacy free-energy calculation.
    """
    x = np.asarray(x).flatten()
    log_x = np.asarray(log_x).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x_pos = np.arange(len(x))
    ax1.bar(x_pos, x, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1)
    ax1.set_title("Input Distribution", pad=12)
    ax1.set_xlabel("State Index")
    ax1.set_ylabel("Probability")
    ax1.set_xticks(x_pos)
    ax1.set_ylim(0, 1.05)

    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in log_x]
    ax2.bar(x_pos, log_x, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    ax2.set_title("spm_log(x)", pad=12)
    ax2.set_xlabel("State Index")
    ax2.set_ylabel("Log Value")
    ax2.set_xticks(x_pos)
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()


def plot_sparse_structure(dense_matrix: "np.ndarray", nnz: int,
                          title: str = "Sparse Matrix Structure") -> None:
    """Spy-style plot showing the non-zero structure of a sparse matrix."""
    mat = np.asarray(dense_matrix)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Heatmap of values
    im = ax1.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title("Dense Values", pad=12)
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    # Spy plot (binary non-zero pattern)
    ax2.spy(mat, markersize=max(4, 20 // max(mat.shape)), color='#d62728')
    ax2.set_title(f"Sparsity Pattern (nnz={nnz})", pad=12)
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

