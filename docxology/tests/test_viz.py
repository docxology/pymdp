from pathlib import Path
import jax.numpy as jnp
from pkg.support.viz import plot_beliefs_heatmap, plot_free_energy, plot_action_probabilities, save_current_figure


def test_plot_beliefs_heatmap(tmp_path: Path):
    qs_seq = [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.1, 0.8, 0.1]),
        jnp.array([0.0, 0.0, 1.0])
    ]
    plot_beliefs_heatmap(qs_seq, title="Test Heatmap")
    out = save_current_figure(tmp_path, stem="test_heatmap")
    assert out is not None and out.exists()


def test_plot_free_energy(tmp_path: Path):
    F_seq = jnp.array([10.5, 8.2, 5.1, 2.0])
    plot_free_energy(F_seq)
    out = save_current_figure(tmp_path, stem="test_fe")
    assert out is not None and out.exists()


def test_plot_action_probabilities(tmp_path: Path):
    action_probs = jnp.array([0.1, 0.7, 0.2])
    plot_action_probabilities(action_probs)
    out = save_current_figure(tmp_path, stem="test_ap")
    assert out is not None and out.exists()
