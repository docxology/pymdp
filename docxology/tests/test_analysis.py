import jax.numpy as jnp
from pkg.support.analysis import compute_entropy, trajectory_divergence, marginalize_actions


def test_compute_entropy():
    # Uniform distribution over 4 states
    probs = jnp.array([0.25, 0.25, 0.25, 0.25])
    h = compute_entropy(probs)
    # log(4) = 1.3862944
    assert jnp.allclose(h, jnp.log(4.0), atol=1e-5)
    
    # Deterministic distribution
    probs_det = jnp.array([1.0, 0.0, 0.0])
    h_det = compute_entropy(probs_det)
    assert jnp.allclose(h_det, 0.0, atol=1e-5)


def test_trajectory_divergence():
    prior = jnp.array([0.5, 0.5])
    qs_seq = [
        jnp.array([0.5, 0.5]),    # matches prior -> KL=0
        jnp.array([0.9, 0.1]),    # diverges
        jnp.array([1.0, 0.0])     # max divergence
    ]
    
    divs = trajectory_divergence(qs_seq, prior)
    assert len(divs) == 3
    assert jnp.allclose(divs[0], 0.0, atol=1e-5)
    assert divs[1] > 0
    assert divs[2] > divs[1]


def test_marginalize_actions():
    q_pi = jnp.array([0.8, 0.2])
    # 2 policies, length 1, 1 factor
    policies = jnp.array([
        [[0]],  # policy 0 -> action 0
        [[1]]   # policy 1 -> action 1
    ])
    
    marginal = marginalize_actions(q_pi, policies)
    # Expected: factor 0 -> action 0: 0.8, action 1: 0.2
    assert marginal.shape == (1, 2)
    assert jnp.allclose(marginal[0, 0], 0.8)
    assert jnp.allclose(marginal[0, 1], 0.2)
