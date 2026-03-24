"""Small generative models for sophisticated-inference / MCTS smoke tests."""

from __future__ import annotations

import jax.numpy as jnp

from pymdp import utils


def build_single_cue_tmaze_like_model():
    """Same structure as ``test_sophisticated_inference_jax._build_single_cue_model``."""
    num_locations = 4
    num_reward_states = 2

    cue_obs = jnp.zeros((5, num_locations, num_reward_states), dtype=jnp.float32)
    reward_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)

    for loc in range(num_locations):
        for reward_state in range(num_reward_states):
            if loc == 0:
                cue_obs = cue_obs.at[0, loc, reward_state].set(1.0)
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)
            elif loc == 3:
                cue_idx = 3 if reward_state == 0 else 4
                cue_obs = cue_obs.at[cue_idx, loc, reward_state].set(1.0)
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)
            elif loc == 1:
                cue_obs = cue_obs.at[1, loc, reward_state].set(1.0)
                observation_idx = 1 if reward_state == 0 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            elif loc == 2:
                cue_obs = cue_obs.at[2, loc, reward_state].set(1.0)
                observation_idx = 1 if reward_state == 1 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)

    A = [cue_obs, reward_obs]
    A_dependencies = [[0, 1], [0, 1]]

    B_loc = utils.create_controllable_B(num_locations, num_locations)[0]
    B_reward = jnp.eye(num_reward_states, dtype=jnp.float32).reshape(
        num_reward_states, num_reward_states, 1
    )
    B = [B_loc, B_reward]
    B_dependencies = [[0], [1]]

    D = [
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    ]

    return A, B, A_dependencies, B_dependencies, D


def si_cue_agent(*, policy_len: int = 1):
    A, B, A_dependencies, B_dependencies, D = build_single_cue_tmaze_like_model()
    cue_cost = -2.0
    reward_value = 6.0
    punishment_value = -12.0
    C = [
        jnp.array([0.0, 0.0, 0.0, cue_cost, cue_cost], dtype=jnp.float32),
        jnp.array([0.0, reward_value, punishment_value], dtype=jnp.float32),
    ]
    from pymdp.agent import Agent

    return Agent(
        A,
        B,
        C=C,
        D=D,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=[4, 1],
        policy_len=policy_len,
    )


def run_si_policy_search_smoke(*, fast: bool) -> dict:
    import jax.tree_util as jtu
    from jax import random as jr

    from pymdp.planning.si import si_policy_search

    agent = si_cue_agent(policy_len=1)
    qs = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
    horizon = 2
    search_fn = si_policy_search(
        horizon=horizon,
        max_nodes=128 if fast else 512,
        max_branching=16 if fast else 32,
        policy_prune_threshold=0.0,
        observation_prune_threshold=0.0,
        entropy_stop_threshold=-1.0,
        gamma=4.0,
        topk_obsspace=2,
    )
    q_pi, _info = search_fn(agent, qs=qs, rng_key=jr.PRNGKey(0))
    return {"q_pi_shape": tuple(q_pi[0].shape), "action_probs": q_pi[0]}


def run_mcts_smoke(*, fast: bool, seed: int) -> dict:
    import jax.tree_util as jtu
    from jax import random as jr

    from pymdp.planning.mcts import mcts_policy_search

    agent = si_cue_agent(policy_len=1)
    qs = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
    num_sim = 8 if fast else 128
    depth = 2 if fast else 4
    search_fn = mcts_policy_search(max_depth=depth, num_simulations=num_sim)
    w, _raw = search_fn(agent, qs, jr.PRNGKey(seed))
    return {"action_weights_shape": tuple(w.shape), "action_probs": w}
