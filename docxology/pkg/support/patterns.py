"""Reusable pymdp call patterns for docxology orchestrations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import random as jr

from pymdp import utils
from pymdp.agent import Agent


def random_agent_one_cycle(
    *,
    seed: int,
    policy_len: int = 1,
    inference_algo: str = "fpi",
) -> dict[str, Any]:
    """Build a random multi-factor model and run one infer_states / infer_policies / sample_action."""
    a_key, b_key, d_key, o_key, act_key = jr.split(jr.PRNGKey(seed), 5)
    num_obs = [3, 5]
    num_states = [3, 2]
    num_controls = [3, 1]
    A_dependencies = [[0], [0, 1]]
    B_dependencies = [[0], [0, 1]]
    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
    C = utils.list_array_uniform([[no] for no in num_obs])
    agent = Agent(
        A=A,
        B=B,
        C=C,
        batch_size=1,
        policy_len=policy_len,
        inference_algo=inference_algo,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        num_controls=num_controls,
    )
    observation = [jnp.array([[1]]), jnp.array([[4]])]
    qs, _info = agent.infer_states(observation, empirical_prior=agent.D, return_info=True)
    q_pi, neg_efe = agent.infer_policies(qs)
    keys = jr.split(act_key, agent.batch_size + 1)
    action = agent.sample_action(q_pi, rng_key=keys[1:])
    return {
        "qs_shapes": [tuple(x.shape) for x in qs],
        "action": [int(jnp.asarray(a).reshape(-1)[0]) for a in action],
        "A_matrix": A,
        "B_matrix": B,
        "C_matrix": C,
        "D_matrix": agent.D,
        "diagnostics": {
            "beliefs": [x.squeeze().tolist() for x in qs],
            "q_pi": q_pi.squeeze().tolist() if hasattr(q_pi, 'tolist') else list(q_pi),
            "neg_efe": neg_efe.squeeze().tolist() if hasattr(neg_efe, 'tolist') else list(neg_efe),
        },
    }


def complex_action_dependency_agent(*, seed: int) -> Agent:
    """Agent with ``B_action_dependencies`` (mirrors notebook / tests)."""
    import math as pymath

    a_key, b_key = jr.split(jr.PRNGKey(seed), 2)
    num_obs = [2, 3]
    num_states = [4, 5, 2]
    num_controls = [2, 3, 2]
    A_dependencies = [[0, 1], [1]]
    B_dependencies = [[0], [0, 1, 2], [2]]
    B_action_dependencies = [[], [0, 1], [0, 2]]
    A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
    B = utils.random_B_array(
        b_key,
        num_states,
        num_controls,
        B_dependencies=B_dependencies,
        B_action_dependencies=B_action_dependencies,
    )
    C = utils.list_array_uniform([[no] for no in num_obs])
    return Agent(
        A,
        B,
        C,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        B_action_dependencies=B_action_dependencies,
        num_controls=num_controls,
        sampling_mode="full",
    )
