#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import jax.numpy as jnp
from .algos import run_factorized_fpi, run_mmp, run_vmp
from jax import tree_util as jtu, lax

def update_posterior_states(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            qs = run_mmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
    
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            qs_hist = qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            qs_hist = qs
    
    return qs_hist

def joint_dist_factor(b, filtered_qs, actions):
    qs_last = filtered_qs[-1]
    qs_filter = filtered_qs[:-1]

    # conditional dist - timestep x s_{t+1} | s_{t}
    time_b = jnp.moveaxis(b[..., actions], -1, 0)

    # joint dist - timestep x s_{t+1} x s_{t}
    qs_joint = time_b * jnp.expand_dims(qs_filter, -1)

    # cond dist - timestep x s_{t} | s_{t+1}
    qs_backward_cond = jnp.moveaxis(
        qs_joint / qs_joint.sum(-2, keepdims=True), -2, -1
    )

    def step_fn(qs_smooth_past, backward_b):
        qs_joint = backward_b * qs_smooth_past
        qs_smooth = qs_joint.sum(-1)
        
        return qs_smooth, (qs_smooth, qs_joint)

    # seq_qs will contain a sequence of smoothed marginals and joints
    _, seq_qs = lax.scan(
        step_fn,
        qs_last,
        qs_backward_cond,
        reverse=True,
        unroll=2
    )

    # we add the last filtered belief to smoothed beliefs
    qs_smooth_all = jnp.concatenate([seq_qs[0], jnp.expand_dims(qs_last, 0)], 0)
    return qs_smooth_all, seq_qs[1]


def smoothing_ovf(filtered_post, B, past_actions):
    assert len(filtered_post) == len(B)
    nf = len(B)  # number of factors
    joint = lambda b, qs, f: joint_dist_factor(b, qs, past_actions[..., f])
    marginals_and_joints = jtu.tree_map(
        joint, B, filtered_post, list(range(nf))
    )

    return marginals_and_joints


    
