# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, random, lax
from typing import List, Optional
from jaxtyping import Array
from jax import random as jr
import matplotlib.pyplot as plt
import numpy as np

from pymdp.envs import GridWorldEnv
from pymdp.jax import control as j_control
from pymdp.jax.agent import Agent as AIFAgent
num_rows, num_columns = 7, 7
num_states = [num_rows*num_columns] # number of states equals the number of grid locations
num_obs = [num_rows*num_columns]    # number of observations equals the number of grid locations (fully observable)

# number of agents
n_batches = 5

# construct A arrays
A = [jnp.broadcast_to(jnp.eye(num_states[0]), (n_batches,) + (num_obs[0], num_states[0]))] # fully observable (identity observation matrix

# construct B arrays
grid_world = GridWorldEnv(shape=[num_rows, num_columns])
B = [jnp.broadcast_to(jnp.array(grid_world.get_transition_dist()), (n_batches,) + (num_states[0], num_states[0], grid_world.n_control))]  # easy way to get the generative model parameters is to extract them from one of pre-made GridWorldEnv classes
num_controls = [grid_world.n_control] # number of control states equals the number of actions
 
# create mapping from gridworld coordinates to linearly-index states
grid = np.arange(grid_world.n_states).reshape(grid_world.shape)
it = np.nditer(grid, flags=["multi_index"])
coord_to_idx_map = {}
while not it.finished:
    coord_to_idx_map[it.multi_index] = it.iterindex
    it.iternext()

# construct C arrays
desired_position = (6,6) # lower corner
desired_state_id = coord_to_idx_map[desired_position]
desired_obs_id = jnp.argmax(A[0][:, desired_state_id]) # throw this in there, in case there is some indeterminism between states and observations
C = [jnp.broadcast_to(nn.one_hot(desired_obs_id, num_obs[0]), (n_batches, num_obs[0]))]

# construct D arrays
starting_position = (3, 3) # middle
# starting_position = (0, 0) # upper left corner
starting_state_id = coord_to_idx_map[starting_position]
starting_obs_id = jnp.argmax(A[0][:, starting_state_id]) # throw this in there, in case there is some indeterminism between states and observations
D = [jnp.broadcast_to(nn.one_hot(starting_state_id, num_states[0]), (n_batches, num_states[0]))]
planning_horizon, inductive_threshold = 1, 0.1
inductive_depth = 7
policy_matrix = j_control.construct_policies(num_states, num_controls, policy_len=planning_horizon)

# inductive planning goal states
H = [jnp.broadcast_to(nn.one_hot(desired_state_id, num_states[0]), (n_batches, num_states[0]))] # list of factor-specific goal vectors (shape of each is (n_batches, num_states[f]))
# create agent
agent = AIFAgent(A, B, C, D, E=None, pA=None, pB=None, policies=policy_matrix, policy_len=planning_horizon, 
                inductive_depth=inductive_depth, inductive_threshold=inductive_threshold,
                H=H, use_utility=True, use_states_info_gain=False, use_param_info_gain=False, use_inductive=True)
# T = 14 # needed if you start further away from the goal (e.g. in upper left corner)
T = 7 # can get away with fewer timesteps if you start closer to the goal (e.g. in the middle)

qs_init = [jnp.broadcast_to(nn.one_hot(starting_state_id, num_states[0]), (n_batches, num_states[0]))] # same as D
obs_idx = [jnp.broadcast_to(starting_obs_id, (n_batches,))] # list of len (num_modalities), each list element of shape (n_batches,)
obs_idx  = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), obs_idx) #  list of len (num_modalities), elements each of shape (n_batches,1), this adds a trivial "time dimension"

state = jnp.broadcast_to(starting_state_id, (n_batches,))
infer_args = (agent.D, None,)
batch_keys = jr.split(jr.PRNGKey(0), n_batches)
batch_to_track = 1

for t in range(T):

    print('Grid position for agent {} at time {}: {}'.format(batch_to_track+1, t, np.unravel_index(state[batch_to_track], grid_world.shape)))

    if t == 0:
        actions = None
    else:
        actions = actions_t
    beliefs = agent.infer_states(obs_idx, empirical_prior=infer_args[0], past_actions=actions, qs_hist=infer_args[1])
    q_pi, _ = agent.infer_policies(beliefs)
    actions_t = agent.sample_action(q_pi, rng_key=batch_keys)
    infer_args = agent.update_empirical_prior(actions_t, beliefs)

    # get next state and observation from the grid world (need to vmap everything over batches)
    state = vmap(lambda b, s, a: jnp.argmax(b[:, s, a]), in_axes=(0,0,0))(B[0], state, actions_t)
    next_obs = vmap(lambda a, s: jnp.argmax(a[:, s]), in_axes=(0,0))(A[0], state)
    obs_idx = [next_obs]
    obs_idx  = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), obs_idx) # add a trivial time dimension to the observation to enable indexing during agent.infer_states

