# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.

import os
# Set cuda device to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# do not prealocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import numpy as np
from functools import partial
from jax import vmap, lax, nn, jit, remat
from jax import random as jr
from pymdp.jax.agent import Agent as AIFAgent
from pymdp.utils import random_A_matrix, random_B_matrix
from opt_einsum import contract
# @partial(jit, static_argnames=['dims', 'keep_dims'])
def factor_dot(M, xs, dims, keep_dims = None):
    """ Dot product of a multidimensional array with `x`.
    
    Parameters
    ----------
    - `qs` [list of 1D numpy.ndarray] - list of jnp.ndarrays
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    all_dims = list(range(M.ndim))
    matrix = [[xs[f], dims[f]] for f in range(len(xs))]
    args = [M, all_dims]
    for row in matrix:
        args.extend(row)

    args += [keep_dims]
    return contract(*args, backend='jax', optimize='auto')

@vmap
def get_marginals(posterior):
  d = posterior.ndim - 1
  marginals = []
  for i in range(d):
     marginals.append( jnp.sum(posterior, axis=(j + 1 for j in range(d) if j != i)) )

  return marginals

@vmap
def merge_marginals(marginals):
  q = marginals[0]
  for m in marginals[1:]:
    q = jnp.expand_dims(q, -1) * m
  
  return q
def make_tuple(i, d, ext):
    l = [i,]
    l.extend(d + i for i in ext)
    return tuple(l)

make_tuple(0, 1, (1, 2))
@partial(vmap, in_axes=(0, 0, None, None))
def delta_A(beliefs, outcomes, deps, num_obs):
  def merge(beliefs, outcomes):
    y = nn.one_hot(outcomes, num_obs)
    d = beliefs.ndim
    marg_beliefs = jnp.sum(beliefs, axis=(i for i in range(d) if i not in deps))
    axis = ( - (i+1) for i in range(len(deps)))
    return jnp.expand_dims(y, axis) * marg_beliefs
  
  return vmap(merge, in_axes=(0, None))(beliefs, outcomes)
  
@partial(vmap, in_axes=(0, 0, 0, None))
def delta_B(post_b, cond_b, action, num_actions):
   a = nn.one_hot(action, num_actions)
   all_dims = tuple(range(cond_b.ndim - 1))
   fd = lambda x, y: factor_dot(x, [y], ((0,),), keep_dims=all_dims)
   b = vmap(fd)(cond_b, post_b)
   return b * a

@partial(vmap, in_axes=(None, 0))
def get_reverse_conditionals(B, beliefs):
  all_dims = tuple(range(B.ndim - 1))
  dims = tuple((i,) for i in all_dims[1:-1])
  fd = lambda x, y: factor_dot(x, y, dims, keep_dims=all_dims)
  joint = vmap(fd)(B, beliefs)
  pred = joint.sum(axis=all_dims[2:], keepdims=True)
  return joint / pred

@partial(vmap, in_axes=(0, 0, None))
def get_reverse_predictive(post, cond, deps):
  def pred(post, cond, deps):
    d = post.ndim
    dims = tuple(make_tuple(i, d, deps[i]) for i in range(len(deps)))
    keep_dims = list(dims[0][1:])
    for row in dims[1:]:
      keep_dims.extend(list(row[1:]))
    
    unique_dims = tuple(set(keep_dims))

    return factor_dot(post, cond, dims, keep_dims=unique_dims)
  
  out = vmap(pred, in_axes=(0, 0, None))(post, cond, deps)
  return out

def learning(agent, beliefs, actions, outcomes, lag=1):
  A_deps = agent.A_dependencies
  B_deps = agent.B_dependencies
  num_obs = agent.num_obs
  posterior_beliefs = merge_marginals( jtu.tree_map(lambda x: x[..., -1, :], beliefs) )
  qA = agent.pA
  qB = agent.pB

  def step_fn(carry, xs):
    posterior_beliefs, qA, qB = carry
    obs, acts, filter_beliefs = xs
    # learn A matrix
    if agent.learn_A:
      qA = jtu.tree_map(
        lambda qa, o, m: qa + delta_A(posterior_beliefs, o, A_deps[m], num_obs[m]).sum(0), 
        qA, 
        obs, 
        list(range(len(num_obs)))
        )

    # learn B matrix
    conditional_beliefs = jtu.tree_map(
       lambda b, f: get_reverse_conditionals(b, [filter_beliefs[i] for i in B_deps[f]]),
       agent.B, 
       list(range(len(agent.B))) 
    )
    post_marg = get_marginals(posterior_beliefs)
    acts =  [acts[..., i] for i in range(acts.shape[-1])]

    qB = jtu.tree_map(
       lambda qb, pb, cb, a, nc: qb + delta_B(pb, cb, a, nc).sum(0),
       qB,
       post_marg,
       conditional_beliefs,
       acts,
       agent.num_controls  
    )

    # compute posterior beliefs for the next time step
    get_transition = lambda cb, a: cb[..., a]
    conditional_beliefs = jtu.tree_map(
      lambda cb, a: vmap(get_transition)(cb, a), conditional_beliefs, acts
    )
    posterior_beliefs = get_reverse_predictive(posterior_beliefs, conditional_beliefs, B_deps)

    return (posterior_beliefs, qA, qB), None

  first_outcomes = jtu.tree_map(lambda x: x[..., 0], outcomes)
  outcomes = jtu.tree_map(lambda x: jnp.flipud(x.swapaxes(0, 1))[1:lag+1], outcomes)
  actions = jnp.flipud(actions.swapaxes(0, 1))[:lag]
  beliefs = jtu.tree_map(lambda x: jnp.flipud(jnp.moveaxis(x, 2, 0))[1:lag+1], beliefs)
  iters = (outcomes, actions, beliefs)
  (last_beliefs, qA, qB), _ = lax.scan(step_fn, (posterior_beliefs, qA, qB), iters)

  # update A with the first outcome 
  if agent.learn_A:
    qA = jtu.tree_map(
      lambda qa, o, m: qa + delta_A(last_beliefs, o, A_deps[m], num_obs[m]).sum(0), 
      qA, 
      first_outcomes, 
      list(range(len(num_obs)))
    )

  if qA is not None:
    E_qA = jtu.tree_map(lambda qa: qa / qa.sum(0), qA)
  else:
    E_qA = agent.A
  E_qB =jtu.tree_map(lambda qb: qb / qb.sum(0), qB)
  agent = eqx.tree_at(
    lambda x: (x.A, x.pA, x.B, x.pB), agent, (E_qA, qA, E_qB, qB), is_leaf=lambda x: x is None
  )

  return agent
class TestEnv:
    def __init__(self, num_agents, num_obs, prng_key=jr.PRNGKey(0)):
      self.num_obs = num_obs
      self.num_agents = num_agents
      self.key = prng_key
    
    def step(self, actions=None):
      # return a list of random observations for each agent or parallel realization (each entry in batch_dim)
      obs = [jr.randint(self.key, (self.num_agents,), 0, no) for no in self.num_obs]
      self.key, _ = jr.split(self.key)
      return obs
def update_agent_state(agent, env, args, key, outcomes, actions):
    beliefs = agent.infer_states(outcomes, args[0], past_actions=actions, qs_hist=args[1])
    # q_pi, _ = agent.infer_policies(beliefs)
    q_pi = jnp.ones((agent.batch_size, 6)) / 6
    batch_keys = jr.split(key, agent.batch_size)
    actions = agent.sample_action(q_pi, rng_key=batch_keys)

    outcomes = env.step(actions)
    outcomes = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), outcomes)
    args = agent.update_empirical_prior(actions, beliefs)
    args = (args[0], None)  # remove belief history from args
    latest_belief = jtu.tree_map(lambda x: x[:, 0], beliefs)

    return args, latest_belief, outcomes, actions

def evolve_trials(agent, env, batch_size, num_timesteps, prng_key=jr.PRNGKey(0)):

    def step_fn(carry, xs):
        actions = carry['actions']
        outcomes = carry['outcomes']
        key = carry['key']
        key, _key = jr.split(key)
        vect_uas = vmap(partial(update_agent_state, agent, env))
        keys = jr.split(_key, batch_size)
        args, beliefs, outcomes, actions = vect_uas(carry['args'], keys, outcomes, actions)
        output = {
           'args': args, 
           'outcomes': outcomes, 
           'actions': actions,
           'key': key
        }

        return output, {'beliefs': beliefs, 'actions': actions[..., 0, :], 'outcomes': outcomes}

   
    outcome_0  = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), env.step())
    outcome_0 = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), outcome_0)
    prior = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), agent.D)
    init = {
      'args': (prior, None),
      'outcomes': outcome_0,
      'actions': - jnp.ones((batch_size, 1, agent.policies.shape[-1]), dtype=jnp.int32),
      'key': prng_key
    }

    last, sequences = lax.scan(step_fn, init, jnp.arange(num_timesteps))
    sequences['outcomes'] = jtu.tree_map(
        lambda x, y: jnp.concatenate([jnp.expand_dims(x.squeeze(), 0), y.squeeze()]), 
        outcome_0, 
        sequences['outcomes']
      )

    return last, sequences

@partial(jit, static_argnums=(1, 2, 3, 4))
def training_step(agent, env, batch_size, num_timesteps, lag=1):
    output, sequences = evolve_trials(agent, env, batch_size, num_timesteps)
    args = output.pop('args')
    
    outcomes = jtu.tree_map(lambda x: x.swapaxes(0, 1), sequences['outcomes'])
    actions = sequences['actions'].swapaxes(0, 1)
    beliefs = jtu.tree_map(lambda x: jnp.moveaxis(x, [0, 2], [1, 1]), sequences['beliefs'])

    def update_beliefs(outcomes, actions, args):
        return agent.infer_states(outcomes, args[0], past_actions=actions, qs_hist=args[1])

    # update beliefs with the last action-outcome pair
    last_belief = vmap(update_beliefs)(
       output['outcomes'], 
       output['actions'],
       args
      )

    beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], -2), beliefs, last_belief)
    # agent, beliefs, actions, outcomes = lax.stop_gradient((agent, beliefs, actions, outcomes))
    agent = learning(agent, beliefs, actions, outcomes, lag=lag)

    return agent
# define an agent and environment here
batch_size = 16
num_agents = 1

num_pixels = 32
# y_pos paddle 1, y_pos paddle 2, (x_pos, y_pos) ball
num_obs = [num_pixels, num_pixels, num_pixels, num_pixels]
num_states = [num_pixels, num_pixels, num_pixels, num_pixels, 96]
num_controls = [1, 1, 1, 1, 6]
num_blocks = 1
num_timesteps = 25

action_lists = [jnp.zeros(6, dtype=jnp.int32)] * 4
action_lists += [jnp.arange(6, dtype=jnp.int32)]

policies = jnp.expand_dims(jnp.stack(action_lists, -1), -2)
num_policies = len(policies)

A_dependencies = [[0], [1], [2], [3]]
B_dependencies = [[0, 4], [1, 4], [2, 4], [3, 4], [4]]

A_np = [np.eye(o) for o in num_obs]
B_np = list(random_B_matrix(num_states=num_states, num_controls=num_controls, B_factor_list=B_dependencies))
A = jtu.tree_map(lambda x: jnp.broadcast_to(x, (num_agents,) + x.shape), A_np)
B = jtu.tree_map(lambda x: jnp.broadcast_to(x, (num_agents,) + x.shape), B_np)
C = [jnp.zeros((num_agents, no)) for no in num_obs]
D = [jnp.ones((num_agents, ns)) / ns for ns in num_states]
E = jnp.ones((num_agents, num_policies )) / num_policies

pA = None # jtu.tree_map(lambda x: jnp.broadcast_to(jnp.ones_like(x), (num_agents,) + x.shape), A_np)
pB = jtu.tree_map(lambda x: jnp.broadcast_to(jnp.ones_like(x), (num_agents,) + x.shape), B_np)

agents = AIFAgent(A, B, C, D, E, pA, pB, learn_A=False, policies=policies, A_dependencies=A_dependencies, B_dependencies=B_dependencies, use_param_info_gain=True, inference_algo='fpi', sampling_mode='marginal', action_selection='deterministic', num_iter=8)
env = TestEnv(num_agents, num_obs)
agents = training_step(agents, env, batch_size, num_timesteps, lag=25)
# agents = lax.stop_gradient(agents)
# %timeit training_step(agents, env, batch_size, num_timesteps, lag=25).A[0].block_until_ready()
# define an agent and environment here
batch_size = 16
num_agents = 1

num_pixels = 32
# y_pos paddle 1, y_pos paddle 2, (x_pos, y_pos) ball
num_obs = [num_pixels, num_pixels, num_pixels, num_pixels]
num_states = [num_pixels, 2, num_pixels, 2, num_pixels, num_pixels, 24]
num_controls = [1, 6, 1, 6, 1, 1, 6]
num_blocks = 1
num_timesteps = 25

action_lists = [jnp.zeros(6, dtype=jnp.int32), jnp.arange(6, dtype=jnp.int32)] * 2
action_lists += [jnp.zeros(6, dtype=jnp.int32), jnp.zeros(6, dtype=jnp.int32), jnp.arange(6, dtype=jnp.int32)]

policies = jnp.expand_dims(jnp.stack(action_lists, -1), -2)
num_policies = len(policies)

A_dependencies = [[0], [2], [4], [5]]
B_dependencies = [[0, 1], [1], [2, 3], [3], [4, 6], [5, 6], [6]]

A_np = [np.eye(o) for o in num_obs]
B_np = list(random_B_matrix(num_states=num_states, num_controls=num_controls, B_factor_list=B_dependencies))
A = jtu.tree_map(lambda x: jnp.broadcast_to(x, (num_agents,) + x.shape), A_np)
B = jtu.tree_map(lambda x: jnp.broadcast_to(x, (num_agents,) + x.shape), B_np)
C = [jnp.zeros((num_agents, no)) for no in num_obs]
D = [jnp.ones((num_agents, ns)) / ns for ns in num_states]
E = jnp.ones((num_agents, num_policies )) / num_policies

pA = None # jtu.tree_map(lambda x: jnp.broadcast_to(jnp.ones_like(x), (num_agents,) + x.shape), A_np)
pB = jtu.tree_map(lambda x: jnp.broadcast_to(jnp.ones_like(x), (num_agents,) + x.shape), B_np)

agents = AIFAgent(A, B, C, D, E, pA, pB, learn_A=False, policies=policies, A_dependencies=A_dependencies, B_dependencies=B_dependencies, use_param_info_gain=True, inference_algo='fpi', sampling_mode='marginal', action_selection='deterministic', num_iter=8)
env = TestEnv(num_agents, num_obs)
agents = training_step(agents, env, batch_size, num_timesteps, lag=25)
# %timeit training_step(agents, env, batch_size, num_timesteps, lag=25).A[0].block_until_ready()
