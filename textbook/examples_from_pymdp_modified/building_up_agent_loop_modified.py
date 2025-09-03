# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random as jr
from pymdp.jax.agent import Agent as AIFAgent
from pymdp.utils import random_A_matrix, random_B_matrix
def scan(f, init, xs, length=None, axis=0):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    if y is not None:
       ys.append(y)
  
  ys = None if len(ys) < 1 else jtu.tree_map(lambda *x: jnp.stack(x,axis=axis), *ys)

  return carry, ys

def evolve_trials(agent, env, block_idx, num_timesteps, prng_key=jr.PRNGKey(0)):

    batch_keys = jr.split(prng_key, batch_size)
    def step_fn(carry, xs):
        actions = carry['actions']
        outcomes = carry['outcomes']
        beliefs = agent.infer_states(outcomes, carry['args'][0], past_actions=actions, qs_hist=carry['args'][1])
        q_pi, _ = agent.infer_policies(beliefs)
        actions_t = agent.sample_action(q_pi, rng_key=batch_keys)

        outcome_t = env.step(actions_t)
        outcomes = jtu.tree_map(
           lambda prev_o, new_o: jnp.concatenate([prev_o, jnp.expand_dims(new_o, -1)], -1), outcomes, outcome_t
          )

        if actions is not None:
          actions = jnp.concatenate([actions, jnp.expand_dims(actions_t, -2)], -2)
        else:
          actions = jnp.expand_dims(actions_t, -2)

        args = agent.update_empirical_prior(actions_t, beliefs)

        ### @ NOTE !!!!: Shape of policy_probs = (num_blocks, num_trials, batch_size, num_policies) if scan axis = 0, but size of `actions` will 
        ### be (num_blocks, batch_size, num_trials, num_controls) -- so we need to 1) swap axes to both to have the same first three dimensiosn aligned,
        # 2) use the action indices (the integers stored in the last dimension of `actions`) to index into the policy_probs array
        
        # args = (pred_{t+1}, [post_1, post_{2}, ..., post_{t}])
        # beliefs =  [post_1, post_{2}, ..., post_{t}]
        return {'args': args, 'outcomes': outcomes, 'beliefs': beliefs, 'actions': actions}, {'policy_probs': q_pi}

   
    outcome_0  = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), env.step())
    # qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D) # add a time dimension to the initial state prior
    init = {
       'args': (agent.D, None,),
       'outcomes': outcome_0, 
       'beliefs': [],
       'actions': None
    }
    last, q_pis_ = scan(step_fn, init, range(num_timesteps), axis=1)

    return last, q_pis_, env

def step_fn(carry, block_idx):
    agent, env = carry
    output, q_pis_, env = evolve_trials(agent, env, block_idx, num_timesteps)
    args = output.pop('args')
    output['beliefs'] = agent.infer_states(output['outcomes'], args[0], past_actions=output['actions'], qs_hist=args[1])
    output.update(q_pis_)

    # How to deal with contiguous blocks of trials? Two options we can imagine: 
    # A) you use final posterior (over current and past timesteps) to compute the smoothing distribution over qs_{t=0} and update pD, and then pass pD as the initial state prior ($D = \mathbb{E}_{pD}[qs_{t=0}]$);
    # B) we don't assume that blocks 'reset time', and are really just adjacent chunks of one long sequence, so you set the initial state prior to be the final output (`output['beliefs']`) passed through
    # the transition model entailed by the action taken at the last timestep of the previous block.
    # print(output['beliefs'].shape)
    agent = agent.learning(**output)
    
    return (agent, env), output

# define an agent and environment here
batch_size = 10
num_obs = [3, 3]
num_states = [3, 3]
num_controls = [2, 2]
num_blocks = 2
num_timesteps = 5

A_np = random_A_matrix(num_obs=num_obs, num_states=num_states)
B_np = random_B_matrix(num_states=num_states, num_controls=num_controls)
A = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), list(A_np))
B = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), list(B_np))
C = [jnp.zeros((batch_size, no)) for no in num_obs]
D = [jnp.ones((batch_size, ns)) / ns for ns in num_states]
E = jnp.ones((batch_size, 4 )) / 4 

pA = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), list(A_np))
pB = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), list(B_np))

class TestEnv:
    def __init__(self, num_obs, prng_key=jr.PRNGKey(0)):
      self.num_obs=num_obs
      self.key = prng_key
    def step(self, actions=None):
      # return a list of random observations for each agent or parallel realization (each entry in batch_dim)
      obs = [jr.randint(self.key, (batch_size,), 0, no) for no in self.num_obs]
      self.key, _ = jr.split(self.key)
      return obs

agents = AIFAgent(A, B, C, D, E, pA, pB, use_param_info_gain=True, use_inductive=False, inference_algo='fpi', sampling_mode='marginal', action_selection='stochastic')
env = TestEnv(num_obs)
init = (agents, env)
(agents, env), sequences = scan(step_fn, init, range(num_blocks) )
print(sequences['policy_probs'].shape)
print(sequences['actions'][0][0][0])
print(agents.A[0].shape)
print(agents.B[0].shape)
# def loss_fn(agents):
#     env = TestEnv(num_obs)
#     init = (agents, env)
#     (agents, env), sequences = scan(step_fn, init, range(num_blocks)) 

#     return jnp.sum(jnp.log(sequences['policy_probs']))

# dLoss_dAgents = jax.grad(loss_fn)(agents)
# print(dLoss_dAgents.A[0].shape)


# sequences = jtu.tree_map(lambda x: x.swapaxes(1, 2), sequences)

# NOTE: all elements of sequences will have dimensionality blocks, trials, batch_size, ...
