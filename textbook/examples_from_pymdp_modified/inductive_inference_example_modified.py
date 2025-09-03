# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.

from pymdp.jax import control
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, random, lax

from typing import List, Optional
from jaxtyping import Array
from jax import random as jr
# Set up a generative model
num_states = [5, 3]
num_controls = [2, 2]

# make some arbitrary policies (policy depth 3, 2 control factors)
policy_1 = jnp.array([[0, 1],
                         [1, 1],
                         [0, 0]])
policy_2 = jnp.array([[1, 0],
                        [0, 0],
                        [1, 1]])
policy_matrix = jnp.stack([policy_1, policy_2]) 

# observation modalities (isomorphic/identical to hidden states, just need to include for the need to include likleihood model)
num_obs = [5, 3]
num_factors = len(num_states)
num_modalities = len(num_obs)

# sample parameters of the model (A, B, C)
key = jr.PRNGKey(1)
factor_keys = jr.split(key, num_factors)

d = [0.1* jr.uniform(factor_key, (ns,)) for factor_key, ns in zip(factor_keys, num_states)]
qs_init = [jr.dirichlet(factor_key, d_f) for factor_key, d_f  in zip(factor_keys, d)]
A = [jnp.eye(no) for no in num_obs]

factor_keys = jr.split(factor_keys[-1], num_factors)
b = [jr.uniform(factor_keys[f], shape=(num_controls[f], num_states[f], num_states[f])) for f in range(num_factors)]
b_sparse = [jnp.where(b_f < 0.75, 1e-5, b_f) for b_f in b]
B = [jnp.swapaxes(jr.dirichlet(factor_keys[f], b_sparse[f]), 2, 0) for f in range(num_factors)]

modality_keys = jr.split(factor_keys[-1], num_modalities)
C = [nn.one_hot(jr.randint(modality_keys[m], shape=(1,), minval=0, maxval=num_obs[m]), num_obs[m]) for m in range(num_modalities)]

# trivial dependencies -- factor 1 drives modality 1, etc.
A_dependencies = [[0], [1]]
B_dependencies = [[0], [1]]
# generate random constraints (H vector)
factor_keys = jr.split(key, num_factors)
H = [jr.uniform(factor_key, (ns,)) for factor_key, ns in zip(factor_keys, num_states)]
H = [jnp.where(h < 0.75, 0., 1.) for h in H]

# depth and threshold for inductive planning algorithm. I made policy-depth equal to inductive planning depth, out of ignorance -- need to ask Tim or Tommaso about this
inductive_depth, inductive_threshold = 3, 0.5
I = control.generate_I_matrix(H, B, inductive_threshold, inductive_depth)
# evaluate Q(pi) and negative EFE using the inductive planning algorithm

E = jnp.ones(policy_matrix.shape[0])
pA = jtu.tree_map(lambda a: jnp.ones_like(a), A)
pB = jtu.tree_map(lambda b: jnp.ones_like(b), B)

q_pi, neg_efe = control.update_posterior_policies_inductive(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, use_utility=True, use_inductive=True, inductive_epsilon=1e-3)
