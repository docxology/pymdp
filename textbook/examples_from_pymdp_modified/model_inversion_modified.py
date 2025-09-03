# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.

import jax
import jax.numpy as jnp

import seaborn as sns
import matplotlib.pyplot as plt

from pymdp.jax.agent import Agent as AIFAgent
from pymdp.envs import TMazeEnv
from pybefit import ModelInference

def param_transform(z):
    init = {}  # define some initial values of random variables that should be infered
    params = {}  # define parameters that should be infered
    return init, params


# we could simplify the interface so that AIFAgent class is constructed as 
# aif_agent = AIFAgent(init_variables, params, options)

# define some static options for the AIFAgent class
agent_options = {

}

# define properties of inference
inference_options = {
    # e.g. method can be svi or nuts
    'method': 'SVI',
    # different forms of the parameteric prior, such as, NormalGamma, NormalHorseshoe, NormalRegularizedHorseshoe
    'prior': 'NormalGamma',
    # hierachical inference with group level 
    'type': 'Hierarchical',

}

inference = ModelInference(AIFAgent, agent_options, inference_options)

num_samples = 1000
max_iterations = 1000
tolerance = 1e-3
# optimizer options
opts = {
    'learning_rate': 1e-3
}

inference.fit(behavioural_data, num_samples, max_iterations, tolerance, opts)
def plot_beliefs(belief_dist, title=""):
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title)
    plt.show()
    
def plot_likelihood(A, title=""):
    ax = sns.heatmap(A, cmap="OrRd", linewidth=2.5)
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
    plt.title(title)
    plt.show()
reward_probabilities = [0.98, 0.02] # probabilities used in the original SPM T-maze demo
env = TMazeEnv(reward_probs = reward_probabilities)
A_gp = env.get_likelihood_dist()
plot_likelihood(A_gp[1][:, :, 0],'Reward Right')
plot_likelihood(A_gp[1][:, :, 1],'Reward Left')
plot_likelihood(A_gp[2][:, 3, :],'Cue Mapping')
B_gp = env.get_transition_dist()
plot_likelihood(B_gp[1][:, :, 0],'Reward Condition Transitions')
plot_likelihood(B_gp[0][:,:,0],'Transition likelihood for "Move to Center"')
plot_likelihood(B_gp[0][:,:,1],'Transition likelihood for "Move to Right Arm"')
plot_likelihood(B_gp[0][:,:,2],'Transition likelihood for "Move to Left Arm"')
plot_likelihood(B_gp[0][:,:,3],'Transition likelihood for "Move to Cue Location"')
num_agents = 50  # number of different agents 
A_gm = [jnp.broadcast_to(jnp.array(a), (num_agents,) + a.shape) for a in A_gp]  # map the true observation likelihood to jax arrays
B_gm = [jnp.broadcast_to(jnp.array(b), (num_agents,) + b.shape) for b in B_gp]  # map the true transition likelihood to jax arrays
D_gm = [jnp.broadcast_to(jnp.array([1., 0., 0., 0.]), (num_agents, 4)), jnp.broadcast_to(jnp.array([.5, .5]), (num_agents, 2))]
C_gm = [jnp.zeros((num_agents, 4)), jnp.broadcast_to(jnp.array([0., -3., 3.]), (num_agents, 3)),jnp.zeros((num_agents, 2))]
E_gm = jnp.ones((num_agents, 4))
controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable
agent = Agent(A_gm, B_gm, C_gm, D_gm, E_gm, control_fac_idx=controllable_indices)
policies = jnp.stack(agent.policies)
print(policies.shape)
print(policies.dtype)
import jax.tree_util as jtu

vals, tree = jtu.tree_flatten(agent)

print(tree)
T = 5 # number of timesteps

emp_prior = D_gm
_obs = env.reset() # reset the environment and get an initial observation
obs = jnp.broadcast_to(jnp.array(_obs), (num_agents, len(_obs)))

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']
msg = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
print(msg.format(reward_conditions[env.reward_condition], location_observations[_obs[0]], reward_observations[_obs[1]], cue_observations[_obs[2]]))

measurements = {'actions': [], 'outcomes': [obs]}
for t in range(T):
    qs = agent.infer_states(obs, emp_prior)

    q_pi, efe = agent.infer_policies(qs)

    actions = agent.sample_action(q_pi)
    emp_prior = agent.update_empirical_prior(actions, qs)

    measurements["actions"].append( actions )
    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(actions[0, 0])]))

    obs = []
    for a in actions:
        obs.append( jnp.array(env.step(list(a))) )
    obs = jnp.stack(obs)
    measurements["outcomes"].append(obs)

    msg = """[Step {}] Observation: [{},  {}, {}]"""
    print(msg.format(t, location_observations[obs[0, 0]], reward_observations[obs[0, 1]], cue_observations[obs[0, 2]]))
    
measurements['actions'] = jnp.stack(measurements['actions']).astype(jnp.int32)
measurements['outcomes'] = jnp.stack(measurements['outcomes'])

measurements['outcomes'] = measurements['outcomes'][None, :T]
measurements['actions'] = measurements['actions'][None]
plot_beliefs(qs[1][0],"Final posterior beliefs about reward condition")
import numpyro as npyro
from jax import random
from numpyro.infer import Predictive
from pymdp.jax.likelihoods import aif_likelihood, evolve_trials

print(measurements['outcomes'].shape)
print(measurements['actions'].shape)

Nb, Nt, Na, _ = measurements['actions'].shape

xs = {'outcomes': measurements['outcomes'][0], 'actions': measurements['actions'][0]}
evolve_trials(agent, xs)
%timeit evolve_trials(agent, xs)

rng_key = random.PRNGKey(0)

with npyro.handlers.seed(rng_seed=0):
    aif_likelihood(Nb, Nt, Na, measurements, agent)

%timeit pred_samples = Predictive(aif_likelihood, num_samples=11)(rng_key, Nb, Nt, Na, measurements, agent)
print(pred_samples.keys())
import numpyro as npyro
import numpyro.distributions as dist
from jax import nn, lax, vmap

@vmap
def trans_params(z):

    a = nn.sigmoid(z[0])
    lam = nn.softplus(z[1])
    d = nn.sigmoid(z[2])

    A = lax.stop_gradient([jnp.array(x) for x in list(A_gp)])

    middle_matrix1 = jnp.array([[0., 0.], [a, 1-a], [1-a, a]])
    middle_matrix2 = jnp.array([[0., 0.], [1-a, a], [a, 1-a]])

    side_vector = jnp.stack([jnp.array([1.0, 0., 0.]), jnp.array([1.0, 0., 0.])], -1)

    A[1] = jnp.stack([side_vector, middle_matrix1, middle_matrix2, side_vector], -2)
    
    C = [
        jnp.zeros(4),
        lam * jnp.array([0., 1., -1.]),
        jnp.zeros(2)
    ]

    D = [nn.one_hot(0, 4), jnp.array([d, 1-d])]

    E = jnp.ones(4)/4

    params = {
        'A': A,
        'B': lax.stop_gradient([jnp.array(x) for x in list(B_gp)]),
        'C': C,
        'D': D,
        'E': E
    }

    return  params, a, lam, d
def model(data, num_blocks, num_steps, num_agents, num_params=3):
    with npyro.plate('agents', num_agents):
        z = npyro.sample('z', dist.Normal(0., 1.).expand([num_params]).to_event(1))
        params, a, lmbd, d = trans_params(z)
        # register parameter values
        npyro.deterministic('a', a)
        npyro.deterministic('lambda', lmbd)
        npyro.deterministic('d', d)

    agents = Agent(
        params['A'], 
        params['B'], 
        params['C'], 
        params['D'], 
        params['E'], 
        control_fac_idx=controllable_indices
    )

    aif_likelihood(num_blocks, num_steps, num_agents, data, agents)
    
with npyro.handlers.seed(rng_seed=101111):
    model(measurements, Nb, Nt, Na)

%timeit pred_samples = Predictive(model, num_samples=11)(rng_key, measurements, Nb, Nt, Na)
print(pred_samples.keys())
# inference with NUTS and MCMC
from numpyro.infer import NUTS, MCMC
from numpyro.infer import init_to_feasible, init_to_sample

rng_key = random.PRNGKey(0)
kernel = NUTS(model, init_strategy=init_to_feasible)

mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, progress_bar=False)

rng_key, _rng_key = random.split(rng_key)
mcmc.run(_rng_key, measurements, Nb, Nt, Na)
import arviz as az
az.style.use('arviz-darkgrid')

coords = {
    'idx': jnp.arange(num_agents),
    'vars': jnp.arange(3), 
}
dims = {'z': ["idx", "vars"], 'd': ["idx"], 'lambda': ["idx"], 'a': ["idx"]}
data_kwargs = {
    "dims": dims,
    "coords": coords,
}
data_mcmc = az.from_numpyro(posterior=mcmc, **data_kwargs)
az.plot_trace(data_mcmc, kind="rank_bars", var_names=['d', 'lambda', 'a']);

#TODO: maybe plot real values on top of samples from the posterior
# inferenace with SVI and autoguides
import optax
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal

num_iters = 1000
guide = AutoMultivariateNormal(model)
optimizer = npyro.optim.optax_to_numpyro(optax.chain(optax.adabelief(1e-3)))
svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=10))
rng_key, _rng_key = random.split(rng_key)
svi_res = svi.run(_rng_key, num_iters, measurements, Nb, Nt, Na, progress_bar=False)
plt.figure(figsize=(16,5))
plt.plot(svi_res.losses)
plt.ylabel('Variational free energy');
plt.xlabel('iter step');
rng_key, _rng_key = random.split(rng_key)
pred = Predictive(
    model, 
    guide=guide, 
    params=svi_res.params, 
    num_samples=1000, 
    return_sites=["d", "a", "lambda"]
)
post_sample = pred(_rng_key, measurements, Nb, Nt, Na)

for key in post_sample:
    post_sample[key] = jnp.expand_dims(post_sample[key], 0)

data_svi = az.convert_to_inference_data(post_sample, group="posterior", **data_kwargs)
axes = az.plot_forest(
    [data_mcmc, data_svi],
    model_names = ["nuts", "svi"],
    kind='forestplot',
    var_names=['d', 'lambda', 'a'],
    coords={"idx": 0},
    combined=True,
    figsize=(20, 6)
)
