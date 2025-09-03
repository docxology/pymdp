# ! pip install inferactively-pymdp
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Latex

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from pymdp import utils, maths
## Define a quick helper function for printing arrays nicely (see https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75)
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
num_obs = 2 # dimensionality of observations (2 possible observation levels)
num_states = 2 # dimensionality of observations (2 possible hidden state levels)

observation = 0 # index representation
# observation = np.array([1, 0]) # one hot representation
# observation = utils.onehot(0, num_obs) # one hot representation, but use one of pymdp's utility functions to create a one-hot

prior = np.array([0.5, 0.5]) # simply set it by hand
# prior = np.ones(num_states) / num_states # create a uniform prior distribution by creating a vector of ones and dividing each by the number of states
# prior = utils.norm_dist(np.ones(num_states)) # use one of pymdp's utility functions to normalize any vector into a probability distribution

likelihood_dist = np.array([ [0.8, 0.2],
                        [0.2, 0.8] ])

if isinstance(observation, np.ndarray):
    print(f'Observation:{observation}\n')
else:
    print(f'Observation: {observation}\n')
print(f'Prior:{prior}\n')
print('Likelihood:')
matprint(likelihood_dist)

if isinstance(observation, np.ndarray): # use the matrix-vector version if your observation is a one-hot vector
    likelihood_s = likelihood_dist.T @ observation
elif isinstance(observation, int): # use the index version if your observation is an integer index
    likelihood_s = likelihood_dist[observation,:] 
display(Latex(f'$P(o=o_t|s):$'))
print(f'{likelihood_s}')
print('==================\n')
joint_prob = likelihood_s * prior # element-wise product of the likelihood of each hidden state, given the observation, with the prior probability assigned to each hidden state
display(Latex(f'$P(o = o_t,s) = P(o = o_t|s)P(s):$'))
print(f'{joint_prob}')
print('==================\n')
p_o = joint_prob.sum()
posterior = joint_prob / p_o # divide the joint by the marginal
display(Latex('$P(s|o = o_t) = \\frac{P(o = o_t,s)}{P(o=o_t)}$:'))
print(f'Posterior over hidden states: {posterior}')
print('==================')
surprise = - np.log(p_o)

display(Latex('$- \ln P(o=o_t)$:'))
print(f'Surprise: {surprise.round(3)}')
print('==================')
# to begin with, set our initial Q(s) equal to our prior distribution -- i.e. a flat/uninformative belief about hidden states
initial_qs = np.array([0.5, 0.5])

# redefine generative model and observation, for ease
observation = 0
likelihood_dist = np.array([[0.8, 0.2], [0.2, 0.8]])
prior = utils.norm_dist(np.ones(2))

# compute the joint or generative model using the factorization: P(o=o|s)P(s)
joint_prob = likelihood_dist[observation,:] * prior

# compute the variational free energy using the expected log difference formulation
initial_F = initial_qs.dot(np.log(initial_qs) - np.log(joint_prob))
## @NOTE: because of log-rules, `initial_F` can also be computed using the division inside the logarithm:
# initial_F = initial_qs.dot(np.log(initial_qs/joint_prob))

display(Latex('$\mathcal{F} = \mathbb{E}_{Q(s)}[\ln \\frac{Q(s)}{P(o, s)}] = \mathbb{E}_{Q(s)}[\ln Q(s) - \ln P(o, s)]:$'))
print(f'Variational free energy (F) = {initial_F.round(3)}')
print('==================')
final_qs = posterior.copy() # now we just assert that the approximate posterior is equal to the true posterior
final_F = final_qs.dot(np.log(final_qs) - np.log(joint_prob))

display(Latex('$\mathcal{F} = \mathbb{E}_{Q(s)}[\ln \\frac{Q(s)}{P(o, s)}] = \mathbb{E}_{Q(s)}[\ln Q(s) - \ln P(o, s)]:$'))
print(f'F = {final_F.round(3)}')
print('==================')
# compute the surprise (which we can do analytically for this simple generative model)
p_o = joint_prob.sum()
surprise = - np.log(p_o)
display(Latex('$-\ln P(o):$'))
print(f'{surprise.round(3)}')
print('==================\n')
import autograd.numpy as np_auto   # Thinly-wrapped version of Numpy that is auto-differentiable
from autograd import grad          # this is the function that we use to evaluate derivatives
from functools import partial
# define the variational free energy as a function of the approximate posterior, an observation, and the generative model
def vfe(qs, obs, likelihood, prior):
    """
    Quick docstring below on inputs
    Arguments:
    =========
    `qs` [1D np_auto.ndarray]: variational posterior over hidden states
    `obs` [int]: index of the observation
    `likelihood` [2D np_auto.ndarray]: likelihood distribution P(o|s), relating hidden states probabilistically to observations
    `prior` [1D np_auto.ndarray]: prior over hidden states
    """

    likelihood_s = likelihood[obs,:]

    joint = likelihood_s * prior

    vfe = qs @ (np_auto.log(qs) - np_auto.log(joint))

    return vfe

# initialize an observation, an initial variational posterior, a prior, and a likelihood matrix
obs = 0
init_qs = np_auto.array([0.5, 0.5])
prior = np_auto.array([0.5, 0.5])
likelihood_dist = np_auto.array([ [0.8, 0.2],
                        [0.2, 0.8] ])

# this use of `partial` creates a version of the vfe function that is a function of Qs only, 
# with the other parameters (the observation, the generative model) fixed as constant parameters
vfe_qs = partial(vfe, obs = obs, likelihood = likelihood_dist, prior = prior)

# By calling `grad` on a function, we get out a function that can be used to compute the gradients of the VFE with respect to its input (in our case, `qs`)
grad_vfe_qs = grad(vfe_qs)
# number of iterations of gradient descent to perform
n_iter = 40

qs_hist = np_auto.zeros((n_iter, 2))
qs_hist[0,:] = init_qs

vfe_hist = np_auto.zeros(n_iter)
vfe_hist[0] = vfe_qs(qs = init_qs)

learning_rate = 0.1 # learning rate to prevent gradient steps that are too big (overshooting)
for i in range(n_iter-1):   

    dFdqs = grad_vfe_qs(qs_hist[i,:])

    ln_qs = np_auto.log(qs_hist[i,:]) - learning_rate * dFdqs # transform qs to log-space to perform gradient descent
    qs_hist[i+1,:] = maths.softmax(ln_qs) # re-normalize to make it a proper, categorical Q(s) again

    vfe_hist[i+1] = vfe_qs(qs = qs_hist[i+1,:]) # measure final variational free energy
fig = plt.figure(figsize=(8,6))
plt.plot(vfe_hist)
plt.ylabel('$\\mathcal{F}$', fontsize = 22)
plt.xlabel("Iteration", fontsize = 22)
plt.xlim(0, n_iter)
plt.ylim(vfe_hist[-1], vfe_hist[0])
plt.title('Gradient descent on VFE', fontsize = 24)
final_qs, initial_F, final_F = qs_hist[-1,:], vfe_hist[0], vfe_hist[-1]

display(Latex('$\mathcal{F} = \mathbb{E}_{Q(s)}[\ln \\frac{Q(s)}{P(o, s)}] = \mathbb{E}_{Q(s)}[\ln Q(s) - \ln P(o, s)]:$'))
print(f'Initial F = {initial_F.round(3)}')
print('==================')

print('Final posterior:')
print(f'{final_qs.round(1)}') # note that because of numerical imprecision in the gradient descent (constant learning rate, etc.), the approximate posterior will not exactly be the optimal posterior
print('==================')
print(f'Final F = {vfe_qs(final_qs).round(3)}')
print('==================')
