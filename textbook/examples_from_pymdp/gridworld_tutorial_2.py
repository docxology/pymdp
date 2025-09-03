# This is needed (on my machine at least) due to weird python import issues
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
print(path)
module_path = str(path.parent) + '/'
sys.path.append(module_path)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from pymdp import maths, utils
from pymdp.maths import spm_log_single as log_stable # @NOTE: we use the `spm_log_single` helper function from the `maths` sub-library of pymdp. This is a numerically stable version of np.log()
from pymdp import control
print("imports loaded")
state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5:(2,1), 6: (0,2), 7:(1,2), 8:(2,2)}

A = np.eye(9)
def plot_beliefs(Qs, title=""):
    #values = Qs.values[:, 0]
    plt.grid(zorder=0)
    plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
    plt.xticks(range(Qs.shape[0]))
    plt.title(title)
    plt.show()
    
labels = [state_mapping[i] for i in range(A.shape[1])]
def plot_likelihood(A):
    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(A, xticklabels = labels, yticklabels = labels, cbar = False)
    plt.title("Likelihood distribution (A)")
    plt.show()
    
def plot_empirical_prior(B):
    fig, axes = plt.subplots(3,2, figsize=(8, 10))
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'STAY']
    count = 0
    for i in range(3):
        for j in range(2):
            if count >= 5:
                break
                
            g = sns.heatmap(B[:,:,count], cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i,j])

            g.set_title(actions[count])
            count += 1
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()
    
def plot_transition(B):
    fig, axes = plt.subplots(2,3, figsize = (15,8))
    a = list(actions.keys())
    count = 0
    for i in range(dim-1):
        for j in range(dim):
            if count >= 5:
                break 
            g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
            g.set_title(a[count])
            count +=1 
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()
# A matrix
A = np.eye(9)
plot_likelihood(A)
# construct B matrix

P = {}
dim = 3
actions = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'STAY':4}

for state_index, xy_coordinates in state_mapping.items():
    P[state_index] = {a : [] for a in range(len(actions))}
    x, y = xy_coordinates

    '''if your y-coordinate is all the way at the top (i.e. y == 0), you stay in the same place -- otherwise you move one upwards (achieved by subtracting 3 from your linear state index'''
    P[state_index][actions['UP']] = state_index if y == 0 else state_index - dim 

    '''f your x-coordinate is all the way to the right (i.e. x == 2), you stay in the same place -- otherwise you move one to the right (achieved by adding 1 to your linear state index)'''
    P[state_index][actions["RIGHT"]] = state_index if x == (dim -1) else state_index+1 

    '''if your y-coordinate is all the way at the bottom (i.e. y == 2), you stay in the same place -- otherwise you move one down (achieved by adding 3 to your linear state index)'''
    P[state_index][actions['DOWN']] = state_index if y == (dim -1) else state_index + dim 

    ''' if your x-coordinate is all the way at the left (i.e. x == 0), you stay at the same place -- otherwise, you move one to the left (achieved by subtracting 1 from your linear state index)'''
    P[state_index][actions['LEFT']] = state_index if x == 0 else state_index -1 

    ''' Stay in the same place (self explanatory) '''
    P[state_index][actions['STAY']] = state_index


num_states = 9
B = np.zeros([num_states, num_states, len(actions)])
for s in range(num_states):
    for a in range(len(actions)):
        ns = int(P[s][a])
        B[ns, s, a] = 1

plot_transition(B)
class GridWorldEnv():
    
    def __init__(self,A,B):
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        print("B:", B.shape)
        self.state = np.zeros(9)
        # start at state 3
        self.state[2] = 1
    
    def step(self,a):
        self.state = np.dot(self.B[:,:,a], self.state)
        obs = utils.sample(np.dot(self.A, self.state))
        return obs

    def reset(self):
        self.state =np.zeros(9)
        self.state[2] =1 
        obs = utils.sample(np.dot(self.A, self.state))
        return obs
    
env = GridWorldEnv(A,B)
def KL_divergence(q,p):
    return np.sum(q * (log_stable(q) - log_stable(p)))
def compute_free_energy(q,A, B):
    return np.sum(q * (log_stable(q) - log_stable(A) - log_stable(B)))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def perform_inference(likelihood, prior):
    return softmax(log_stable(likelihood) + log_stable(prior))
# setup initial prior beliefs -- uncertain -- completely unknown which state it is in
Qs = np.ones(9) * 1/9
plot_beliefs(Qs)
# C matrix -- desires

REWARD_LOCATION = 7
reward_state = state_mapping[REWARD_LOCATION]
print(reward_state)

C = np.zeros(num_states)
C[REWARD_LOCATION] = 1. 
print(C)
plot_beliefs(C)
def evaluate_policy(policy, Qs, A, B, C):
    # initialize expected free energy at 0
    G = 0

    # loop over policy
    for t in range(len(policy)):

        # get action entailed by the policy at timestep `t`
        u = int(policy[t])

        # work out expected state, given the action
        Qs_pi = B[:,:,u].dot(Qs)

        # work out expected observations, given the action
        Qo_pi = A.dot(Qs_pi)

        # get entropy
        H = - (A * log_stable(A)).sum(axis = 0)

        # get predicted divergence
        # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
        divergence = KL_divergence(Qo_pi, C)
        
        # compute the expected uncertainty or ambiguity 
        uncertainty = H.dot(Qs_pi)

        # increment the expected free energy counter for the policy, using the expected free energy at this timestep
        G += (divergence + uncertainty)

    return -G
def infer_action(Qs, A, B, C, n_actions, policies):
    
    # initialize the negative expected free energy
    neg_G = np.zeros(len(policies))

    # loop over every possible policy and compute the EFE of each policy
    for i, policy in enumerate(policies):
        neg_G[i] = evaluate_policy(policy, Qs, A, B, C)

    # get distribution over policies
    Q_pi = maths.softmax(neg_G)

    # initialize probabilites of control states (convert from policies to actions)
    Qu = np.zeros(n_actions)

    # sum probabilites of control states or actions 
    for i, policy in enumerate(policies):
        # control state specified by policy
        u = int(policy[0])
        # add probability of policy
        Qu[u] += Q_pi[i]

    # normalize action marginal
    utils.norm_dist(Qu)

    # sample control from action marginal
    u = utils.sample(Qu)

    return u
# number of time steps
T = 10

#n_actions = env.n_control
n_actions = 5

# length of policies we consider
policy_len = 4

# this function generates all possible combinations of policies
policies = control.construct_policies([B.shape[0]], [n_actions], policy_len)

# reset environment
o = env.reset()

# loop over time
for t in range(T):

    # infer which action to take
    a = infer_action(Qs, A, B, C, n_actions, policies)
    
    # perform action in the environment and update the environment
    o = env.step(int(a))
    
    # infer new hidden state (this is the same equation as above but with PyMDP functions)
    likelihood = A[o,:]
    prior = B[:,:,int(a)].dot(Qs)

    Qs = maths.softmax(log_stable(likelihood) + log_stable(prior))
    
    print(Qs.round(3))
    plot_beliefs(Qs, "Beliefs (Qs) at time {}".format(t))
