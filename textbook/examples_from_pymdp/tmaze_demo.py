import os
import sys
import pathlib
import numpy as np
import copy

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood
from pymdp import utils
from pymdp.envs import TMazeEnv
reward_probabilities = [0.98, 0.02] # probabilities used in the original SPM T-maze demo
env = TMazeEnv(reward_probs = reward_probabilities)
A_gp = env.get_likelihood_dist()
plot_likelihood(A_gp[1][:,:,0],'Reward Right')
plot_likelihood(A_gp[1][:,:,1],'Reward Left')
plot_likelihood(A_gp[2][:,3,:],'Cue Mapping')
B_gp = env.get_transition_dist()
plot_likelihood(B_gp[1][:,:,0],'Reward Condition Transitions')
plot_likelihood(B_gp[0][:,:,0],'Transition likelihood for "Move to Center"')
plot_likelihood(B_gp[0][:,:,1],'Transition likelihood for "Move to Right Arm"')
plot_likelihood(B_gp[0][:,:,2],'Transition likelihood for "Move to Left Arm"')
plot_likelihood(B_gp[0][:,:,3],'Transition likelihood for "Move to Cue Location"')
A_gm = copy.deepcopy(A_gp) # make a copy of the true observation likelihood to initialize the observation model
B_gm = copy.deepcopy(B_gp) # make a copy of the true transition likelihood to initialize the transition model
controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable
agent = Agent(A=A_gm, B=B_gm, control_fac_idx=controllable_indices)
plot_beliefs(agent.D[0],"Beliefs about initial location")
plot_beliefs(agent.D[1],"Beliefs about reward condition")
agent.D[0] = utils.onehot(0, agent.num_states[0])
plot_beliefs(agent.D[0],"Beliefs about initial location")
agent.C[1][1] = 3.0
agent.C[1][2] = -3.0
plot_beliefs(agent.C[1],"Prior beliefs about observations")
T = 5 # number of timesteps

obs = env.reset() # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']
msg = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
print(msg.format(reward_conditions[env.reward_condition], location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

for t in range(T):
    qx = agent.infer_states(obs)

    q_pi, efe = agent.infer_policies()

    action = agent.sample_action()

    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    obs = env.step(action)

    msg = """[Step {}] Observation: [{},  {}, {}]"""
    print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))
plot_likelihood(A_gp[2][:,:,0],'Cue Observations when condition is Reward on Right, for Different Locations')
plot_likelihood(A_gp[2][:,:,1],'Cue Observations when condition is Reward on Left, for Different Locations')
plot_beliefs(qx[1],"Final posterior beliefs about reward condition")
