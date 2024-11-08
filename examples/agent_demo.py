import numpy as np
from pymdp.agent import Agent
from pymdp import utils
from pymdp.gnn.gnn_matrix_factory import GNNMatrixFactory
import os

# Initialize the GNN factory with the model file
gnn_file = "pymdp/gnn/models/ABCD_demo.gnn"
factory = GNNMatrixFactory(gnn_file)

# Create sandbox directory if it doesn't exist
sandbox_dir = "sandbox"
os.makedirs(sandbox_dir, exist_ok=True)

# Get matrices from the GNN model
matrices = factory.create_matrices()

# Extract matrices and parameters
A = matrices['A']
B = matrices['B']
C = matrices['C']
D = matrices['D']
control_fac_idx = matrices['control_fac_idx']

# Create the agent
agent = Agent(A=A, B=B, C=C, D=D, control_fac_idx=control_fac_idx)

# Set up simulation parameters
T = 10  # Number of timesteps

# Initialize state and observation
s = [0, 0]  # Start at first location in first context
o = [0, 0, 0]  # Initial observations (no hint, neutral reward, first location)

# Run simulation
for t in range(T):
    print(f"\nTimestep {t}")
    print(f"Observations: Hint={o[0]}, Reward={o[1]}, Location={o[2]}")

    # Infer states
    qx = agent.infer_states(o)
    print(f"Beliefs: Location={qx[0].round(3)}, Context={qx[1].round(3)}")

    # Plan and act
    agent.infer_policies()
    action = agent.sample_action()
    
    # State transition
    for f, s_i in enumerate(s):
        if f in control_fac_idx:  # Only location is controllable
            s[f] = utils.sample(B[f][:, s_i, int(action[f])])
        else:  # Context stays the same
            s[f] = s_i

    # Generate observations
    for g in range(len(o)):
        o[g] = utils.sample(A[g][:, s[0], s[1]])
    
    print(f"Action: {action[0]} / State: Location={s[0]}, Context={s[1]}")
