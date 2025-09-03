# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5:(2,1), 6: (0,2), 7:(1,2), 8:(2,2)}
grid = np.zeros((3,3))
for linear_index, xy_coordinates in state_mapping.items():
    x, y = xy_coordinates
    grid[y,x] = linear_index # rows are the y-coordinate, columns are the x-coordinate -- so we index into the grid we'll be visualizing using '[y, x]'
fig = plt.figure(figsize = (3,3))
sns.set(font_scale=1.5)
sns.heatmap(grid, annot=True,  cbar = False, fmt='.0f', cmap='crest')
A = np.eye(9)
A
labels = [state_mapping[i] for i in range(A.shape[1])]
def plot_likelihood(A):
    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(A, xticklabels = labels, yticklabels = labels, cbar = False)
    plt.title("Likelihood distribution (A)")
    plt.show()
plot_likelihood(A)
state_mapping
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
P
num_states = 9
B = np.zeros([num_states, num_states, len(actions)])
for s in range(num_states):
    for a in range(len(actions)):
        ns = int(P[s][a])
        B[ns, s, a] = 1
        
B.shape
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
    
