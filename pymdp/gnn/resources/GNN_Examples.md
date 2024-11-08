# GNN Examples

## Basic Models

### 1. Step-by-Step POMDP
The simplest complete active inference model:

```json
{
    "modelName": "step_by_step",
    "modelType": ["Dynamic", "POMDP"],
    "stateSpace": {
        "factors": [
            {
                "name": "position",
                "num_states": 3,
                "controllable": true
            }
        ]
    }
}
```

### 2. Two-Factor Model
Model with multiple state factors:

```json
{
    "modelName": "two_factor",
    "modelType": ["Dynamic", "Hierarchical"],
    "stateSpace": {
        "factors": [
            {
                "name": "location",
                "num_states": 4,
                "controllable": true
            },
            {
                "name": "context",
                "num_states": 2,
                "controllable": false
            }
        ]
    }
}
```

## Running Examples

### Basic Experiment
```python
from pymdp.gnn import GNNRunner
from pymdp.environments import GridWorldEnv

# Load and run model
runner = GNNRunner("models/step_by_step.gnn")
matrices = runner.generate_matrices()
runner.run_experiment(GridWorldEnv, T=100)
```

### Custom Environment
```python
from pymdp.agentmaker import Environment

class CustomEnv(Environment):
    def step(self, action):
        # Implement dynamics
        return observation
        
runner.run_experiment(CustomEnv, T=50)
```

## Matrix Examples

### A-Matrix (Observation Model)
```json
"matrices": {
    "A": [
        {
            "modality": 0,
            "values": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            "description": "Perfect observation mapping"
        }
    ]
}
```

### B-Matrix (Transition Model)
```json
"B": [
    {
        "factor": 0,
        "values": [
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9]
            ]
        ],
        "description": "Noisy state transitions"
    }
]
```

## Visualization Examples

### Node Layout
```json
"visualization": {
    "nodes": {
        "position": {
            "label": "Position",
            "equations": ["s_{t+1} = T(s_t, a_t)"]
        }
    },
    "layout": {
        "type": "hierarchical",
        "direction": "TB"
    }
}
```

### Markdown Documentation
```json
"markdown": {
    "title": "# Model Documentation",
    "description": [
        "This model implements:",
        "1. Feature A",
        "2. Feature B"
    ],
    "equations": [
        "## Key Equations",
        "$F = \\mathbb{E}_{Q(s)}[\\log Q(s) - \\log P(o,s)]$"
    ]
}
```

## Output Examples

### Generated Directory Structure
```
sandbox/
└── step_by_step_20240315_123456/
    ├── matrices/
    │   ├── A_matrices/
    │   ├── B_matrices/
    │   └── C_matrices/
    ├── visualizations/
    │   ├── model_graph.png
    │   └── belief_evolution.png
    └── exports/
        ├── model.md
        └── summary.json
```

### Experiment Results
```python
# Access results
results = runner.get_results()
print(f"Final Free Energy: {results['free_energy'][-1]}")
print(f"Action Sequence: {results['actions']}")

# Generate plots
runner.plot_belief_evolution()
runner.plot_free_energy()
```

## Common Patterns

1. **State Factor Definition**
   ```json
   "factors": [
       {
           "name": "state_name",
           "num_states": N,
           "controllable": bool,
           "initial_beliefs": [probabilities]
       }
   ]
   ```

2. **Matrix Definition**
   ```json
   "matrix_type": [
       {
           "index": int,
           "values": array,
           "description": "string"
       }
   ]
   ```

3. **Visualization Setup**
   ```json
   "visualization": {
       "nodes": {node_dict},
       "connections": [connection_list],
       "layout": {layout_options}
   }
   ```
