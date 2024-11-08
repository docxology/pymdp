# Biofirm Active Inference Agent

## Overview
Biofirm is an active inference agent that maintains homeostasis in a simple ecological environment. The agent uses the Generalized Notation Notation (GNN) notation to define its generative model and operates in a discrete state space with three states: LOW, HOMEO (homeostasis), and HIGH.

## Architecture

### Environment Model
- **State Space**: 3 discrete states (LOW, HOMEO, HIGH)
- **Observation Model**: Direct but noisy observations of states
- **Transition Model**: Action-dependent state transitions
- **Initial State Prior**: Centered on homeostatic state

### Agent Model
- **Generative Model**: Matches environment structure
- **Preference Model**: Strong preference for homeostatic state
- **Action Space**: 3 actions (DECREASE, MAINTAIN, INCREASE)
- **Inference**: Active inference with state and policy inference

## Usage

### Basic Execution
```bash
python3 Run_Biofirm.py
```

### Configuration
Edit `config.py` to modify:
- Number of timesteps
- Policy precision
- Inference parameters
- Tracking options
- Output settings

### Directory Structure
```
experiment_YYYYMMDD_HHMMSS/
├── 1_render/                 # Matrix generation
│   ├── matrices/            # Generated matrices
│   ├── config/             # Model configurations
│   └── matrix_visualizations/
├── 2_simulation/            # Simulation results
│   ├── data/               # Raw simulation data
│   ├── results/            # Processed results
│   ├── metrics/            # Performance metrics
│   └── checkpoints/        # Simulation states
└── 3_analysis/             # Visualizations
    ├── core_analysis/      # Basic metrics
    ├── policy_analysis/    # Action selection
    ├── homeostasis/        # State maintenance
    ├── belief_action/      # Decision making
    ├── convergence/        # Learning curves
    └── inference_cycle/    # Belief updates
```

### Key Files
- `Run_Biofirm.py`: Main experiment runner
- `Biofirm_Execute_GNN.py`: Simulation executor
- `Biofirm_Render_GNN.py`: Matrix generator
- `Biofirm_Visualize.py`: Analysis visualizer
- `config.py`: Configuration settings

## Analysis Outputs

### Core Analysis
- Belief dynamics
- Action frequencies
- State transitions
- Policy probabilities

### Advanced Analysis
- Homeostasis metrics
- Belief-action correlations
- Convergence analysis
- Inference cycle visualization

### Performance Metrics
- Time in homeostasis
- Control efficiency
- Belief accuracy
- Policy entropy

## Configuration Options

### Simulation Parameters
```python
EXPERIMENT_CONFIG = {
    'timesteps': 6000,
    'parameters': {
        'ecological_noise': 0.1,
        'controllability': 0.8,
        'policy_precision': 16.0,
        'inference_horizon': 1
    }
}
```

### Tracking Options
```python
'active_inference': {
    'track_beliefs': True,
    'track_free_energy': True,
    'track_policy_probs': True,
    'track_observations': True
}
```

## Dependencies
- Python 3.8+
- NumPy
- Matplotlib
- Seaborn (for visualizations)

## References
- Active Inference: Theory and Implementation
- GNN Specification Document
- Homeostatic Control in Active Inference

## Related Modules
- **Shapley Analysis**: See [README_Biofirm_Shapley.md](README_Biofirm_Shapley.md) for detailed coalition and contribution analysis
- **Matrix Generation**: See [Biofirm_Render_GNN.py](Biofirm_Render_GNN.py) for matrix generation details
- **Execution Engine**: See [Biofirm_Execute_GNN.py](Biofirm_Execute_GNN.py) for simulation execution