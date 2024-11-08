"""Shared configuration for Biofirm experiments"""

from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default paths and filenames
DEFAULT_PATHS = {
    'env_gnn': PROJECT_ROOT / "pymdp/gnn/envs/gnn_env_biofirm.gnn",
    'agent_gnn': PROJECT_ROOT / "pymdp/gnn/models/gnn_agent_biofirm.gnn",
    'output_base': PROJECT_ROOT / "pymdp/agentmaker/sandbox/Biofirm"
}

# Default GNN models
DEFAULT_ENV_GNN = {
    "modelName": "biofirm_environment",
    "modelType": ["Environment", "Homeostatic"],
    "stateSpace": {
        "factors": ["EcologicalState"],
        "sizes": [3],
        "labels": [["LOW", "HOMEO", "HIGH"]]
    }
}

DEFAULT_AGENT_GNN = {
    "modelName": "biofirm_agent",
    "modelType": ["Active", "Homeostatic"],
    "stateSpace": {
        "factors": ["EcologicalState"],
        "sizes": [3],
        "labels": [["LOW", "HOMEO", "HIGH"]]
    }
}

# Default experiment configuration
EXPERIMENT_CONFIG = {
    'name': 'biofirm_exp',
    'timesteps': 1000,
    'render_only': False,
    'skip_visualization': False,
    'clean_output': False,
    'parameters': {
        'ecological_noise': 0.1,
        'controllability': 0.8,
        'policy_precision': 16.0,
        'use_states_info_gain': True,
        'inference_horizon': 1,
        'action_selection': 'deterministic'
    },
    'active_inference': {
        'track_beliefs': True,
        'track_free_energy': True,
        'track_policy_probs': True,
        'track_observations': True,
        'track_actions': True,
        'track_states': True,
        'track_messages': True,
        'track_prediction_errors': True,
        'track_entropy': True
    }
} 