"""
PyMDP Textbook Source Extensions
===============================

Additional methods and utilities for PyMDP textbook examples and exercises.

This package provides:
- Enhanced utility functions for model building
- Visualization tools for active inference
- Analysis methods for understanding agent behavior  
- Educational examples and demonstrations

Modules:
- model_utils: Enhanced model construction utilities
- visualization: Plotting and visualization functions
- analysis: Analysis and diagnostic tools
- educational: Educational examples and demonstrations
- experiments: Experimental frameworks and utilities
"""

from .model_utils import *
from .visualization import *
from .analysis import *
from .educational import *

__version__ = "1.0.0"
__author__ = "PyMDP Textbook Contributors"

__all__ = [
    # Model utilities
    'create_gridworld_model',
    'create_tmaze_model',
    'create_random_model',
    'validate_model',
    
    # Visualization
    'plot_beliefs',
    'plot_policy_tree',
    'plot_free_energy',
    'animate_agent',
    
    # Analysis
    'compute_model_entropy',
    'analyze_policy_complexity',
    'measure_exploration',
    'evaluate_performance',
    
    # Educational
    'step_by_step_inference',
    'interactive_demo',
    'concept_illustration'
]
