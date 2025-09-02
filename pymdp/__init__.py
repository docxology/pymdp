"""
PyMDP Package
============

A Python implementation of Active Inference and Message Passing.

This package includes:
- Core Active Inference algorithms
- Relational Generative Models (RGM)
- Multi-agent systems
- Visualization tools
"""

from pathlib import Path

# Package metadata
__version__ = '0.1.0'
__author__ = 'PyMDP Contributors'

# Package paths
PACKAGE_ROOT = Path(__file__).parent.absolute()

# Import modules
from . import agent
from . import envs
from . import utils
from . import maths
from . import control
from . import inference
from . import learning
from . import algos
from . import default_models
from . import jax

# Import core components  
from .inference import update_posterior_states
from .control import sample_action
from .utils import obj_array_zeros, sample
from .maths import softmax

# Import RGM components (optional)
# Temporarily disabled due to torch/torchvision compatibility issues
# try:
#     from . import rgm
# except ImportError as e:
#     import warnings
#     warnings.warn(f"RGM module could not be imported: {e}. RGM functionality will not be available.")
#     rgm = None
rgm = None

__all__ = [
    # Core components
    'update_posterior_states',
    'sample_action',
    'obj_array_zeros',
    'sample',
    'softmax',
    'PACKAGE_ROOT'
]

# Add rgm to exports if it was successfully imported
if rgm is not None:
    __all__.append('rgm')
