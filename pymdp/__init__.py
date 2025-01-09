"""
PyMDP Package
============

A Python implementation of Active Inference and Message Passing.
"""

from pathlib import Path

# Package metadata
__version__ = '0.1.0'
__author__ = 'PyMDP Contributors'

# Package paths
PACKAGE_ROOT = Path(__file__).parent.absolute()

# Import core components
from .inference import infer_states
from .control import sample_action, infer_policies
from .utils import obj_array_zeros, sample
from .maths import softmax

__all__ = [
    'infer_states',
    'sample_action',
    'infer_policies',
    'obj_array_zeros',
    'sample',
    'softmax',
    'PACKAGE_ROOT'
]
