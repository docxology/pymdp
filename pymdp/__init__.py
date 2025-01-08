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
from .rgm import (
    RGMRenderer,
    RGMExecutor,
    RGMAnalyzer,
    RGMExperimentUtils,
    RGMPipelineManager
)

__all__ = [
    'RGMRenderer',
    'RGMExecutor', 
    'RGMAnalyzer',
    'RGMExperimentUtils',
    'RGMPipelineManager',
    'PACKAGE_ROOT'
]
