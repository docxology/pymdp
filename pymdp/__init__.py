"""
PyMDP: Active Inference Framework
===============================

A Python implementation of Active Inference for Markov Decision Processes.

Core Modules
-----------
gnn : Generalized Neural Notation
    Model definition and matrix generation
    
agentmaker : Experiment Framework
    Experiment orchestration and execution
    
visualization : Visualization Tools
    Matrix and results visualization
"""

from pathlib import Path
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Define package root for resource loading
PACKAGE_ROOT = Path(__file__).parent

# Import key modules
from . import gnn
from . import agentmaker
from . import visualization

__version__ = "0.1.0"
__all__ = ['gnn', 'agentmaker', 'visualization']
