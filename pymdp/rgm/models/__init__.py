"""
Renormalization Generative Model Components
========================================

Model implementations for the Renormalization Generative Model (RGM).
"""

from pathlib import Path
from .rgm_mnist import RGMMNISTModel

# Define paths to GNN specification files
MODEL_DIR = Path(__file__).parent
GNN_FILES = {
    'base': MODEL_DIR / 'rgm_base.gnn',
    'mnist': MODEL_DIR / 'rgm_mnist.gnn',
    'hierarchical': MODEL_DIR / 'rgm_hierarchical_level.gnn',
    'message_passing': MODEL_DIR / 'rgm_message_passing.gnn'
}

__all__ = ['RGMMNISTModel', 'GNN_FILES'] 