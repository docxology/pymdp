"""
RGM Model Specifications
======================

GNN model specifications for RGM implementation.
"""

from pathlib import Path

# Get model directory
MODEL_DIR = Path(__file__).parent

# Model file paths
BASE_MODEL = MODEL_DIR / "rgm_base.gnn"
SVD_BLOCK_MODEL = MODEL_DIR / "rgm_svd_block.gnn"
HIERARCHICAL_MODEL = MODEL_DIR / "rgm_hierarchical_level.gnn"
MNIST_MODEL = MODEL_DIR / "rgm_mnist.gnn"
MESSAGE_PASSING_MODEL = MODEL_DIR / "rgm_message_passing.gnn"
ACTIVE_LEARNING_MODEL = MODEL_DIR / "rgm_active_learning.gnn"

__all__ = [
    'MODEL_DIR',
    'BASE_MODEL',
    'SVD_BLOCK_MODEL',
    'HIERARCHICAL_MODEL',
    'MNIST_MODEL',
    'MESSAGE_PASSING_MODEL',
    'ACTIVE_LEARNING_MODEL'
] 