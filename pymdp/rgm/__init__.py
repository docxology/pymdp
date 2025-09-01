"""
Renormalization Generative Model (RGM) Package

This package implements the RGM architecture for generative modeling,
providing a complete pipeline for training and evaluating RGM models.

Key Components:
- RGMPipeline: Main pipeline for training and evaluation
- Models: Implementation of RGM model architectures
- Training: Training and optimization utilities
- Utils: Configuration, logging, and data handling
"""

from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = "RGM Team"

# Set up package-level paths
PACKAGE_ROOT = Path(__file__).parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
DATA_DIR = PACKAGE_ROOT / "data"
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"

# Ensure required directories exist
for directory in [CONFIG_DIR, DATA_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Import utilities first
try:
    from .utils.rgm_logging import RGMLogging
    from .utils.rgm_config_validator import RGMConfigValidator
    from .utils.rgm_data_loader import RGMDataLoader
    from .utils.rgm_gnn_loader import RGMGNNLoader
    from .utils.rgm_gnn_matrix_factory import RGMGNNMatrixFactory
    from .utils.rgm_experiment_utils import RGMExperimentUtils
    from .utils.rgm_matrix_visualization_utils import RGMMatrixVisualizationUtils
    _RGM_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"RGM utilities could not be imported (likely due to missing dependencies): {e}")
    _RGM_AVAILABLE = False

# Import data handling
if _RGM_AVAILABLE:
    try:
        from .mnist_download import MNISTPreprocessor
        
        # Import model components
        from .models.rgm_mnist import RGMMNISTModel
        from .training.rgm_trainer import RGMTrainer
        
        # Import visualization
        from .visualization import RGMRenderer
        
        # Import pipeline last to avoid circular imports
        from .run_rgm import RGMPipeline
        
    except ImportError as e:
        warnings.warn(f"Additional RGM components could not be imported: {e}")
        _RGM_AVAILABLE = False

# Define public API - only include what's available
__all__ = [
    # Package info (always available)
    '__version__',
    '__author__',
    'PACKAGE_ROOT',
    'CONFIG_DIR',
    'DATA_DIR',
    'EXPERIMENTS_DIR'
]

# Add RGM components if they were successfully imported
if _RGM_AVAILABLE:
    __all__.extend([
        # Core components
        'RGMPipeline',
        'RGMRenderer',
        
        # Utilities
        'RGMLogging',
        'RGMConfigValidator',
        'RGMDataLoader',
        'RGMGNNLoader',
        'RGMGNNMatrixFactory',
        'RGMExperimentUtils',
        'RGMMatrixVisualizationUtils',
        
        # Data handling
        'MNISTPreprocessor',
        
        # Model components
        'RGMMNISTModel',
        'RGMTrainer'
    ])