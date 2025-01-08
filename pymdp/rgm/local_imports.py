"""
Local Imports Handler
===================

Centralizes import management and path setup for RGM modules.
Ensures consistent module access across the pipeline.
"""

import os
import sys
from pathlib import Path

# Get key directories
CURRENT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = CURRENT_DIR.parent
UTILS_DIR = CURRENT_DIR / "utils"
MODELS_DIR = CURRENT_DIR / "models"

# Add directories to path
sys.path.extend([
    str(CURRENT_DIR),
    str(PARENT_DIR),
    str(UTILS_DIR),
    str(MODELS_DIR)
])

# Import all utils
from utils.rgm_experiment_utils import RGMExperimentUtils
from utils.rgm_core_utils import RGMCoreUtils
from utils.rgm_config_utils import RGMConfigUtils
from utils.rgm_validation_utils import RGMValidationUtils
from utils.rgm_svd_utils import RGMSVDUtils
from utils.rgm_gnn_matrix_factory import RGMGNNMatrixFactory
from utils.rgm_matrix_visualization_utils import RGMMatrixVisualizationUtils
from utils.rgm_learning_utils import RGMLearningUtils
from utils.rgm_message_utils import RGMMessageUtils
from utils.rgm_metrics_utils import RGMMetricsUtils

# Initialize experiment utils
def init_rgm_experiment(base_name: str = "rgm_mnist"):
    """Initialize RGM experiment with proper path setup"""
    # Ensure we're in the right directory
    os.chdir(CURRENT_DIR)
    
    # Initialize experiment
    experiment = RGMExperimentUtils.init_experiment(base_name)
    
    return experiment