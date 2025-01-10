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

# Initialize experiment utils
def init_rgm_experiment(base_name: str = "rgm_mnist"):
    """Initialize RGM experiment with proper path setup"""
    # Ensure we're in the right directory
    os.chdir(CURRENT_DIR)
    
    # Initialize experiment
    experiment = RGMExperimentUtils.init_experiment(base_name)
    
    return experiment