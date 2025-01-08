"""
RGM Package
==========

Recursive Generative Models implementation.
"""

from pathlib import Path

# Package paths
RGM_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = RGM_ROOT / "models"
UTILS_DIR = RGM_ROOT / "utils"
CONFIGS_DIR = RGM_ROOT / "configs"

# Import core components
from .rgm_render import RGMRenderer
from .rgm_execute import RGMExecutor
from .rgm_analyze import RGMAnalyzer

# Import utilities
from .utils.rgm_experiment_utils import RGMExperimentUtils
from .utils.rgm_pipeline_manager import RGMPipelineManager
from .utils.rgm_config_loader import RGMConfigLoader
from .utils.rgm_validation_utils import RGMValidationUtils
from .utils.rgm_matrix_normalizer import RGMMatrixNormalizer
from .utils.rgm_message_passing import RGMMessagePassing
from .utils.rgm_model_state import RGMModelState
from .utils.rgm_data_manager import RGMDataManager
from .utils.rgm_data_utils import RGMDataUtils
from .utils.rgm_core_utils import RGMCoreUtils
from .utils.rgm_svd_utils import RGMSVDUtils
from .utils.rgm_metrics_utils import RGMMetricsUtils

# Version
__version__ = '0.1.0'

# Make paths available
__all__ = [
    'RGM_ROOT',
    'MODELS_DIR',
    'UTILS_DIR',
    'CONFIGS_DIR',
    'RGMRenderer',
    'RGMExecutor',
    'RGMAnalyzer',
    'RGMExperimentUtils',
    'RGMPipelineManager',
    'RGMConfigLoader',
    'RGMValidationUtils',
    'RGMMatrixNormalizer',
    'RGMMessagePassing',
    'RGMModelState',
    'RGMDataManager',
    'RGMDataUtils',
    'RGMCoreUtils',
    'RGMSVDUtils',
    'RGMMetricsUtils'
]