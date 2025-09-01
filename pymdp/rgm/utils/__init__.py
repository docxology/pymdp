"""
RGM Utilities Module

This module provides utility functions and classes for the RGM package.
"""

# Import basic utilities first
try:
    from .rgm_logging import RGMLogging
    from .rgm_config_validator import RGMConfigValidator
    _BASIC_UTILS_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Basic RGM utilities could not be imported: {e}")
    _BASIC_UTILS_AVAILABLE = False

# Import data loader separately (may have additional dependencies)
try:
    from .rgm_data_loader import RGMDataLoader
    _DATA_LOADER_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"RGM data loader could not be imported (likely missing torch/torchvision): {e}")
    _DATA_LOADER_AVAILABLE = False

# Define exports based on what's available
__all__ = []

if _BASIC_UTILS_AVAILABLE:
    __all__.extend(['RGMLogging', 'RGMConfigValidator'])

if _DATA_LOADER_AVAILABLE:
    __all__.append('RGMDataLoader') 