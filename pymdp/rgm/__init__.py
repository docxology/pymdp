"""
Renormalization Generative Model (RGM)
====================================

Implementation of the Renormalization Generative Model for hierarchical pattern
recognition and generation. This package provides the core functionality for
building and training models using renormalization group principles.
"""

from .models import RGMMNISTModel
from .utils import RGMLogging, RGMConfigLoader

__all__ = [
    'RGMMNISTModel',
    'RGMLogging',
    'RGMConfigLoader'
]