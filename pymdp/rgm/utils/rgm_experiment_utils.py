"""
RGM Experiment Utilities
=======================

Utilities for managing RGM experiments.
"""

import os
from pathlib import Path
from typing import Optional

from .rgm_logging import RGMLogging
from .rgm_experiment_state import RGMExperimentState

class RGMExperimentUtils:
    """Utilities for managing RGM experiments"""
    
    def __init__(self):
        """Initialize experiment utilities"""
        self.logger = RGMLogging.get_logger("rgm.experiment")
        
    def create_experiment_state(self, name: str, base_dir: Optional[Path] = None) -> RGMExperimentState:
        """
        Create new experiment state.
        
        Args:
            name: Name of the experiment
            base_dir: Optional base directory for experiments
            
        Returns:
            Initialized experiment state
        """
        try:
            self.logger.info(f"Creating experiment state: {name}")
            exp_state = RGMExperimentState(name, base_dir)
            self.logger.info(f"Created experiment in: {exp_state.exp_dir}")
            return exp_state
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment state: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
            
    def load_experiment_state(self, exp_dir: Path) -> RGMExperimentState:
        """
        Load existing experiment state.
        
        Args:
            exp_dir: Path to experiment directory
            
        Returns:
            Loaded experiment state
        """
        try:
            self.logger.info(f"Loading experiment state from: {exp_dir}")
            exp_state = RGMExperimentState.load(exp_dir)
            self.logger.info(f"Loaded experiment: {exp_state.name}")
            return exp_state
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment state: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            raise
            
    @staticmethod
    def get_logger(name: str) -> RGMLogging:
        """Get logger for component"""
        return RGMLogging.get_logger(f"rgm.{name}")

    @staticmethod
    def get_dir(name: str) -> Path:
        """Get path to named directory"""
        if name not in self.directories:
            raise ValueError(f"Unknown directory: {name}")
        return self.directories[name]