"""
RGM Experiment Utilities
=======================

Manages experiment directories and logging for RGM pipeline.
Provides centralized experiment path management.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

class RGMExperimentState:
    """Manages experiment state and configuration"""
    
    def __init__(self):
        """Initialize experiment state"""
        self._experiment = None
        self._root_logger = None
        
    def initialize_experiment(self, name: str, base_dir: Path) -> Dict:
        """
        Initialize new experiment.
        
        Args:
            name: Base name for experiment
            base_dir: Base directory for experiments
            
        Returns:
            Dictionary containing experiment state
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create experiment name with timestamp
            exp_name = f"{name}_{timestamp}"
            
            # Create experiment root directory
            root_dir = base_dir / exp_name
            root_dir.mkdir(parents=True, exist_ok=True)
            
            # Create experiment directory structure
            dirs = {
                'root': root_dir,
                'logs': root_dir / "logs",
                'render': root_dir / "render",
                'matrices': root_dir / "render" / "matrices",
                'config': root_dir / "render" / "config",
                'gnn_specs': root_dir / "render" / "config" / "gnn_specs",
                'visualizations': root_dir / "render" / "visualizations",
                'simulation': root_dir / "simulation",
                'results': root_dir / "simulation" / "results",
                'analysis': root_dir / "analysis",
                'validation': root_dir / "validation",
                'checkpoints': root_dir / "checkpoints",
                'errors': root_dir / "errors"
            }
            
            # Create all directories
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
                
            # Create experiment state
            self._experiment = {
                'name': exp_name,
                'timestamp': timestamp,
                'dirs': dirs,
                'status': 'initialized',
                'current_stage': None,
                'error_state': None
            }
            
            # Initialize logging
            self._setup_logging()
            
            # Log directory structure
            self._log_directory_structure()
            
            return self._experiment
            
        except Exception as e:
            if self._root_logger:
                self._root_logger.error(f"Failed to initialize experiment: {str(e)}")
            raise
            
    def get_experiment(self) -> Dict:
        """Get current experiment state"""
        if self._experiment is None:
            raise RuntimeError("No experiment initialized")
        return self._experiment
        
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component"""
        if self._experiment is None:
            raise RuntimeError("No experiment initialized")
            
        logger = logging.getLogger(f"rgm.{component}")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger
        
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create logs directory
            logs_dir = self._experiment['dirs']['logs']
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log file path
            log_file = logs_dir / f"{self._experiment['name']}.log"
            
            # Create formatters
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # Create handlers
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            
            # Configure root logger
            self._root_logger = logging.getLogger('rgm')
            self._root_logger.setLevel(logging.INFO)
            self._root_logger.addHandler(file_handler)
            self._root_logger.addHandler(console_handler)
            
        except Exception as e:
            if self._root_logger:
                self._root_logger.error(f"Failed to setup logging: {str(e)}")
            raise
            
    def _log_directory_structure(self):
        """Log experiment directory structure"""
        logger = logging.getLogger('rgm')
        
        logger.info(f"Initialized experiment: {self._experiment['name']}")
        for name, path in self._experiment['dirs'].items():
            logger.info(f"- {name}: {path}")
            
    def update_status(self, status: str, stage: Optional[str] = None):
        """Update experiment status"""
        if self._experiment is None:
            raise RuntimeError("No experiment initialized")
            
        self._experiment['status'] = status
        if stage:
            self._experiment['current_stage'] = stage
            
        # Log status change
        logger = logging.getLogger('rgm')
        logger.info(f"Experiment status: {status}" + (f" ({stage})" if stage else ""))
        
    def save_error_state(self, error: Exception):
        """Save error state information"""
        if self._experiment is None:
            raise RuntimeError("No experiment initialized")
            
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'status': self._experiment['status'],
                'stage': self._experiment['current_stage']
            }
            
            # Save to experiment state
            self._experiment['error_state'] = error_info
            
            # Save to file
            error_path = self._experiment['dirs']['errors'] / "error_state.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
                
            # Log error
            logger = logging.getLogger('rgm')
            logger.error(f"Error state saved to: {error_path}")
            
        except Exception as e:
            logger = logging.getLogger('rgm')
            logger.error(f"Failed to save error state: {str(e)}")

class RGMExperimentUtils:
    """Static interface to experiment state"""
    
    _state = RGMExperimentState()
    
    @classmethod
    def initialize_experiment(cls, name: str, base_dir: Path) -> Dict:
        """Initialize new experiment"""
        return cls._state.initialize_experiment(name, base_dir)
        
    @classmethod
    def get_experiment(cls) -> Dict:
        """Get current experiment state"""
        return cls._state.get_experiment()
        
    @classmethod
    def get_logger(cls, component: str) -> logging.Logger:
        """Get logger for specific component"""
        return cls._state.get_logger(component)
        
    @classmethod
    def update_status(cls, status: str, stage: Optional[str] = None):
        """Update experiment status"""
        cls._state.update_status(status, stage)
        
    @classmethod
    def save_error_state(cls, error: Exception):
        """Save error state information"""
        cls._state.save_error_state(error)