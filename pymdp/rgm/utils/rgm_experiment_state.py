"""
RGM Experiment State
===================

Manages experiment state and directory structure.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

class RGMExperimentState:
    """Manages state and directory structure for RGM experiments"""
    
    def __init__(self, name: str, base_dir: Optional[Path] = None):
        """
        Initialize experiment state.
        
        Args:
            name: Name of the experiment
            base_dir: Base directory for experiments (default: rgm/experiments)
        """
        self.name = name
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up directory structure
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "experiments"
        self.base_dir = Path(base_dir)
        
        # Create experiment directory with timestamp
        self.exp_dir = self.base_dir / f"{name}_{self.timestamp}"
        self._create_directory_structure()
        
        # Initialize state
        self.state = {
            "name": name,
            "timestamp": self.timestamp,
            "start_time": self.start_time,
            "directories": self._get_directories(),
            "status": "initialized"
        }
        
        # Save initial state
        self._save_state()
        
    def _create_directory_structure(self):
        """Create experiment directory structure"""
        # Create main directories
        dirs = {
            "root": self.exp_dir,
            "logs": self.exp_dir / "logs",
            "data": self.exp_dir / "data",
            "models": self.exp_dir / "models",
            "matrices": self.exp_dir / "matrices",
            "results": self.exp_dir / "results",
            "analysis": self.exp_dir / "analysis",
            "render": self.exp_dir / "render",
            "config": self.exp_dir / "config",
            "gnn_specs": self.exp_dir / "gnn_specs",
            "checkpoints": self.exp_dir / "checkpoints"
        }
        
        # Create all directories
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.directories = dirs
        
    def _get_directories(self) -> Dict[str, str]:
        """Get dictionary of directory paths"""
        return {
            name: str(path) for name, path in self.directories.items()
        }
        
    def _save_state(self):
        """Save experiment state to file"""
        state_file = self.exp_dir / "experiment_state.json"
        with open(state_file, "w") as f:
            json.dump(self.state, f, indent=4)
            
    def get_dir(self, name: str) -> Path:
        """Get path to named directory"""
        if name not in self.directories:
            raise ValueError(f"Unknown directory: {name}")
        return self.directories[name]
        
    def get_log_dir(self) -> Path:
        """Get path to log directory"""
        return self.directories["logs"]
        
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
        
    def update_status(self, status: str):
        """Update experiment status"""
        self.state["status"] = status
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()
        
    def add_metadata(self, key: str, value: any):
        """Add metadata to experiment state"""
        if "metadata" not in self.state:
            self.state["metadata"] = {}
        self.state["metadata"][key] = value
        self._save_state()
        
    def get_metadata(self, key: str) -> any:
        """Get metadata value"""
        if "metadata" not in self.state or key not in self.state["metadata"]:
            raise KeyError(f"Metadata key not found: {key}")
        return self.state["metadata"][key] 