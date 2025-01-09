"""
RGM Experiment Utilities
======================

Utility functions for managing Renormalization Generative Model experiments.
"""

from pathlib import Path
from typing import Dict, Optional, List
import shutil

from .rgm_logging import RGMLogging

class RGMExperimentUtils:
    """Utility functions for RGM experiments."""
    
    logger = RGMLogging.get_logger("rgm.experiment_utils")
    
    @staticmethod
    def get_gnn_dir() -> Path:
        """Get the directory containing GNN specification files."""
        return Path(__file__).parent.parent / "models"
    
    @staticmethod
    def copy_gnn_files(source_dir: Path, target_dir: Path) -> List[str]:
        """
        Copy GNN specification files to experiment directory.
        
        Args:
            source_dir: Source directory containing GNN files
            target_dir: Target directory to copy files to
            
        Returns:
            List of copied file names
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        copied_files = []
        
        RGMExperimentUtils.logger.info(f"📁 Copying GNN files from: {source_dir}")
        RGMExperimentUtils.logger.info(f"📁 Target directory: {target_dir}")
        
        for gnn_file in source_dir.glob("*.gnn"):
            target_path = target_dir / gnn_file.name
            shutil.copy2(gnn_file, target_path)
            copied_files.append(gnn_file.name)
            RGMExperimentUtils.logger.info(f"   ↳ Copied: {gnn_file.name}")
            
        return copied_files
    
    def validate_gnn_directory(self, gnn_dir: Path) -> bool:
        """
        Validate GNN specification directory.
        
        Args:
            gnn_dir: Path to GNN specification directory
            
        Returns:
            True if valid, raises error otherwise
        """
        try:
            # Check directory exists
            if not gnn_dir.exists():
                raise FileNotFoundError(f"GNN directory not found: {gnn_dir}")
                
            # Check required files
            required_files = [
                'rgm_base.gnn',
                'rgm_mnist.gnn',
                'rgm_message_passing.gnn',
                'rgm_hierarchical_level.gnn'
            ]
            
            for file in required_files:
                file_path = gnn_dir / file
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing required GNN file: {file}")
                    
                # Validate file content
                with open(file_path) as f:
                    content = f.read().strip()
                    if not content:
                        raise ValueError(f"Empty GNN file: {file}")
                        
            self.logger.info(f"✅ GNN directory validated: {gnn_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"GNN directory validation failed: {str(e)}")
            raise