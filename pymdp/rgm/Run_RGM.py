"""
RGM Pipeline Runner
==================

Main entry point for RGM MNIST pipeline execution.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from pymdp package
from pymdp.rgm.utils.rgm_experiment_utils import RGMExperimentUtils
from pymdp.rgm.utils.rgm_pipeline_manager import RGMPipelineManager
from pymdp.rgm.utils.rgm_config_loader import RGMConfigLoader
from pymdp.rgm.utils.rgm_validation_utils import RGMValidationUtils

def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging configuration"""
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
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
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def initialize_experiment(base_dir: Path) -> Dict:
    """
    Initialize experiment directory structure.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Dictionary containing experiment configuration
    """
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"rgm_mnist_pipeline_{timestamp}"
        
        # Create base directory if it doesn't exist
        base_dir = base_dir.absolute()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment
        experiment = RGMExperimentUtils.initialize_experiment(
            name=exp_name,
            base_dir=base_dir
        )
        
        # Set up logging
        setup_logging(experiment['dirs']['logs'])
        
        # Load and validate configuration
        config_loader = RGMConfigLoader()
        config = config_loader.load_config()
        
        # Validate experiment structure
        validator = RGMValidationUtils()
        if not validator.validate_experiment_structure(experiment):
            raise ValueError("Invalid experiment structure")
            
        return experiment
        
    except Exception as e:
        logging.error(f"Failed to initialize experiment: {str(e)}")
        raise

def run_pipeline(checkpoint_path: Optional[Path] = None) -> Path:
    """
    Run complete RGM pipeline.
    
    Args:
        checkpoint_path: Optional path to resume from checkpoint
        
    Returns:
        Path to analysis directory
    """
    try:
        # Initialize experiment
        experiment_dir = project_root / "experiments"
        experiment = initialize_experiment(experiment_dir)
        
        # Initialize pipeline manager
        pipeline = RGMPipelineManager()
        
        # Execute pipeline
        if checkpoint_path:
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            analysis_dir = pipeline.resume_from_checkpoint(checkpoint_path)
        else:
            analysis_dir = pipeline.execute_pipeline()
            
        return analysis_dir
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        
        # Save error state
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': logging.traceback.format_exc()
            }
            
            error_dir = project_root / "experiments" / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            error_path = error_dir / f"error_{datetime.now():%Y%m%d_%H%M%S}.json"
            
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
                
            logging.info(f"Error details saved to: {error_path}")
            
        except Exception as save_error:
            logging.error(f"Failed to save error state: {str(save_error)}")
            
        raise

def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        checkpoint_path = None
        if len(sys.argv) > 1:
            checkpoint_path = Path(sys.argv[1])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
        # Run pipeline
        analysis_dir = run_pipeline(checkpoint_path)
        
        # Log completion
        logging.info(f"Pipeline completed successfully. Results in: {analysis_dir}")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()