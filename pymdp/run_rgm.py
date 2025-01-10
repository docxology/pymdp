#!/usr/bin/env python3
"""RGM Pipeline Runner

Main entry point for training and evaluating RGM models.
Handles configuration, data loading, training, and visualization.

The pipeline consists of 5 stages:
1. MNIST Dataset Preparation
   - Download and validate MNIST dataset
   - Generate sample visualizations
   - Confirm data integrity

2. GNN Matrix Construction
   - Initialize matrices based on configuration
   - Analyze prior states and matrix properties
   - Generate matrix visualizations

3. Model Training
   - Initialize model with matrices
   - Train on MNIST dataset
   - Track and visualize progress

4. Model Evaluation
   - Evaluate on test set
   - Generate performance metrics
   - Create reconstruction samples

5. Post-Test Analysis
   - Analyze matrix evolution
   - Generate final visualizations
   - Compile comprehensive report
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import logging

# Add parent directory to Python path for development mode
PACKAGE_ROOT = Path(__file__).parent
if os.path.exists(PACKAGE_ROOT / ".env"):
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from rgm.pipeline import RGMPipeline
from rgm.utils.rgm_logging import RGMLogging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate RGM model on MNIST dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional, will use default if not provided)"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments/mnist",
        help="Directory for experiment outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def main():
    """Main entry point for RGM pipeline."""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = RGMLogging.setup_logging(
        log_dir=Path(args.exp_dir) / "logs",
        level=log_level,
        name="rgm.runner"
    )
    logger.info("Starting RGM pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = RGMPipeline(
            config_path=args.config,
            exp_dir=args.exp_dir,
            device=args.device
        )
        logger.info("✓ Pipeline initialized")
        
        # Run training
        logger.info("Starting training phase...")
        pipeline.train()
        logger.info("✓ Training complete")
        
        # Run evaluation
        logger.info("Starting evaluation phase...")
        metrics = pipeline.evaluate()
        logger.info("✓ Evaluation complete")
        
        # Run analysis
        logger.info("Starting analysis phase...")
        analysis = pipeline.analyze()
        logger.info("✓ Analysis complete")
        
        logger.info("🎉 RGM pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 