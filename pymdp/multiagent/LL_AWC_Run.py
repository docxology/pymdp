#!/usr/bin/env python3
"""
Active Web Campaign (AWC) Runner
==============================

This script runs the Active Web Campaign simulation using the pymdp framework.
It coordinates the interaction between Website and Administrator agents.

The AWC model consists of three agents:
1. Administrator - manages ad campaigns and settings
2. Website - manages website content and user experience
3. Consumer - interacts with the website and ads

This script orchestrates the interaction between these agents and saves results.
"""

import os
import sys
from pathlib import Path
import logging
import traceback

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    try:
        logger.info("=" * 80)
        logger.info("Starting Active Web Campaign simulation")
        logger.info("=" * 80)
        
        # Log current working directory and Python version
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        
        # Setup paths first
        logger.info("-" * 40)
        logger.info("Setting up Python paths...")
        
        # Add project root to Python path
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # Adjusted to reach project root correctly
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            logger.info(f"Added project root to Python path: {project_root}")
        
        # Import LL_Methods after setting up paths
        from pymdp.multiagent.LL_Methods import (
            check_and_install_dependencies,
            setup_paths,
            check_pymdp_installation,
            import_dependencies,
            setup_output_directories,
            run_simulation,
            save_results,
            save_visualizations,
            get_config_summary
        )
        
        # Check and install required packages
        logger.info("-" * 40)
        logger.info("Checking dependencies...")
        if not check_and_install_dependencies():
            logger.error("Failed to install required packages")
            return 1
            
        # Check pymdp installation
        logger.info("-" * 40)
        logger.info("Verifying pymdp installation...")
        if not check_pymdp_installation():
            logger.error("Failed to verify pymdp installation")
            return 1
            
        # Import dependencies
        logger.info("-" * 40)
        logger.info("Importing dependencies...")
        if not import_dependencies():
            logger.error("Failed to import dependencies")
            return 1
            
        try:
            # Create output directories
            logger.info("-" * 40)
            logger.info("Setting up output directories...")
            timestamp, run_dir, analysis_dir, figures_dir = setup_output_directories()
            logger.info(f"Created output directories in {run_dir}")
            
            # Run simulation
            logger.info("-" * 40)
            logger.info("Running simulation...")
            results = run_simulation()
            
            # Save results
            logger.info("-" * 40)
            logger.info("Saving results...")
            save_results(results, get_config_summary(), analysis_dir)
            
            # Generate visualizations
            logger.info("-" * 40)
            logger.info("Generating visualizations...")
            save_visualizations(T=25, 
                              aAdm_facs=results['Administrator']['actions'],
                              sAdm_facs=results['Administrator']['states'],
                              s̆Web_facs=results['Administrator']['true_states'],
                              yWeb_mods=results['Administrator']['observations'],
                              aWeb_facs=results['Website']['actions'],
                              sWeb_facs=results['Website']['states'],
                              s̆Csr_facs=results['Website']['true_states'],
                              yCsr_mods=results['Website']['observations'],
                              GAdmNegs=results['Administrator']['free_energy'],
                              GWebNegs=results['Website']['free_energy'],
                              labAdm=_labAdm,
                              labWeb=_labWeb,
                              figures_dir=figures_dir)
            
            logger.info("=" * 80)
            logger.info("Simulation completed successfully")
            logger.info(f"Results saved in: {run_dir}")
            logger.info(f"Analysis files: {analysis_dir}")
            logger.info(f"Figures: {figures_dir}")
            logger.info("=" * 80)
            return 0
            
        except Exception as e:
            logger.error("Error during simulation execution:")
            logger.error(traceback.format_exc())
            return 1
            
    except Exception as e:
        logger.error("Error during setup:")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())