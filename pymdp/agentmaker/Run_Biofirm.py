"""
Biofirm Experiment Runner
========================

Main script for running complete Biofirm active inference experiments.
"""

import os
import sys
import json
import time
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pymdp modules using absolute imports
from pymdp.gnn.gnn_matrix_factory import GNNMatrixFactory
from pymdp.gnn.biofirm_utils import BiofirmUtils
from pymdp.agentmaker.biofirm_execution_utils import BiofirmExecutionUtils
from pymdp.agentmaker.config import (
    DEFAULT_PATHS,
    DEFAULT_ENV_GNN,
    DEFAULT_AGENT_GNN,
    EXPERIMENT_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class BiofirmExperiment:
    """Execute and analyze complete Biofirm experiments"""
    
    def __init__(self, name: str = "biofirm_exp", config: Optional[Dict] = None):
        """Initialize experiment with configuration"""
        try:
            # Ensure name is a string
            if not isinstance(name, str):
                logger.warning(f"Invalid name type {type(name)}, using default")
                name = "biofirm_exp"
            self.name = name
            
            # Use default config if none provided
            if config is None:
                config = EXPERIMENT_CONFIG.copy()
            self.config = config
            
            # Initialize tracking
            self.stage_times = {}
            self.stage_results = {}
            
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = DEFAULT_PATHS['output_base'] / f"experiment_{timestamp}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create stage directories
            self.render_dir = self.output_dir / "1_render"
            self.simulation_dir = self.output_dir / "2_simulation"
            self.analysis_dir = self.output_dir / "3_analysis"
            
            # Create logs directory
            self.logs_dir = self.output_dir / "logs"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup logging
            self._setup_logging()
            
            # Print experiment header
            logger.info("\n" + "=" * 80)
            logger.info(f"Biofirm Experiment: {self.name}")
            logger.info("=" * 80 + "\n")
            logger.info(f"Output Directory: {self.output_dir}")
            
            # Get GNN model paths
            self.env_gnn = DEFAULT_PATHS['env_gnn']
            self.agent_gnn = DEFAULT_PATHS['agent_gnn']
            logger.info("\nUsing GNN models:")
            logger.info(f"- Environment: {self.env_gnn}")
            logger.info(f"- Agent: {self.agent_gnn}")
            
        except Exception as e:
            logger.error(f"Error initializing experiment: {str(e)}")
            raise

    def _setup_logging(self):
        """Configure logging for experiment"""
        try:
            # Create log file with timestamp
            log_file = self.logs_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)7s | %(message)s',
                datefmt='%H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            logger.info(f"Experiment log file: {log_file}")
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")  # Use print since logging might not work
            raise

    def run_experiment(self):
        """Run complete experiment pipeline"""
        try:
            # Initialize timing
            experiment_start = time.time()
            
            # Stage 1: Matrix Rendering
            logger.info("\n------------------------------------------------------------")
            logger.info("Stage 1: Matrix Rendering")
            logger.info("------------------------------------------------------------\n")
            
            render_start = time.time()
            matrices_dir = BiofirmExecutionUtils.run_rendering_stage(
                self.env_gnn,
                self.agent_gnn,
                self.render_dir
            )
            render_time = time.time() - render_start
            self.stage_times['render'] = render_time
            
            # Stage 2: Simulation
            logger.info("\n------------------------------------------------------------")
            logger.info("Stage 2: Simulation Execution")
            logger.info("------------------------------------------------------------\n")
            
            sim_start = time.time()
            simulation_dir = BiofirmExecutionUtils.run_simulation_stage(
                matrices_dir,
                self.simulation_dir
            )
            sim_time = time.time() - sim_start
            self.stage_times['simulation'] = sim_time
            
            # Stage 3: Analysis
            logger.info("\n------------------------------------------------------------")
            logger.info("Stage 3: Analysis")
            logger.info("------------------------------------------------------------\n")
            
            analysis_start = time.time()
            # Validate simulation data before analysis
            if BiofirmExecutionUtils.validate_simulation_transition(simulation_dir):
                analysis_dir = BiofirmExecutionUtils.run_analysis_stage(
                    simulation_dir,
                    self.analysis_dir
                )
                analysis_time = time.time() - analysis_start
                self.stage_times['analysis'] = analysis_time
                self.stage_results['analysis'] = {
                    'completed': True,
                    'output': str(analysis_dir),
                    'time': analysis_time
                }
            else:
                logger.error("Cannot proceed to analysis: simulation validation failed")
                self.stage_results['analysis'] = {
                    'completed': False,
                    'output': None,
                    'time': 0.0
                }
                
            # Calculate total experiment time
            total_time = time.time() - experiment_start
            
            # Save experiment summary with accurate timings
            self._save_experiment_summary(
                start_time=experiment_start,
                total_time=total_time
            )
            
        except Exception as e:
            logger.error("\n------------------------------------------------------------")
            logger.error("Error during experiment")
            logger.error("------------------------------------------------------------\n")
            logger.error(str(e))
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Experiment failed: {str(e)}")

    def _save_experiment_summary(self, start_time: float, total_time: float):
        """Save experiment summary with results from all stages"""
        try:
            # Create summary with accurate timestamps
            summary = {
                'experiment': {
                    'name': self.name,
                    'started': datetime.fromtimestamp(start_time).isoformat(),
                    'completed': datetime.fromtimestamp(start_time + total_time).isoformat(),
                    'total_duration': f"{total_time:.2f}s",
                    'config': self.config
                },
                'stage_results': {
                    'render': {
                        'completed': True,
                        'output': str(self.render_dir / "matrices"),
                        'time': f"{self.stage_times.get('render', 0.0):.2f}s"
                    },
                    'simulation': {
                        'completed': True,
                        'output': str(self.simulation_dir),
                        'time': f"{self.stage_times.get('simulation', 0.0):.2f}s"
                    },
                    'analysis': {
                        'completed': self.stage_results.get('analysis', {}).get('completed', False),
                        'output': self.stage_results.get('analysis', {}).get('output', None),
                        'time': f"{self.stage_times.get('analysis', 0.0):.2f}s"
                    }
                },
                'output_structure': {
                    'root': str(self.output_dir),
                    'render': str(self.render_dir),
                    'simulation': str(self.simulation_dir),
                    'analysis': str(self.analysis_dir),
                    'logs': str(self.logs_dir)
                }
            }
            
            # Load and include simulation metrics if available
            metrics_file = self.simulation_dir / "metrics" / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    summary['metrics'] = json.load(f)
                    
            # Load and include analysis summary if available
            analysis_summary = self.analysis_dir / "analysis_summary.json"
            if analysis_summary.exists():
                with open(analysis_summary) as f:
                    summary['analysis'] = json.load(f)
                    
            # Save summary
            summary_file = self.output_dir / "experiment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Log summary info
            logger.info("\n" + "=" * 80)
            logger.info("Experiment Summary")
            logger.info("=" * 80 + "\n")
            logger.info(f"\nExperiment: {self.name}")
            logger.info(f"Started: {summary['experiment']['started']}")
            logger.info(f"Completed: {summary['experiment']['completed']}")
            logger.info(f"Total Duration: {summary['experiment']['total_duration']}")
            logger.info("\nStage Results:")
            
            for stage, results in summary['stage_results'].items():
                logger.info(f"\n{stage.title()} Stage:")
                logger.info(f"- Completed: {results['completed']}")
                logger.info(f"- Output: {results['output']}")
                logger.info(f"- Time: {results['time']}")
                
            logger.info("\n" + "=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"Error saving experiment summary: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Create experiment with default name and config
        experiment = BiofirmExperiment(
            name=EXPERIMENT_CONFIG['name'],
            config=EXPERIMENT_CONFIG
        )
        
        # Run experiment
        experiment.run_experiment()
        
    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise RuntimeError(f"Experiment failed: {str(e)}")

if __name__ == "__main__":
    main()
