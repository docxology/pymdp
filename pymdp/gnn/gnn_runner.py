import os
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import shutil

from pymdp.gnn.gnn_matrix_factory import GNNMatrixFactory
from pymdp.visualization.matrix_visualization import (plot_A_matrices, plot_B_matrices, 
                                                    plot_model_graph, plot_belief_history)
from pymdp.agentmaker.Run_Biofirm import ActiveInferenceExperiment

logger = logging.getLogger(__name__)

class GNNRunner:
    """
    Runner class for GNN model execution and visualization
    """
    
    def __init__(self, gnn_file, sandbox_root="sandbox"):
        """
        Initialize GNN runner
        
        Parameters
        ----------
        gnn_file : str
            Path to .gnn model file
        sandbox_root : str
            Root directory for sandbox outputs
        """
        self.gnn_file = Path(gnn_file)
        self.model_name = self.gnn_file.stem
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup sandbox structure
        self.sandbox_dir = self._setup_sandbox(sandbox_root)
        
        # Initialize factory
        self.factory = GNNMatrixFactory(gnn_file)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_sandbox(self, sandbox_root):
        """Create sandbox directory structure"""
        sandbox_dir = Path(sandbox_root) / f"{self.model_name}_{self.timestamp}"
        
        # Create subdirectories
        self.dirs = {
            'matrices': sandbox_dir / 'matrices',
            'visualizations': sandbox_dir / 'visualizations',
            'logs': sandbox_dir / 'logs',
            'exports': sandbox_dir / 'exports',
            'traces': sandbox_dir / 'traces'
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        return sandbox_dir
        
    def _setup_logging(self):
        """Configure logging"""
        log_file = self.dirs['logs'] / f"{self.model_name}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def generate_matrices(self):
        """Generate and save matrices from GNN definition"""
        logger.info("Generating matrices from GNN model")
        self.matrices = self.factory.create_matrices(self.dirs['matrices'])
        return self.matrices
        
    def generate_visualizations(self):
        """Generate all model visualizations"""
        logger.info("Generating model visualizations")
        
        viz_dir = self.dirs['visualizations']
        
        # Plot model graph
        plot_model_graph(self.factory.gnn, 
                        save_to=viz_dir / "model_graph.png")
        
        # Plot A matrices
        plot_A_matrices(self.matrices['A'],
                       modality_names=[m['name'] for m in self.factory.gnn['observations']['modalities']],
                       state_names=[f['name'] for f in self.factory.gnn['stateSpace']['factors']],
                       save_dir=viz_dir / "A_matrices")
        
        # Plot B matrices
        plot_B_matrices(self.matrices['B'],
                       factor_names=[f['name'] for f in self.factory.gnn['stateSpace']['factors']],
                       save_dir=viz_dir / "B_matrices")
                       
    def run_experiment(self, env_class, T=10, plot_interval=1):
        """
        Run active inference experiment with generated model
        
        Parameters
        ----------
        env_class : class
            Environment class to instantiate
        T : int
            Number of timesteps
        plot_interval : int
            Interval for saving visualizations
        """
        logger.info(f"Running experiment for {T} timesteps")
        
        # Create experiment instance
        experiment = ActiveInferenceExperiment(self.model_name)
        
        # Initialize environment
        env = env_class()
        
        # Create agent using generated matrices
        from pymdp.agent import Agent
        agent = Agent(**self.matrices)
        
        # Run experiment
        experiment.run_experiment(agent, env, T=T, plot_interval=plot_interval)
        
        # Copy experiment outputs to sandbox
        self._collect_experiment_outputs(experiment)
        
    def _collect_experiment_outputs(self, experiment):
        """Collect and organize experiment outputs in sandbox"""
        logger.info("Collecting experiment outputs")
        
        # Copy traces
        for trace_file in experiment.dirs['traces'].glob("*.npz"):
            shutil.copy2(trace_file, self.dirs['traces'])
            
        # Copy visualizations
        for viz_file in experiment.dirs['viz'].glob("*.png"):
            shutil.copy2(viz_file, self.dirs['visualizations'] / 'experiment')
            
        # Copy results
        for result_file in experiment.dirs['results'].glob("*.json"):
            shutil.copy2(result_file, self.dirs['exports'])
            
    def export_model(self):
        """Export final model documentation"""
        logger.info("Exporting model documentation")
        
        export_dir = self.dirs['exports']
        
        # Generate markdown documentation
        md_content = self.factory.to_markdown()
        with open(export_dir / "model.md", 'w') as f:
            f.write(md_content)
            
        # Copy key visualizations
        shutil.copy2(self.dirs['visualizations'] / "model_graph.png",
                    export_dir / "model_graph.png")
                    
        # Create summary JSON
        summary = {
            'modelName': self.model_name,
            'timestamp': self.timestamp,
            'matrices': {
                'A': [a.shape for a in self.matrices['A']],
                'B': [b.shape for b in self.matrices['B']],
                'C': [c.shape for c in self.matrices['C']]
            }
        }
        
        with open(export_dir / "model_summary.json", 'w') as f:
            json.dump(summary, f, indent=2) 