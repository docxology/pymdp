import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pymdp.visualization import matrix_visualization as mvis
import os

logger = logging.getLogger(__name__)

class GNNVisualizer:
    """Visualization tools for GNN models and their PyMDP matrix representations"""
    
    def __init__(self, model, matrices, output_dir=None):
        """
        Initialize visualizer with GNN model and matrices
        
        Parameters
        ----------
        model : dict
            GNN model definition
        matrices : dict
            PyMDP-compatible matrices
        output_dir : str, optional
            Directory for saving plots
        """
        self.model = model
        self.matrices = matrices
        self.output_dir = output_dir or "pymdp/sandbox/gnn_outputs/plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_visualization_dirs(self):
        """Create organized visualization subdirectories"""
        subdirs = [
            'A_matrices',  # Observation model
            'B_matrices',  # Transition model
            'C_matrices',  # Preferences
            'D_matrices',  # Initial state priors
            'observations',  # Observation sequences
            'states',  # Hidden state beliefs
            'policies',  # Policy evaluations
            'structure'  # Model structure graphs
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
            logger.debug(f"Created visualization subdirectory: {subdir}")

    def visualize_model(self, model_type="agent"):
        """
        Generate all model visualizations
        
        Parameters
        ----------
        model_type : str
            Type of model ("agent" or "environment") to adjust visualization
        """
        try:
            # Create visualization subdirectories
            self.create_visualization_dirs()
            
            logger.info(f"Generating {model_type} model visualizations")
            
            # Plot A matrices
            self._plot_A_matrices()
            
            # Plot B matrices
            self._plot_B_matrices()
            
            # Plot C matrices (agent only)
            if model_type == "agent" and 'C' in self.matrices:
                self._plot_C_matrices()
            
            # Plot D matrices
            self._plot_D_matrices()
            
            # Plot model structure
            self._plot_model_structure()
                    
            logger.info(f"Generated {model_type} model visualizations")
            
        except Exception as e:
            logger.error(f"Error generating {model_type} visualizations: {str(e)}")
            raise
            
    def _plot_A_matrices(self):
        """Plot observation model matrices"""
        for i, A in enumerate(self.matrices['A']):
            modality = self.model['observations']['modalities'][i]['name']
            mvis.plot_matrix_heatmap(
                A,
                title=f"Observation Model - {modality}",
                xlabel="Hidden States",
                ylabel="Observations",
                save_to=os.path.join(self.output_dir, 'A_matrices', f'{modality}.png')
            )
            
    def _plot_B_matrices(self):
        """Plot transition model matrices"""
        for i, B in enumerate(self.matrices['B']):
            factor = self.model['stateSpace']['factors'][i]['name']
            if len(B.shape) == 3:  # Controllable factor
                for a in range(B.shape[2]):
                    mvis.plot_matrix_heatmap(
                        B[:,:,a],
                        title=f"Transition Model - {factor} (Action {a})",
                        xlabel="Current State",
                        ylabel="Next State",
                        save_to=os.path.join(self.output_dir, 'B_matrices', f'{factor}_action_{a}.png')
                    )
            else:
                mvis.plot_matrix_heatmap(
                    B,
                    title=f"Transition Model - {factor}",
                    xlabel="Current State",
                    ylabel="Next State",
                    save_to=os.path.join(self.output_dir, 'B_matrices', f'{factor}.png')
                )
                
    def _plot_C_matrices(self):
        """Plot preference matrices if present"""
        if 'C' in self.matrices:
            for i, C in enumerate(self.matrices['C']):
                modality = self.model['observations']['modalities'][i]['name']
                mvis.plot_matrix_heatmap(
                    C.reshape(-1, 1),
                    title=f"Preferences - {modality}",
                    xlabel="Observation",
                    ylabel="Preference Value",
                    save_to=os.path.join(self.output_dir, 'C_matrices', f'{modality}.png')
                )
                
    def _plot_D_matrices(self):
        """Plot initial state prior matrices"""
        for i, D in enumerate(self.matrices['D']):
            factor = self.model['stateSpace']['factors'][i]['name']
            mvis.plot_matrix_heatmap(
                D.reshape(-1, 1),
                title=f"Initial State Distribution - {factor}",
                xlabel="State",
                ylabel="Probability",
                save_to=os.path.join(self.output_dir, 'D_matrices', f'{factor}.png')
            )
            
    def _plot_model_structure(self):
        """Plot model structure graph"""
        mvis.plot_model_graph(
            self.matrices['A'],
            factor_names=[f['name'] for f in self.model['stateSpace']['factors']],
            modality_names=[m['name'] for m in self.model['observations']['modalities']],
            save_to=os.path.join(self.output_dir, 'structure', 'model_graph.png')
        )
        
    def plot_belief_history(self, belief_history):
        """Plot belief evolution over time"""
        mvis.plot_belief_history(
            belief_history,
            factor_names=[f['name'] for f in self.model['stateSpace']['factors']],
            save_to=os.path.join(self.output_dir, "states", "belief_history.png")
        )
