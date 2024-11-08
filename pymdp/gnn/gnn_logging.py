import logging
import os
from datetime import datetime
import numpy as np

class GNNLogger:
    """Logging utilities for GNN-based active inference agents"""
    
    def __init__(self, agent):
        """Initialize logger for a GNN agent"""
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Set up logging configuration"""
        # Create log directory in sandbox
        log_dir = os.path.join(self.agent.sandbox_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        log_file = os.path.join(log_dir, "gnn_agent.log")
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info("Logging initialized")
        
    def create_sandbox_dir(self):
        """Create sandbox directory for experiment outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_dir = os.path.join("pymdp", "sandbox", f"gnn_agent_{timestamp}")
        os.makedirs(sandbox_dir, exist_ok=True)
        self.logger.info(f"Created sandbox directory: {sandbox_dir}")
        return sandbox_dir
        
    def log_model_loading(self, model_type, path):
        """Log model loading events"""
        self.logger.info(f"Loading {model_type} GNN model from: {path}")
        
    def log_inference(self, obs, beliefs):
        """Log state inference results"""
        self.logger.debug(f"Observation: {obs}")
        self.logger.debug(f"Updated beliefs: {[b.round(3) for b in beliefs]}")
        
    def log_action(self, action):
        """Log action selection"""
        self.logger.debug(f"Selected action: {action}")
        
    def log_error(self, error, context=""):
        """Log error events"""
        self.logger.error(f"Error during {context}: {str(error)}")
        self.logger.error("Traceback:", exc_info=True)
        
    def log_model_structure(self, agent):
        """Log complete model structure before inference"""
        self.logger.info("\n=== Complete Model Structure ===\n")
        
        # Create visualization directory
        viz_dir = os.path.join(agent.sandbox_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Log and visualize state space structure
        self.logger.info("State Space Structure:")
        for i, factor in enumerate(agent.gnn['stateSpace']['factors']):
            self.logger.info(f"\nFactor {i}: {factor['name']}")
            self.logger.info(f"Number of states: {factor['num_states']}")
            self.logger.info(f"Controllable: {factor['controllable']}")
            if 'labels' in factor:
                self.logger.info(f"State labels: {factor['labels']}")
                
            # Visualize initial state distribution
            plot_matrix_heatmap(
                agent.agent_matrices['D'][i].reshape(-1, 1),
                title=f"Initial State Distribution - {factor['name']}",
                xlabel="State",
                ylabel="Probability",
                save_to=os.path.join(viz_dir, f"initial_dist_{factor['name']}.png")
            )
        
        # Log and visualize observation structure  
        self.logger.info("\nObservation Structure:")
        for i, modality in enumerate(agent.gnn['observations']['modalities']):
            self.logger.info(f"\nModality {i}: {modality['name']}")
            self.logger.info(f"Number of observations: {modality['num_observations']}")
            self.logger.info(f"Factors observed: {modality['factors_observed']}")
            if 'labels' in modality:
                self.logger.info(f"Observation labels: {modality['labels']}")
        
        # Log and visualize A matrices (observation model)
        self.logger.info("\nA Matrices (Observation Model):")
        for i, A in enumerate(agent.agent_matrices['A']):
            modality = agent.gnn['observations']['modalities'][i]['name']
            self.logger.info(f"\nA Matrix for {modality}:")
            self.logger.info(f"Shape: {A.shape}")
            with np.printoptions(precision=3, suppress=True):
                self.logger.info(f"Values:\n{A}")
                
            # Visualize A matrix
            plot_matrix_heatmap(
                A,
                title=f"Observation Model - {modality}",
                xlabel="Hidden States",
                ylabel="Observations",
                save_to=os.path.join(viz_dir, f"A_matrix_{modality}.png")
            )
        
        # Log and visualize B matrices (transition model)
        self.logger.info("\nB Matrices (Transition Model):")
        for i, B in enumerate(agent.agent_matrices['B']):
            factor = agent.gnn['stateSpace']['factors'][i]['name']
            self.logger.info(f"\nB Matrix for {factor}:")
            self.logger.info(f"Shape: {B.shape}")
            
            if len(B.shape) == 3:  # Controllable factor
                for a in range(B.shape[2]):
                    self.logger.info(f"\nAction {a}:")
                    with np.printoptions(precision=3, suppress=True):
                        self.logger.info(f"{B[:,:,a]}")
                        
                    # Visualize B matrix for each action
                    plot_matrix_heatmap(
                        B[:,:,a],
                        title=f"Transition Model - {factor} (Action {a})",
                        xlabel="Current State",
                        ylabel="Next State", 
                        save_to=os.path.join(viz_dir, f"B_matrix_{factor}_action_{a}.png")
                    )
            else:
                with np.printoptions(precision=3, suppress=True):
                    self.logger.info(f"{B}")
                    
                # Visualize B matrix
                plot_matrix_heatmap(
                    B,
                    title=f"Transition Model - {factor}",
                    xlabel="Current State",
                    ylabel="Next State",
                    save_to=os.path.join(viz_dir, f"B_matrix_{factor}.png")
                )
        
        # Log and visualize C matrices (preferences) if present
        if 'C' in agent.agent_matrices:
            self.logger.info("\nC Matrices (Preferences):")
            for i, C in enumerate(agent.agent_matrices['C']):
                modality = agent.gnn['observations']['modalities'][i]['name']
                self.logger.info(f"\nC Matrix for {modality}:")
                self.logger.info(f"Shape: {C.shape}")
                with np.printoptions(precision=3, suppress=True):
                    self.logger.info(f"Values:\n{C}")
                    
                # Visualize C matrix
                plot_matrix_heatmap(
                    C.reshape(-1, 1),
                    title=f"Preferences - {modality}",
                    xlabel="Observation",
                    ylabel="Preference Value",
                    save_to=os.path.join(viz_dir, f"C_matrix_{modality}.png")
                )
        
        # Log and visualize D matrices (initial state priors)
        self.logger.info("\nD Matrices (Initial State Priors):")
        for i, D in enumerate(agent.agent_matrices['D']):
            factor = agent.gnn['stateSpace']['factors'][i]['name']
            self.logger.info(f"\nD Matrix for {factor}:")
            self.logger.info(f"Shape: {D.shape}")
            with np.printoptions(precision=3, suppress=True):
                self.logger.info(f"Values:\n{D}")
        
        # Log policies if present
        if 'policies' in agent.gnn:
            self.logger.info("\nPolicy Structure:")
            self.logger.info(f"Number of policies: {agent.gnn['policies']['num_policies']}")
            self.logger.info(f"Policy length: {agent.gnn['policies']['policy_len']}")
            self.logger.info(f"Control factors: {agent.gnn['policies']['control_fac_idx']}")
        
        # Log parameters
        self.logger.info("\nParameters:")
        for param, value in agent.parameters.items():
            self.logger.info(f"{param}: {value}")
        
        # Plot model structure graph
        plot_model_graph(
            agent.agent_matrices['A'],
            factor_names=[f['name'] for f in agent.gnn['stateSpace']['factors']],
            modality_names=[m['name'] for m in agent.gnn['observations']['modalities']],
            save_to=os.path.join(viz_dir, "model_structure.png")
        )
        
        self.logger.info("\n=== End Model Structure ===\n")