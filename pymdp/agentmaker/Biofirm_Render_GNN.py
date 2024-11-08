import os
import sys
import json
import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import argparse

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

# Set default matplotlib style
plt.style.use('default')  # Use default style instead of seaborn
sns.set_theme(style="whitegrid")  # Configure seaborn theme directly

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import local modules
from pymdp.gnn.gnn_matrix_factory import GNNMatrixFactory
from pymdp.gnn.biofirm_utils import BiofirmUtils

# Setup logging
logger = logging.getLogger(__name__)

# Default GNN paths relative to this file
DEFAULT_ENV_GNN = Path(__file__).parent.parent / "gnn" / "envs" / "gnn_env_biofirm.gnn"
DEFAULT_AGENT_GNN = Path(__file__).parent.parent / "gnn" / "models" / "gnn_agent_biofirm.gnn"

class BiofirmRenderer:
    """Render matrices for Biofirm active inference simulation"""
    
    def __init__(self, env_gnn: str, agent_gnn: str, output_dir: Optional[str] = None):
        """Initialize renderer with GNN model paths"""
        try:
            # Configure visualization style
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.5)
            
            # Set default color palette
            sns.set_palette("husl")
            
            # Configure matplotlib params
            plt.rcParams.update({
                'figure.figsize': [8.0, 6.0],
                'figure.dpi': 100,
                'savefig.dpi': 300,
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12
            })
            
            # Store GNN paths
            self.env_gnn = env_gnn
            self.agent_gnn = agent_gnn
            
            # Setup base output directory first
            self.output_dir = Path(output_dir) if output_dir else self._create_output_dir()
            
            # Create complete directory structure upfront
            self.directories = {
                'root': self.output_dir,
                'matrices': self.output_dir / "matrices",
                'matrices_env': self.output_dir / "matrices" / "environment",
                'matrices_agent': self.output_dir / "matrices" / "agent", 
                'config': self.output_dir / "config",
                'viz': self.output_dir / "matrix_visualizations",
                'logs': self.output_dir / "logs"
            }
            
            # Create all directories
            for dir_path in self.directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                
            # Now setup logging after directories exist
            self.logger = self._setup_logging()
            
            self.logger.info("==================================================")
            self.logger.info("Initializing BiofirmRenderer")
            self.logger.info(f"Output directory structure created at: {self.output_dir}")
            for name, path in self.directories.items():
                self.logger.info(f"- {name}: {path}")
            self.logger.info("==================================================")
            
            # Load and validate GNN models
            self.logger.info("Loading GNN models...")
            
            # Create environment matrices with minimal validation
            env_factory = GNNMatrixFactory(env_gnn, model_type='environment')
            env_factory.required_sections = ['stateSpace', 'observations', 'transitionModel']
            self.env_matrices = env_factory.create_matrices()
            self.env_model = env_factory.model
            
            # Add required sections to environment model
            self.env_model = self._validate_env_model_structure(self.env_model)
            
            # Create agent matrices with full validation
            agent_factory = GNNMatrixFactory(agent_gnn, model_type='agent')
            agent_factory.required_sections = [
                'stateSpace', 
                'observations', 
                'transitionModel',
                'policies',
                'preferences'
            ]
            self.agent_matrices = agent_factory.create_matrices()
            self.agent_model = agent_factory.model
            
            # Validate agent model structure
            self._validate_agent_model_structure(self.agent_model)
            
            # Validate matrices
            self.logger.info("Validating matrices...")
            
            # First normalize matrices if needed
            self.env_matrices = self._normalize_matrices(self.env_matrices)
            self.agent_matrices = self._normalize_matrices(self.agent_matrices)
            
            # Then validate
            BiofirmUtils.validate_biofirm_matrices(self.env_matrices, "environment")
            BiofirmUtils.validate_biofirm_matrices(self.agent_matrices, "agent")
            
            # Save model files
            self._save_model_files()
            
            self.logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise

    def _create_output_dir(self) -> Path:
        """Create timestamped output directory with validation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).parent / "sandbox" / "Biofirm"
        output_dir = base_dir / f"render_{timestamp}"
        
        # Ensure parent directories exist
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for renderer"""
        # Create logger
        logger = logging.getLogger('biofirm_renderer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.output_dir / 'render.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatters
        console_formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatters to handlers
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

    def render_matrices(self) -> Path:
        """Render all matrices and save to output directory"""
        try:
            # Create environment matrices
            env_factory = GNNMatrixFactory(self.env_gnn, model_type='environment')
            self.env_matrices = env_factory.create_matrices()
            
            # Create agent matrices
            agent_factory = GNNMatrixFactory(self.agent_gnn, model_type='agent')
            self.agent_matrices = agent_factory.create_matrices()
            
            # Save matrices directly (don't use factory save method)
            matrices_dir = self.directories['matrices']
            
            # Save environment matrices
            env_dir = matrices_dir / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)
            
            # Save A matrix (observation model)
            A = self.env_matrices['A'][0]  # Get first A matrix
            logger.info(f"Saving environment A matrix with shape {A.shape}")
            np.save(env_dir / "A_matrices.npy", A)
            
            # Save B matrix (transition model)
            B = self.env_matrices['B'][0]  # Get first B matrix
            logger.info(f"Saving environment B matrix with shape {B.shape}")
            np.save(env_dir / "B_matrices.npy", B)
            
            # Save D matrix (initial state)
            D = self.env_matrices['D'][0]  # Get first D matrix
            logger.info(f"Saving environment D matrix with shape {D.shape}")
            np.save(env_dir / "D_matrices.npy", D)
            
            # Save agent matrices
            agent_dir = matrices_dir / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            
            # Save A matrix
            A = self.agent_matrices['A'][0]
            logger.info(f"Saving agent A matrix with shape {A.shape}")
            np.save(agent_dir / "A_matrices.npy", A)
            
            # Save B matrix
            B = self.agent_matrices['B'][0]
            logger.info(f"Saving agent B matrix with shape {B.shape}")
            np.save(agent_dir / "B_matrices.npy", B)
            
            # Save C matrix
            C = self.agent_matrices['C'][0]
            logger.info(f"Saving agent C matrix with shape {C.shape}")
            np.save(agent_dir / "C_matrices.npy", C)
            
            # Save D matrix
            D = self.agent_matrices['D'][0]
            logger.info(f"Saving agent D matrix with shape {D.shape}")
            np.save(agent_dir / "D_matrices.npy", D)
            
            # Verify files exist and can be loaded
            logger.info("Verifying saved matrices...")
            
            # Check environment matrices
            for name, path in [
                ('A', env_dir / "A_matrices.npy"),
                ('B', env_dir / "B_matrices.npy"),
                ('D', env_dir / "D_matrices.npy")
            ]:
                if not path.exists():
                    raise RuntimeError(f"Failed to save environment {name} matrix")
                matrix = np.load(path)
                logger.info(f"Verified environment {name} matrix: shape {matrix.shape}")
                
            # Check agent matrices
            for name, path in [
                ('A', agent_dir / "A_matrices.npy"),
                ('B', agent_dir / "B_matrices.npy"),
                ('C', agent_dir / "C_matrices.npy"),
                ('D', agent_dir / "D_matrices.npy")
            ]:
                if not path.exists():
                    raise RuntimeError(f"Failed to save agent {name} matrix")
                matrix = np.load(path)
                logger.info(f"Verified agent {name} matrix: shape {matrix.shape}")
            
            # Generate visualizations
            self._generate_matrix_visualizations()
            
            # Save model configurations
            self._save_model_configs()
            
            return matrices_dir
            
        except Exception as e:
            logger.error(f"Error rendering matrices: {str(e)}")
            logger.error("Directory contents:")
            if hasattr(self, 'directories'):
                for name, path in self.directories.items():
                    if path.exists():
                        logger.error(f"{name}:")
                        for f in path.glob('*'):
                            logger.error(f"- {f.name}")
            raise

    def _generate_matrix_visualizations(self):
        """Generate comprehensive matrix visualizations"""
        try:
            viz_dir = self.directories['viz']
            
            # Environment matrix visualizations
            env_viz_dir = viz_dir / "environment"
            env_viz_dir.mkdir(exist_ok=True)
            
            # Set consistent visualization parameters
            viz_params = {
                'cmap': 'viridis',
                'annot': True,
                'fmt': '.2f',
                'square': True,
                'cbar_kws': {'shrink': .8}
            }
            
            # Use with statement to ensure figures are closed properly
            with plt.style.context('default'):
                # A matrices (observation model)
                for i, A in enumerate(self.env_matrices['A']):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(A, ax=ax, **viz_params)
                    ax.set_title("Environment Observation Model (A Matrix)")
                    ax.set_xlabel("Hidden States")
                    ax.set_ylabel("Observations")
                    fig.tight_layout()
                    fig.savefig(env_viz_dir / f"A_matrix_{i}.png", bbox_inches='tight')
                    plt.close(fig)
                    
                # B matrices (transition model)
                for i, B in enumerate(self.env_matrices['B']):
                    for action in range(B.shape[-1]):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(B[..., action], annot=True, fmt='.2f', cmap='viridis', ax=ax)
                        ax.set_title(f"Environment Transition Model - Action {action}")
                        ax.set_xlabel("Current State")
                        ax.set_ylabel("Next State")
                        fig.tight_layout()
                        fig.savefig(env_viz_dir / f"B_matrix_{i}_action_{action}.png")
                        plt.close(fig)
                        
                # D matrices (initial state)
                for i, D in enumerate(self.env_matrices['D']):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.barplot(x=range(len(D)), y=D, ax=ax)
                    ax.set_title("Environment Initial State Distribution")
                    ax.set_xlabel("State")
                    ax.set_ylabel("Probability")
                    fig.tight_layout()
                    fig.savefig(env_viz_dir / f"D_matrix_{i}.png")
                    plt.close(fig)
                    
                # Agent matrix visualizations
                agent_viz_dir = viz_dir / "agent"
                agent_viz_dir.mkdir(exist_ok=True)
                
                # A matrices
                for i, A in enumerate(self.agent_matrices['A']):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(A, annot=True, fmt='.2f', cmap='viridis', ax=ax)
                    ax.set_title(f"Agent Observation Model (A Matrix)")
                    ax.set_xlabel("Hidden States")
                    ax.set_ylabel("Observations")
                    fig.tight_layout()
                    fig.savefig(agent_viz_dir / f"A_matrix_{i}.png")
                    plt.close(fig)
                    
                # B matrices
                for i, B in enumerate(self.agent_matrices['B']):
                    for action in range(B.shape[-1]):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(B[..., action], annot=True, fmt='.2f', cmap='viridis', ax=ax)
                        ax.set_title(f"Agent Transition Model - Action {action}")
                        ax.set_xlabel("Current State")
                        ax.set_ylabel("Next State")
                        fig.tight_layout()
                        fig.savefig(agent_viz_dir / f"B_matrix_{i}_action_{action}.png")
                        plt.close(fig)
                        
                # C matrices (preferences)
                for i, C in enumerate(self.agent_matrices['C']):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.barplot(x=range(len(C)), y=C, ax=ax)
                    ax.set_title("Agent Preferences")
                    ax.set_xlabel("Observation")
                    ax.set_ylabel("Preference Value")
                    fig.tight_layout()
                    fig.savefig(agent_viz_dir / f"C_matrix_{i}.png")
                    plt.close(fig)
                    
                # D matrices
                for i, D in enumerate(self.agent_matrices['D']):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.barplot(x=range(len(D)), y=D, ax=ax)
                    ax.set_title("Agent Initial Beliefs")
                    ax.set_xlabel("State")
                    ax.set_ylabel("Probability")
                    fig.tight_layout()
                    fig.savefig(agent_viz_dir / f"D_matrix_{i}.png")
                    plt.close(fig)
                    
            self.logger.info(f"Generated matrix visualizations in {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            # Clean up any open figures
            plt.close('all')
            raise

    def _verify_matrix_files(self, matrices_dir: Path):
        """Verify all required matrix files exist"""
        # Check environment matrices
        env_dir = matrices_dir / "environment"
        required_env_files = ['A_matrices.npy', 'B_matrices.npy', 'D_matrices.npy']
        for f in required_env_files:
            if not (env_dir / f).exists():
                raise RuntimeError(f"Failed to save environment matrix: {f}")
                
        # Check agent matrices
        agent_dir = matrices_dir / "agent"
        required_agent_files = ['A_matrices.npy', 'B_matrices.npy', 'C_matrices.npy', 'D_matrices.npy']
        for f in required_agent_files:
            if not (agent_dir / f).exists():
                raise RuntimeError(f"Failed to save agent matrix: {f}")
                
        self.logger.info("All matrix files saved successfully")

    def _save_model_configs(self):
        """Save model configurations to JSON files"""
        try:
            config_dir = self.directories['config']
            
            # Save environment model config
            env_config_file = config_dir / "environment_model.json"
            with open(env_config_file, 'w') as f:
                json.dump(self.env_model, f, indent=2)
                
            # Save agent model config
            agent_config_file = config_dir / "agent_model.json"
            with open(agent_config_file, 'w') as f:
                json.dump(self.agent_model, f, indent=2)
                
            # Save simulation parameters
            sim_params = {
                'timesteps': 1000,
                'parameters': {
                    'ecological_noise': 0.1,
                    'controllability': 0.8,
                    'policy_precision': 16.0,
                    'use_states_info_gain': True
                }
            }
            
            params_file = config_dir / "simulation_params.json"
            with open(params_file, 'w') as f:
                json.dump(sim_params, f, indent=2)
                
            self.logger.info(f"Saved model configurations to {config_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model configs: {str(e)}")
            raise

    def _normalize_matrices(self, matrices: Dict) -> Dict:
        """Normalize probability matrices to ensure they sum to 1"""
        normalized = matrices.copy()
        
        # Normalize A matrix (observation model)
        if 'A' in normalized:
            for i in range(len(normalized['A'])):
                A = normalized['A'][i]
                # Normalize columns (each column should sum to 1)
                normalized['A'][i] = A / A.sum(axis=0, keepdims=True)
                
        # Normalize B matrix (transition model)
        if 'B' in normalized:
            for i in range(len(normalized['B'])):
                B = normalized['B'][i]
                # Normalize for each action
                for action in range(B.shape[-1]):
                    B[..., action] = B[..., action] / B[..., action].sum(axis=0, keepdims=True)
                normalized['B'][i] = B
                
        # Normalize D matrix (initial state prior)
        if 'D' in normalized:
            for i in range(len(normalized['D'])):
                D = normalized['D'][i]
                normalized['D'][i] = D / D.sum()
                
        return normalized

    def _validate_env_model_structure(self, model: Dict) -> Dict:
        """Validate and fix environment model structure, returning updated model"""
        try:
            # Create copy to avoid modifying original
            model = model.copy()
            
            # Check and fix required sections
            required_base = ['stateSpace', 'observations', 'transitionModel']
            for section in required_base:
                if section not in model:
                    raise ValueError(f"Environment model missing required section: {section}")
            
            # Add or update policies section
            if 'policies' not in model:
                model['policies'] = {
                    'controlFactors': [0],  # First factor is controllable
                    'numControls': [3],     # 3 possible actions
                    'policyLen': 1,         # Single-step policies
                    'control_fac_idx': [0], # Control factor indices
                    'labels': [['DECREASE', 'MAINTAIN', 'INCREASE']]
                }
                self.logger.info("Added default policies section to environment model")
            else:
                # Ensure policies section has required fields
                policies = model['policies']
                if 'controlFactors' not in policies:
                    policies['controlFactors'] = [0]
                if 'numControls' not in policies:
                    policies['numControls'] = [3]
                if 'policyLen' not in policies:
                    policies['policyLen'] = 1
                if 'control_fac_idx' not in policies:
                    policies['control_fac_idx'] = [0]
                if 'labels' not in policies:
                    policies['labels'] = [['DECREASE', 'MAINTAIN', 'INCREASE']]
            
            # Ensure state space structure
            state_space = model.get('stateSpace', {})
            if 'factors' not in state_space:
                state_space['factors'] = ['EcologicalState']
            if 'sizes' not in state_space:
                state_space['sizes'] = [3]  # LOW, HOMEO, HIGH
            if 'labels' not in state_space:
                state_space['labels'] = [['LOW', 'HOMEO', 'HIGH']]
            model['stateSpace'] = state_space
            
            # Ensure observation structure
            obs = model.get('observations', {})
            if 'modalities' not in obs:
                obs['modalities'] = ['StateObservation']
            if 'sizes' not in obs:
                obs['sizes'] = [3]  # LOW, HOMEO, HIGH
            if 'labels' not in obs:
                obs['labels'] = [['LOW', 'HOMEO', 'HIGH']]
            model['observations'] = obs
            
            # Add preferences section if missing (for environment model)
            if 'preferences' not in model:
                model['preferences'] = {
                    'modalities': ['StateObservation'],
                    'values': [[0.0, 1.0, 0.0]],  # Prefer HOMEO state
                    'labels': [['LOW', 'HOMEO', 'HIGH']]
                }
                self.logger.info("Added default preferences to environment model")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error validating environment model: {str(e)}")
            raise

    def _validate_agent_model_structure(self, model: Dict) -> None:
        """Validate and fix agent model structure"""
        required_sections = [
            'stateSpace',
            'observations',
            'transitionModel',
            'policies',
            'preferences'
        ]
        
        for section in required_sections:
            if section not in model:
                raise ValueError(f"Agent model missing required section: {section}")
                
        # Validate policies section
        policies = model['policies']
        if 'controlFactors' not in policies or 'numControls' not in policies:
            raise ValueError("Agent model policies section missing required fields")
            
        # Validate preferences section
        preferences = model['preferences']
        if 'modalities' not in preferences or 'values' not in preferences:
            raise ValueError("Agent model preferences section missing required fields")
            
        # Ensure state space structure
        if 'stateSpace' in model:
            state_space = model['stateSpace']
            if 'factors' not in state_space:
                state_space['factors'] = ['EcologicalState']
            if 'sizes' not in state_space:
                state_space['sizes'] = [3]  # LOW, HOMEO, HIGH
                
        # Ensure observation structure
        if 'observations' in model:
            obs = model['observations']
            if 'modalities' not in obs:
                obs['modalities'] = ['StateObservation']
            if 'sizes' not in obs:
                obs['sizes'] = [3]  # LOW, HOMEO, HIGH

    def _save_model_files(self):
        """Save model definitions and configuration"""
        try:
            # Create config directory
            config_dir = self.output_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Use validate_env_model_structure to get updated model
            env_model = self._validate_env_model_structure(self.env_model)
            
            # Save environment model
            env_file = config_dir / "environment_model.json"
            with open(env_file, 'w') as f:
                json.dump(env_model, f, indent=2)
            self.logger.info("Saved environment files to config directory")
            
            # Save agent model
            agent_file = config_dir / "agent_model.json"
            with open(agent_file, 'w') as f:
                json.dump(self.agent_model, f, indent=2)
            self.logger.info("Saved agent files to config directory")
            
            # Save simulation parameters
            params = BiofirmUtils.get_default_simulation_params()
            param_file = config_dir / "simulation_params.json"
            with open(param_file, 'w') as f:
                json.dump(params, f, indent=2)
                
            # Save matrix validation report
            validation_report = {
                'environment': {
                    'matrices_validated': True,
                    'warnings': [],
                    'timestamp': datetime.now().isoformat()
                },
                'agent': {
                    'matrices_validated': True,
                    'warnings': [],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            validation_file = config_dir / "validation_report.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving model files: {str(e)}")
            raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Render Biofirm matrices from GNN models')
    parser.add_argument('--env_gnn', type=str, required=True,
                      help='Path to environment GNN model file')
    parser.add_argument('--agent_gnn', type=str, required=True,
                      help='Path to agent GNN model file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for rendered matrices')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--clean', action='store_true',
                      help='Clean output directory before rendering')
    return parser.parse_args()

def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set logging level based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info("Starting Biofirm matrix rendering")
        logger.info(f"Environment GNN: {args.env_gnn}")
        logger.info(f"Agent GNN: {args.agent_gnn}")
        
        # Clean output directory if requested
        output_dir = Path(args.output_dir)
        if args.clean and output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"Cleaned output directory: {output_dir}")
        
        # Verify GNN files exist
        for gnn_file, file_type in [(args.env_gnn, "Environment"), (args.agent_gnn, "Agent")]:
            if not os.path.exists(gnn_file):
                logger.error(f"{file_type} GNN file not found: {gnn_file}")
                sys.exit(1)
        
        # Create renderer and generate matrices
        renderer = BiofirmRenderer(
            env_gnn=args.env_gnn,
            agent_gnn=args.agent_gnn,
            output_dir=args.output_dir
        )
        
        # Render matrices
        matrices_dir = renderer.render_matrices()
        
        # Verify output
        if not matrices_dir.exists():
            logger.error(f"Matrix rendering failed: {matrices_dir} not created")
            sys.exit(1)
            
        if not any(matrices_dir.iterdir()):
            logger.error(f"Matrix rendering failed: {matrices_dir} is empty")
            sys.exit(1)
            
        # Verify specific matrix files
        env_dir = matrices_dir / "environment"
        agent_dir = matrices_dir / "agent"
        
        required_env_files = ['A_matrices.npy', 'B_matrices.npy', 'D_matrices.npy']
        required_agent_files = ['A_matrices.npy', 'B_matrices.npy', 'C_matrices.npy', 'D_matrices.npy']
        
        missing_files = False
        for d, files in [(env_dir, required_env_files), (agent_dir, required_agent_files)]:
            for f in files:
                if not (d / f).exists():
                    logger.error(f"Missing required file: {d/f}")
                    missing_files = True
                else:
                    logger.info(f"Verified file: {d/f}")
                    
        if missing_files:
            sys.exit(1)
        
        logger.info(f"\nMatrix rendering complete.")
        logger.info(f"Outputs saved to: {matrices_dir}")
        logger.info("\nYou can now run Biofirm_Execute_GNN.py using this matrices directory")
        
    except Exception as e:
        logger.error(f"Error during matrix rendering: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
