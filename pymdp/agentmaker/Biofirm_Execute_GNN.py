# Add at top of file
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now do imports
import argparse
import numpy as np
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Union, Any, Tuple

# Import local modules
from pymdp.gnn.gnn_utils import GNNUtils
from pymdp.agentmaker.Active_Inference_GNN import ActiveInferenceGNN
from pymdp.agentmaker.biofirm_execution_utils import BiofirmExecutionUtils

# Update logger initialization at top of file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class BiofirmExecutor:
    """Execute active inference simulation using rendered matrices"""
    
    def __init__(self, config_file: Path):
        """Initialize executor with config file"""
        try:
            # Load config
            with open(config_file) as f:
                self.config = json.load(f)
                
            # Get directories from config
            self.matrices_dir = Path(self.config['matrices_dir'])
            self.output_dir = Path(self.config['output_dir'])
            
            # Create complete directory structure upfront
            self.directories = {
                'root': self.output_dir,
                'data': self.output_dir / "data",
                'results': self.output_dir / "results",
                'logs': self.output_dir / "logs",
                'config': self.output_dir / "config",
                'metrics': self.output_dir / "metrics",
                'checkpoints': self.output_dir / "checkpoints"
            }
            
            # Create all directories upfront with error checking
            for dir_name, dir_path in self.directories.items():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {dir_path}")  # Use print since logger not setup yet
                except Exception as e:
                    print(f"Failed to create {dir_name} directory at {dir_path}: {str(e)}")
                    raise

            # Setup logging after directories exist
            self._setup_logging()
            
            # Load matrices
            logger.info("Loading matrices...")
            
            # Load environment matrices
            env_dir = self.matrices_dir / "environment"
            self.env_matrices = {
                'A': np.load(env_dir / "A_matrices.npy"),
                'B': np.load(env_dir / "B_matrices.npy"),
                'D': np.load(env_dir / "D_matrices.npy")
            }
            logger.info("Loaded environment matrices:")
            for name, matrix in self.env_matrices.items():
                logger.info(f"- {name}: shape {matrix.shape}")

            # Load agent matrices
            agent_dir = self.matrices_dir / "agent"
            self.agent_matrices = {
                'A': np.load(agent_dir / "A_matrices.npy"),
                'B': np.load(agent_dir / "B_matrices.npy"),
                'C': np.load(agent_dir / "C_matrices.npy"),
                'D': np.load(agent_dir / "D_matrices.npy")
            }
            logger.info("Loaded agent matrices:")
            for name, matrix in self.agent_matrices.items():
                logger.info(f"- {name}: shape {matrix.shape}")

            # Initialize active inference agent
            self.agent = ActiveInferenceGNN(
                A=self.agent_matrices['A'],
                B=self.agent_matrices['B'],
                C=self.agent_matrices['C'],
                D=self.agent_matrices['D'],
                control_fac_idx=[0],
                gamma=self.config.get('parameters', {}).get('policy_precision', 16.0),
                use_states_info_gain=self.config.get('parameters', {}).get('use_states_info_gain', True)
            )
            logger.info("Initialized active inference agent")

            # Store matrices for saving metadata later
            self.A = self.agent_matrices['A']
            self.B = self.agent_matrices['B']
            self.C = self.agent_matrices['C']
            self.D = self.agent_matrices['D']

            logger.info("BiofirmExecutor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing BiofirmExecutor: {str(e)}")
            raise
            
    def _create_output_dir(self) -> Path:
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / "sandbox" / "Biofirm" / f"simulation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
        
    def run_simulation(self):
        """Run active inference simulation"""
        try:
            # Get simulation parameters from config
            timesteps = self.config.get('timesteps', 1000)  # Default to 1000 if not specified
            logger.info(f"Running simulation for {timesteps} timesteps")
            
            # Initialize histories with only available metrics
            histories = {
                'observations': np.zeros(timesteps, dtype=np.int32),
                'actions': np.zeros(timesteps, dtype=np.int32),
                'beliefs': np.zeros((timesteps, 3), dtype=np.float64),
                'true_states': np.zeros(timesteps, dtype=np.int32),
                'policy_probs': np.zeros((timesteps, 3), dtype=np.float64),
                'belief_entropy': np.zeros(timesteps, dtype=np.float64),
                'policy_entropy': np.zeros(timesteps, dtype=np.float64),
                'accuracy': np.zeros(timesteps, dtype=np.float64)
            }
            
            # Initialize state and observation
            current_state = np.random.choice(3, p=self.env_matrices['D'])
            current_obs = self._generate_observation(current_state)
            
            # Run simulation
            logger.info(f"Starting simulation for {timesteps} timesteps")
            
            for t in range(timesteps):
                # Store current state and observation
                histories['true_states'][t] = current_state
                histories['observations'][t] = current_obs
                
                # State inference
                qs = self.agent.infer_states(obs=current_obs)
                histories['beliefs'][t] = qs
                
                # Policy inference
                q_pi = self.agent.infer_policies()
                histories['policy_probs'][t] = q_pi
                
                # Action selection
                action = self.agent.sample_action()
                histories['actions'][t] = action
                
                # Calculate metrics directly
                histories['belief_entropy'][t] = -np.sum(qs * np.log(qs + 1e-16))
                histories['policy_entropy'][t] = -np.sum(q_pi * np.log(q_pi + 1e-16))
                histories['accuracy'][t] = float(np.argmax(qs) == current_state)
                
                # Environment dynamics
                current_state = self._state_transition(current_state, action)
                current_obs = self._generate_observation(current_state)
                
                if t % 100 == 0:
                    logger.info(f"Timestep {t}: state={current_state}, obs={current_obs}, action={action}")
                    
            # Save all data
            self._save_histories(histories)
            self._save_summary(histories)
            self._save_metrics(histories)
            self._save_checkpoint()
            
            # Save metadata with updated metrics
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'matrix_shapes': {
                    'A': self.A.shape,
                    'B': self.B.shape,
                    'C': self.C.shape,
                    'D': self.D.shape
                },
                'simulation_info': {
                    'total_timesteps': timesteps,
                    'final_state': int(histories['true_states'][-1]),
                    'final_observation': int(histories['observations'][-1]),
                    'time_in_homeo': float(np.mean(histories['true_states'] == 1)),
                    'mean_accuracy': float(np.mean(histories['accuracy'])),
                    'mean_belief_entropy': float(np.mean(histories['belief_entropy'])),
                    'mean_policy_entropy': float(np.mean(histories['policy_entropy']))
                }
            }
            
            metadata_file = self.directories['data'] / "simulation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Update metrics saving
            self._save_metrics(histories)
            
            logger.info("Simulation complete")
            
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            raise
            
    def _save_summary(self, histories: Dict):
        """Save summary statistics"""
        summary = {
            'mean_beliefs': np.mean(histories['beliefs'], axis=0).tolist(),
            'action_frequencies': np.bincount(histories['actions']).tolist(),
            'final_state': int(histories['true_states'][-1]),
            'total_timesteps': len(histories['observations']),
            'homeostasis_metrics': {
                'time_in_homeo': float(np.mean(histories['true_states'] == 1)),
                'transitions': int(len(np.where(np.diff(histories['true_states']))[0])),
                'control_efficiency': float(np.mean(histories['actions'] == 1))
            }
        }
        
        summary_file = self.directories['results'] / "simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
    def _save_histories(self, histories: Dict):
        """Save simulation histories and results"""
        try:
            # Save raw histories to data directory
            histories_file = self.directories['data'] / "histories.json"
            json_histories = {k: v.tolist() for k, v in histories.items()}
            with open(histories_file, 'w') as f:
                json.dump(json_histories, f, indent=2)
            logger.info(f"Saved histories to {histories_file}")
            
            # Save results summary to results directory
            results = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'mean_beliefs': np.mean(histories['beliefs'], axis=0).tolist(),
                    'action_frequencies': np.bincount(histories['actions']).tolist(),
                    'final_state': int(histories['true_states'][-1]),
                    'total_timesteps': len(histories['observations']),
                    'homeostasis_metrics': {
                        'time_in_homeo': float(np.mean(histories['true_states'] == 1)),
                        'transitions': int(len(np.where(np.diff(histories['true_states']))[0])),
                        'control_efficiency': float(np.mean(histories['actions'] == 1))
                    }
                }
            }
            
            results_file = self.directories['results'] / "simulation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {results_file}")
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'matrix_shapes': {
                    'A': self.A.shape,
                    'B': self.B.shape,
                    'C': self.C.shape,
                    'D': self.D.shape
                }
            }
            
            metadata_file = self.directories['data'] / "simulation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_file}")
            
            # Verify files were saved
            if not histories_file.exists():
                raise RuntimeError(f"Failed to save histories file: {histories_file}")
            if not results_file.exists():
                raise RuntimeError(f"Failed to save results file: {results_file}")
            if not metadata_file.exists():
                raise RuntimeError(f"Failed to save metadata file: {metadata_file}")
                
            logger.info(f"Saved simulation data to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving simulation data: {str(e)}")
            raise
            
    def _save_metrics(self, histories: Dict):
        """Save performance metrics"""
        try:
            metrics = {
                'belief_metrics': {
                    'entropy': histories['belief_entropy'].tolist(),
                    'mean_entropy': float(np.mean(histories['belief_entropy'])),
                    'accuracy': float(np.mean(histories['accuracy']))
                },
                'policy_metrics': {
                    'entropy': histories['policy_entropy'].tolist(),
                    'mean_entropy': float(np.mean(histories['policy_entropy'])),
                    'action_frequencies': np.bincount(histories['actions']).tolist()
                },
                'homeostatic_metrics': {
                    'time_in_homeo': float(np.mean(histories['true_states'] == 1)),
                    'transitions': int(len(np.where(np.diff(histories['true_states']))[0])),
                    'control_efficiency': float(np.mean(histories['actions'] == 1))
                }
            }
            
            metrics_file = self.directories['metrics'] / "performance_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise
            
    def _save_checkpoint(self):
        """Save checkpoint of agent state"""
        try:
            # Save checkpoint to checkpoints directory
            checkpoint_file = self.directories['checkpoints'] / "final_state.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'current_beliefs': self.agent.qs.tolist(),
                    'policy_probs': self.agent.qpi.tolist() if self.agent.qpi is not None else None,
                    'history_lengths': {k: len(v) for k, v in self.agent.history.items()}
                }, f, indent=2)
            logger.info(f"Saved checkpoint to {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def _save_initial_config(self):
        """Save initial configuration and metadata"""
        try:
            # Save simulation config
            config_file = self.directories['root'] / "simulation_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            # Save matrix metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'matrix_shapes': {
                    'environment': {
                        'A': self.env_matrices['A'].shape,
                        'B': self.env_matrices['B'].shape,
                        'D': self.env_matrices['D'].shape
                    },
                    'agent': {
                        'A': self.agent_matrices['A'].shape,
                        'B': self.agent_matrices['B'].shape,
                        'C': self.agent_matrices['C'].shape,
                        'D': self.agent_matrices['D'].shape
                    }
                },
                'directories': {k: str(v) for k, v in self.directories.items()}
            }
            
            metadata_file = self.directories['root'] / "simulation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Create empty results file to ensure directory validation passes
            results_file = self.directories['results'] / "simulation_summary.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'status': 'initialized',
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
            logger.info("Saved initial configuration and metadata")
            
        except Exception as e:
            logger.error(f"Error saving initial config: {str(e)}")
            raise

    def _validate_directories(self) -> bool:
        """Validate all required directories exist"""
        try:
            required_dirs = [
                'root',
                'data',
                'results',
                'logs',
                'config',
                'metrics',
                'checkpoints'
            ]
            
            missing_dirs = []
            for dir_name in required_dirs:
                if dir_name not in self.directories:
                    missing_dirs.append(dir_name)
                elif not self.directories[dir_name].exists():
                    missing_dirs.append(dir_name)
                    
            if missing_dirs:
                logger.error("Missing required directories:")
                for d in missing_dirs:
                    logger.error(f"- {d}")
                return False
                
            logger.info("Directory structure validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating directories: {str(e)}")
            return False

    def _generate_observation(self, state: int) -> int:
        """Generate observation from environment state"""
        obs_probs = self.env_matrices['A'][:, state]
        return np.random.choice(len(obs_probs), p=obs_probs)

    def _state_transition(self, state: int, action: int) -> int:
        """Execute state transition in environment"""
        trans_probs = self.env_matrices['B'][:, state, action]
        return np.random.choice(len(trans_probs), p=trans_probs)

    def _setup_logging(self):
        """Configure logging for executor"""
        try:
            # Create logs directory
            logs_dir = self.directories['logs']
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log file with timestamp
            log_file = logs_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
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
            
            logger.info(f"Logging initialized: {log_file}")
            
        except Exception as e:
            logger.error(f"Error setting up logging: {str(e)}")
            raise

class BiofirmSimulation:
    """Execute and analyze biofirm active inference simulations"""
    
    def __init__(self, config_file: Union[str, Path]):
        """Initialize simulation from config file"""
        try:
            self.config_file = Path(config_file)
            
            # Load config
            with open(self.config_file) as f:
                self.config = json.load(f)
                
            # Setup paths
            self.matrices_dir = Path(self.config['matrices_dir'])
            self.output_dir = Path(self.config['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup logging
            self._setup_logging()
            
            # Load and validate matrices
            logger.info("Loading matrices...")
            self.env_matrices = self._load_validated_matrices("environment")
            self.agent_matrices = self._load_validated_matrices("agent")
            
            # Initialize active inference
            self._initialize_active_inference()
            
        except Exception as e:
            logger.error(f"Error initializing simulation: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise
            
    def _setup_logging(self):
        """Configure logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "simulation.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)7s | %(message)s'
        ))
        logger.addHandler(file_handler)
        
    def _load_validated_matrices(self, model_type: str) -> Dict:
        """Load and validate matrices for environment or agent"""
        try:
            matrices_dir = self.matrices_dir / model_type
            
            # Load matrices
            matrices = {
                'A': np.load(matrices_dir / "A_matrices.npy"),
                'B': np.load(matrices_dir / "B_matrices.npy"),
                'D': np.load(matrices_dir / "D_matrices.npy")
            }
            
            # Add C matrix for agent
            if model_type == "agent":
                matrices['C'] = np.load(matrices_dir / "C_matrices.npy")
                
            # Validate shapes
            expected_shapes = {
                'A': (3, 3),
                'B': (3, 3, 3),
                'D': (3,)
            }
            if model_type == "agent":
                expected_shapes['C'] = (3,)
                
            for name, matrix in matrices.items():
                if matrix.shape != expected_shapes[name]:
                    raise ValueError(f"Invalid shape for {name} matrix: {matrix.shape} (expected {expected_shapes[name]})")
                    
            logger.info(f"Loaded {model_type} matrices:")
            for name, matrix in matrices.items():
                logger.info(f"- {name}: shape {matrix.shape}")
                
            return matrices
            
        except Exception as e:
            logger.error(f"Error loading {model_type} matrices: {str(e)}")
            raise
            
    def _initialize_active_inference(self):
        """Initialize active inference components"""
        try:
            # Get parameters from config
            params = self.config.get('parameters', {})
            
            # Create active inference agent
            self.agent = ActiveInferenceGNN(
                A=self.agent_matrices['A'],
                B=self.agent_matrices['B'],
                C=self.agent_matrices['C'],
                D=self.agent_matrices['D'],
                control_fac_idx=[0],
                policy_len=1,
                num_states=[3],
                num_controls=[3],
                gamma=params.get('policy_precision', 16.0),
                use_states_info_gain=params.get('use_states_info_gain', True)
            )
            
            logger.info("Initialized active inference agent")
            
        except Exception as e:
            logger.error(f"Error initializing active inference: {str(e)}")
            raise
            
    def run_simulation(self):
        """Run active inference simulation"""
        try:
            # Get simulation parameters
            timesteps = self.config.get('timesteps', 1000)
            
            # Initialize histories with only available metrics
            histories = {
                'observations': np.zeros(timesteps, dtype=np.int32),
                'actions': np.zeros(timesteps, dtype=np.int32),
                'beliefs': np.zeros((timesteps, 3), dtype=np.float64),
                'true_states': np.zeros(timesteps, dtype=np.int32),
                'policy_probs': np.zeros((timesteps, 3), dtype=np.float64),
                # Calculate these metrics directly instead of accessing agent attributes
                'belief_entropy': np.zeros(timesteps, dtype=np.float64),
                'policy_entropy': np.zeros(timesteps, dtype=np.float64),
                'accuracy': np.zeros(timesteps, dtype=np.float64)
            }
            
            # Initialize state and observation
            current_state = np.random.choice(3, p=self.env_matrices['D'])
            current_obs = self._generate_observation(current_state)
            
            # Run simulation
            logger.info(f"Starting simulation for {timesteps} timesteps")
            
            for t in range(timesteps):
                # Store current state and observation
                histories['true_states'][t] = current_state
                histories['observations'][t] = current_obs
                
                # State inference
                qs = self.agent.infer_states(obs=current_obs)
                histories['beliefs'][t] = qs
                
                # Policy inference
                q_pi = self.agent.infer_policies()
                histories['policy_probs'][t] = q_pi
                
                # Calculate metrics directly
                histories['belief_entropy'][t] = -np.sum(qs * np.log(qs + 1e-16))
                histories['policy_entropy'][t] = -np.sum(q_pi * np.log(q_pi + 1e-16))
                histories['accuracy'][t] = float(np.argmax(qs) == current_state)
                
                # Action selection
                action = self.agent.sample_action()
                histories['actions'][t] = action
                
                # Environment dynamics
                current_state = self._state_transition(current_state, action)
                current_obs = self._generate_observation(current_state)
                
                if t % 100 == 0:
                    logger.info(f"Timestep {t}: state={current_state}, obs={current_obs}, action={action}")
                    
            # Save all data
            self._save_histories(histories)
            self._save_summary(histories)
            self._save_metrics(histories)
            self._save_checkpoint()
            
            # Save metadata with updated metrics
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'matrix_shapes': {
                    'A': self.env_matrices['A'].shape,
                    'B': self.env_matrices['B'].shape,
                    'C': self.env_matrices['C'].shape,
                    'D': self.env_matrices['D'].shape
                },
                'simulation_info': {
                    'total_timesteps': timesteps,
                    'final_state': int(histories['true_states'][-1]),
                    'final_observation': int(histories['observations'][-1]),
                    'time_in_homeo': float(np.mean(histories['true_states'] == 1)),
                    'mean_accuracy': float(np.mean(histories['accuracy'])),
                    'mean_belief_entropy': float(np.mean(histories['belief_entropy'])),
                    'mean_policy_entropy': float(np.mean(histories['policy_entropy']))
                }
            }
            
            metadata_file = self.output_dir / "simulation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Update metrics saving
            self._save_metrics(histories)
            
            logger.info("Simulation complete")
            
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise
            
    def _generate_observation(self, state: int) -> int:
        """Generate observation from environment state"""
        obs_probs = self.env_matrices['A'][:, state]
        return np.random.choice(len(obs_probs), p=obs_probs)
        
    def _state_transition(self, state: int, action: int) -> int:
        """Execute state transition in environment"""
        trans_probs = self.env_matrices['B'][:, state, action]
        return np.random.choice(len(trans_probs), p=trans_probs)
        
    def _save_histories(self, histories: Dict):
        """Save simulation histories"""
        try:
            # Save raw histories
            histories_file = self.output_dir / "histories.json"
            with open(histories_file, 'w') as f:
                json.dump({k: v.tolist() for k, v in histories.items()}, f)
                
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'matrix_shapes': {
                    'environment': {
                        'A': self.env_matrices['A'].shape,
                        'B': self.env_matrices['B'].shape,
                        'D': self.env_matrices['D'].shape
                    },
                    'agent': {
                        'A': self.agent_matrices['A'].shape,
                        'B': self.agent_matrices['B'].shape,
                        'C': self.agent_matrices['C'].shape,
                        'D': self.agent_matrices['D'].shape
                    }
                }
            }
            
            metadata_file = self.output_dir / "simulation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved simulation data to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving histories: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Configure basic logging before argument parsing
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)7s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run Biofirm active inference simulation')
        parser.add_argument('--config', type=str, required=True,
                          help='Path to simulation config file')
        args = parser.parse_args()
        
        logger.info("Starting Biofirm simulation")
        logger.info(f"Using config file: {args.config}")
        
        # Create executor
        executor = BiofirmExecutor(config_file=Path(args.config))
        
        # Run simulation
        executor.run_simulation()
        
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
