import logging
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, Union
import numpy as np
import sys
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports instead of relative
from pymdp.agentmaker.config import DEFAULT_PATHS, DEFAULT_ENV_GNN, DEFAULT_AGENT_GNN

logger = logging.getLogger(__name__)

class BiofirmExecutionUtils:
    """Utilities for managing Biofirm experiment execution stages"""
    
    MATRIX_FORMATS = {
        'A': {'dtype': np.float64, 'shape': [(1, 3, 3), (3, 3)]},
        'B': {'dtype': np.float64, 'shape': [(1, 3, 3, 3), (3, 3, 3)]},
        'C': {'dtype': np.float64, 'shape': [(1, 3), (3,)]},
        'D': {'dtype': np.float64, 'shape': [(1, 3), (3,)]}
    }
    
    HISTORY_FORMATS = {
        'observations': {'dtype': np.int32, 'ndim': 1},
        'actions': {'dtype': np.int32, 'ndim': 1},
        'beliefs': {'dtype': np.float64, 'ndim': 2},
        'true_states': {'dtype': np.int32, 'ndim': 1},
        'policy_probs': {'dtype': np.float64, 'ndim': 2}
    }

    ENVIRONMENT_MATRICES = ['A', 'B', 'D']  # Environment only needs these
    AGENT_MATRICES = ['A', 'B', 'C', 'D']   # Agent needs all matrices

    @staticmethod
    def ensure_gnn_files_exist() -> tuple:
        """Create default GNN files if they don't exist and return paths"""
        try:
            # Get default paths
            env_gnn = DEFAULT_PATHS['env_gnn']
            agent_gnn = DEFAULT_PATHS['agent_gnn']
            
            # Create directories if they don't exist
            env_gnn.parent.mkdir(parents=True, exist_ok=True)
            agent_gnn.parent.mkdir(parents=True, exist_ok=True)
            
            # Create default GNN files if they don't exist
            if not env_gnn.exists():
                logger.info(f"Creating default environment GNN at {env_gnn}")
                with open(env_gnn, 'w') as f:
                    json.dump(DEFAULT_ENV_GNN, f, indent=4)
                    
            if not agent_gnn.exists():
                logger.info(f"Creating default agent GNN at {agent_gnn}")
                with open(agent_gnn, 'w') as f:
                    json.dump(DEFAULT_AGENT_GNN, f, indent=4)
                    
            return env_gnn, agent_gnn
            
        except Exception as e:
            logger.error(f"Error ensuring GNN files exist: {str(e)}")
            raise

    @staticmethod
    def run_rendering_stage(env_gnn: Path, agent_gnn: Path, render_dir: Path) -> Path:
        """Execute matrix rendering stage with improved error handling"""
        try:
            # Create render directory
            render_dir.mkdir(parents=True, exist_ok=True)
            
            # Create required subdirectories
            matrices_dir = render_dir / "matrices"
            config_dir = render_dir / "config"
            viz_dir = render_dir / "matrix_visualizations"
            logs_dir = render_dir / "logs"
            
            for d in [matrices_dir, config_dir, viz_dir, logs_dir]:
                d.mkdir(parents=True, exist_ok=True)
                
            # Run matrix rendering
            render_cmd = [
                "python3",
                str(Path(__file__).parent / "Biofirm_Render_GNN.py"),
                "--env_gnn", str(env_gnn),
                "--agent_gnn", str(agent_gnn),
                "--output_dir", str(render_dir)
            ]
            
            # Execute with output capture
            result = subprocess.run(render_cmd, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                logger.warning("Renderer warnings/errors:")
                for line in result.stdout.splitlines():
                    logger.warning(line)
                
            if result.stderr:
                logger.warning("Renderer stderr:")
                for line in result.stderr.splitlines():
                    logger.warning(line)
                
            if result.returncode != 0:
                raise RuntimeError(f"Matrix rendering failed with return code {result.returncode}")
                
            # Verify matrices were created
            matrices_env = matrices_dir / "environment"
            matrices_agent = matrices_dir / "agent"
            
            if not matrices_env.exists() or not matrices_agent.exists():
                raise RuntimeError("Matrix rendering failed - missing matrix directories")
                
            # Return matrices directory
            return matrices_dir
            
        except Exception as e:
            logger.error(f"Error in rendering stage: {str(e)}")
            raise

    @staticmethod
    def run_simulation_stage(matrices_dir: Path, sim_dir: Path) -> Path:
        """Execute simulation stage with improved error handling"""
        try:
            # Setup simulation directory
            sim_dir.mkdir(parents=True, exist_ok=True)
            
            # Create required subdirectories
            data_dir = sim_dir / "data"
            results_dir = sim_dir / "results"
            logs_dir = sim_dir / "logs"
            
            for d in [data_dir, results_dir, logs_dir]:
                d.mkdir(parents=True, exist_ok=True)
                
            # Create simulation config using EXPERIMENT_CONFIG from config module
            from pymdp.agentmaker.config import EXPERIMENT_CONFIG
            
            config = {
                'matrices_dir': str(matrices_dir),
                'output_dir': str(sim_dir),
                'timesteps': EXPERIMENT_CONFIG['timesteps'],  # Use timesteps from EXPERIMENT_CONFIG
                'parameters': EXPERIMENT_CONFIG['parameters']  # Use parameters from EXPERIMENT_CONFIG
            }
            
            # Save config
            config_file = sim_dir / "config" / "simulation_config.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            # Run simulation
            logger.info(f"Running simulation with config: {config_file}")
            sim_cmd = [
                "python3",
                str(Path(__file__).parent / "Biofirm_Execute_GNN.py"),
                "--config", str(config_file)
            ]
            
            # Execute with output capture
            result = subprocess.run(sim_cmd, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                logger.info("Simulation output:")
                for line in result.stdout.splitlines():
                    logger.info(line)
                    
            if result.stderr:
                logger.warning("Simulation warnings/errors:")
                for line in result.stderr.splitlines():
                    logger.warning(line)
                    
            if result.returncode != 0:
                raise RuntimeError(f"Simulation failed with return code {result.returncode}")
                
            # Verify required files exist
            required_files = {
                'data': ['histories.json', 'simulation_metadata.json'],
                'results': ['simulation_results.json'],
                'config': ['simulation_config.json']
            }
            
            for dir_name, files in required_files.items():
                dir_path = sim_dir / dir_name
                for file_name in files:
                    file_path = dir_path / file_name
                    if not file_path.exists():
                        raise RuntimeError(f"Missing required file: {file_path}")
                        
            return sim_dir
            
        except Exception as e:
            logger.error(f"Error in simulation stage: {str(e)}")
            raise

    @staticmethod
    def run_analysis_stage(simulation_dir: Path, analysis_dir: Path) -> Path:
        """Execute analysis stage with improved error handling"""
        try:
            # Create analysis directory
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # First verify simulation data exists
            data_dir = simulation_dir / "data"
            histories_file = data_dir / "histories.json"
            
            if not histories_file.exists():
                logger.error(f"Cannot find histories file at {histories_file}")
                logger.error("Directory contents:")
                BiofirmExecutionUtils._print_directory_contents(simulation_dir)
                raise RuntimeError(f"Missing required simulation data: {histories_file}")
                
            # Verify other required simulation files
            required_files = {
                'data': ['simulation_metadata.json'],
                'results': ['simulation_results.json', 'simulation_summary.json'],
                'metrics': ['performance_metrics.json'],
                'checkpoints': ['final_state.json']
            }
            
            # Create analysis subdirectories
            analysis_subdirs = [
                'core_analysis',
                'policy_analysis',
                'inference_cycle',
                'convergence',
                'belief_action',
                'homeostasis'
            ]
            
            for subdir in analysis_subdirs:
                (analysis_dir / subdir).mkdir(parents=True, exist_ok=True)
                
            # Run visualization with verified paths
            visualize_cmd = [
                "python3",
                str(Path(__file__).parent / "Biofirm_Visualize.py"),
                "--simulation_dir", str(simulation_dir),
                "--output_dir", str(analysis_dir)
            ]
            
            logger.info(f"Running visualization command: {' '.join(visualize_cmd)}")
            result = subprocess.run(visualize_cmd, capture_output=True, text=True)
            
            # Log visualization output
            if result.stdout:
                logger.info("Visualization output:")
                for line in result.stdout.splitlines():
                    logger.info(line)
                    
            if result.stderr:
                logger.warning("Visualization warnings/errors:")
                for line in result.stderr.splitlines():
                    logger.warning(line)
                    
            if result.returncode != 0:
                raise RuntimeError(f"Visualization failed with return code {result.returncode}")
                
            # Verify visualization outputs
            required_plots = {
                'core_analysis': [
                    'belief_dynamics.png',
                    'action_frequencies.png',
                    'state_transitions.png',
                    'policy_probabilities.png'
                ],
                'policy_analysis': [
                    'policy_entropy.png',
                    'action_distribution.png'
                ],
                'homeostasis': [
                    'state_distribution.png',
                    'homeostasis_trajectory.png'
                ]
            }
            
            missing_plots = []
            for dir_name, plots in required_plots.items():
                plot_dir = analysis_dir / dir_name
                for plot in plots:
                    plot_path = plot_dir / plot
                    if not plot_path.exists():
                        missing_plots.append(str(plot_path))
                        
            if missing_plots:
                logger.error("Missing visualization outputs:")
                for p in missing_plots:
                    logger.error(f"- {p}")
                raise RuntimeError("Visualization outputs missing")
                
            # Create analysis summary
            analysis_summary = {
                'timestamp': datetime.now().isoformat(),
                'simulation_dir': str(simulation_dir),
                'analysis_dir': str(analysis_dir),
                'plots_generated': {
                    dir_name: plots 
                    for dir_name, plots in required_plots.items()
                }
            }
            
            # Load simulation metrics to include in summary
            metrics_file = simulation_dir / "metrics" / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    analysis_summary['metrics'] = json.load(f)
                    
            # Save analysis summary
            summary_file = analysis_dir / "analysis_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(analysis_summary, f, indent=2)
                
            logger.info(f"Analysis complete. Results saved to {analysis_dir}")
            return analysis_dir
            
        except Exception as e:
            logger.error(f"Error in analysis stage: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise

    @staticmethod
    def _print_directory_contents(directory: Path, indent: int = 0):
        """Print directory contents recursively"""
        try:
            for path in sorted(directory.iterdir()):
                logger.error("  " * indent + f"- {path.name}")
                if path.is_dir():
                    BiofirmExecutionUtils._print_directory_contents(path, indent + 1)
        except Exception as e:
            logger.error(f"Error printing directory contents: {str(e)}")

    @staticmethod
    def _validate_matrices_output(matrices_dir: Path) -> None:
        """Validate matrix rendering output with proper requirements"""
        if not matrices_dir.exists():
            raise RuntimeError("Matrix rendering failed - no matrices directory")
            
        # Check environment matrices
        env_dir = matrices_dir / "environment"
        if not env_dir.exists():
            raise RuntimeError(f"Missing environment directory: {env_dir}")
            
        # Check agent matrices  
        agent_dir = matrices_dir / "agent"
        if not agent_dir.exists():
            raise RuntimeError(f"Missing agent directory: {agent_dir}")
            
        # Validate environment matrices (only A, B, D)
        for matrix_type in BiofirmExecutionUtils.ENVIRONMENT_MATRICES:
            file_path = env_dir / f"{matrix_type}_matrices.npy"
            if not file_path.exists():
                raise RuntimeError(f"Missing environment matrix file: {file_path}")
            if not BiofirmExecutionUtils.validate_matrix_file(file_path, matrix_type):
                raise RuntimeError(f"Invalid environment matrix format: {matrix_type}")
                
        # Validate agent matrices (A, B, C, D)
        for matrix_type in BiofirmExecutionUtils.AGENT_MATRICES:
            file_path = agent_dir / f"{matrix_type}_matrices.npy"
            if not file_path.exists():
                raise RuntimeError(f"Missing agent matrix file: {file_path}")
            if not BiofirmExecutionUtils.validate_matrix_file(file_path, matrix_type):
                raise RuntimeError(f"Invalid agent matrix format: {matrix_type}")
                
        # Check consolidated matrices.json exists
        if not (matrices_dir / "matrices.json").exists():
            raise RuntimeError("Missing matrices.json")

    @staticmethod
    def _setup_simulation_directory(matrices_dir: Path, sim_dir: Path) -> None:
        """Setup simulation directory structure"""
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config files from render stage
        config_src = matrices_dir.parent / "config"
        config_dst = sim_dir / "config"
        if config_src.exists():
            shutil.copytree(config_src, config_dst, dirs_exist_ok=True)

    @staticmethod
    def _create_simulation_config(matrices_dir: Path, sim_dir: Path, timesteps: int) -> Dict:
        """Create simulation configuration"""
        config = {
            'timesteps': timesteps,
            'matrices_dir': str(matrices_dir),
            'output_dir': str(sim_dir),
            'parameters': {
                'ecological_noise': 0.1,
                'controllability': 0.8,
                'policy_precision': 16.0,
                'use_states_info_gain': True
            }
        }
        
        config_file = sim_dir / "simulation_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        return config

    @staticmethod
    def setup_python_path():
        """Setup Python path to include project root"""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up two levels to project root
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
    @staticmethod
    def _execute_simulation(config_file: Path) -> None:
        """Execute simulation with proper error handling"""
        # Setup Python path
        BiofirmExecutionUtils.setup_python_path()
        
        execute_cmd = [
            "python3",
            "-c",
            f"""
import sys
sys.path.insert(0, '{Path(__file__).parent.parent.parent}')
from pymdp.agentmaker.Biofirm_Execute_GNN import main
sys.argv = ['Biofirm_Execute_GNN.py', '--config', '{config_file}']
main()
        """
        ]
        
        logger.info(f"Running simulation command with proper Python path")
        
        try:
            result = subprocess.run(
                execute_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                logger.debug("Simulation output:")
                for line in result.stdout.splitlines():
                    logger.debug(line)
                    
            if result.stderr:
                logger.warning("Simulation warnings:")
                for line in result.stderr.splitlines():
                    logger.warning(line)
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Simulation failed with return code {e.returncode}")
            if e.stdout:
                logger.error("Simulation output:")
                logger.error(e.stdout)
            if e.stderr:
                logger.error("Simulation errors:")
                logger.error(e.stderr)
            raise RuntimeError(f"Simulation failed with return code {e.returncode}")

    @staticmethod
    def _validate_simulation_output(sim_dir: Path, timesteps: int) -> None:
        """Validate simulation output"""
        history_file = sim_dir / "histories.json"
        if not history_file.exists():
            raise RuntimeError("Missing history file")
            
        with open(history_file) as f:
            history = json.load(f)
            
        if not BiofirmExecutionUtils.validate_history_data(history, timesteps):
            raise RuntimeError("Invalid history data format")

    @staticmethod
    def validate_matrix_file(file_path: Path, matrix_type: str) -> bool:
        """Validate matrix file format with detailed logging"""
        try:
            data = np.load(file_path)
            format_spec = BiofirmExecutionUtils.MATRIX_FORMATS[matrix_type]
            
            # Log matrix info
            logger.debug(f"\nValidating {matrix_type} matrix from {file_path}")
            logger.debug(f"Loaded shape: {data.shape}")
            logger.debug(f"Loaded dtype: {data.dtype}")
            logger.debug(f"Expected shapes: {format_spec['shape']}")
            logger.debug(f"Expected dtype: {format_spec['dtype']}")
            
            if data.dtype != format_spec['dtype']:
                logger.error(f"Invalid dtype for {matrix_type}: {data.dtype}")
                return False
                
            # Check if shape matches any valid shape
            valid_shapes = format_spec['shape']
            if not any(data.shape == shape for shape in valid_shapes):
                logger.error(f"Invalid shape for {matrix_type}: {data.shape}")
                logger.error(f"Expected one of: {valid_shapes}")
                return False
                
            # Validate probability matrices
            if matrix_type in ['A', 'B', 'D']:
                if not np.allclose(np.sum(data, axis=0), 1.0):
                    logger.warning(f"{matrix_type} matrix not normalized")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating {matrix_type} matrix: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return False

    @staticmethod
    def validate_history_data(data: Dict[str, Any], timesteps: int) -> bool:
        """Validate history data format"""
        try:
            for key, format_spec in BiofirmExecutionUtils.HISTORY_FORMATS.items():
                if key not in data:
                    logger.error(f"Missing history key: {key}")
                    return False
                    
                arr = np.array(data[key])
                if arr.dtype != format_spec['dtype']:
                    logger.error(f"Invalid dtype for {key}: {arr.dtype}")
                    return False
                    
                if arr.ndim != format_spec['ndim']:
                    logger.error(f"Invalid dimensions for {key}: {arr.ndim}")
                    return False
                    
                if len(arr) != timesteps:
                    logger.error(f"Invalid length for {key}: {len(arr)}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating history data: {str(e)}")
            return False

    @staticmethod
    def save_matrices(matrices: Dict[str, np.ndarray], output_dir: Path) -> None:
        """Save matrices with consistent format"""
        try:
            for matrix_type, data in matrices.items():
                file_path = output_dir / f"{matrix_type}_matrices.npy"
                np.save(file_path, data.astype(BiofirmExecutionUtils.MATRIX_FORMATS[matrix_type]['dtype']))
                
        except Exception as e:
            logger.error(f"Error saving matrices: {str(e)}")
            raise

    @staticmethod
    def save_history(history: Dict[str, Any], output_file: Path) -> None:
        """Save history data with consistent format"""
        try:
            formatted_history = {}
            for key, data in history.items():
                if key in BiofirmExecutionUtils.HISTORY_FORMATS:
                    formatted_history[key] = np.array(data).astype(
                        BiofirmExecutionUtils.HISTORY_FORMATS[key]['dtype']
                    ).tolist()
                else:
                    formatted_history[key] = data
                    
            with open(output_file, 'w') as f:
                json.dump(formatted_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            raise

    @staticmethod
    def preprocess_matrices(matrices: Dict[str, np.ndarray], model_type: str) -> Dict[str, np.ndarray]:
        """Standardize matrix formats by unwrapping if needed"""
        processed = {}
        for matrix_type, data in matrices.items():
            # Handle list-wrapped matrices (shape like (1, ...))
            if data.ndim > len(BiofirmExecutionUtils.MATRIX_FORMATS[matrix_type]['shape'][1]):
                data = data[0]  # Unwrap first dimension
                
            processed[matrix_type] = data
            
            # Validate final shape
            expected_shape = BiofirmExecutionUtils.MATRIX_FORMATS[matrix_type]['shape'][1]
            if data.shape != expected_shape:
                raise ValueError(
                    f"Invalid {model_type} matrix shape for {matrix_type}: "
                    f"got {data.shape}, expected {expected_shape}"
                )
                
        return processed

    @staticmethod
    def validate_directory_structure(base_dir: Path, stage: str) -> bool:
        """Validate directory structure for a given stage"""
        try:
            if stage == 'render':
                required_dirs = [
                    "matrices",
                    "matrices/environment",
                    "matrices/agent",
                    "config"
                ]
            elif stage == 'simulation':
                required_dirs = [
                    "logs",
                    "results",
                    "config"
                ]
            elif stage == 'analysis':
                required_dirs = [
                    "core_analysis",
                    "policy_analysis",
                    "inference_cycle",
                    "convergence",
                    "belief_action"
                ]
            else:
                logger.error(f"Unknown stage: {stage}")
                return False
                
            for dir_path in required_dirs:
                full_path = base_dir / dir_path
                if not full_path.exists():
                    logger.error(f"Missing required directory: {full_path}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating directory structure: {str(e)}")
            return False

    @staticmethod
    def validate_simulation_transition(simulation_dir: Path) -> bool:
        """Validate simulation data for analysis transition"""
        try:
            # Check for histories.json in data directory
            histories_file = simulation_dir / "data" / "histories.json"
            if not histories_file.exists():
                logger.error(f"Cannot find histories file: {histories_file}")
                logger.error("Directory contents:")
                BiofirmExecutionUtils._print_directory_contents(simulation_dir)
                return False
                
            # Verify histories.json is valid JSON and has required fields
            try:
                with open(histories_file) as f:
                    histories = json.load(f)
                    required_fields = [
                        'observations', 'actions', 'beliefs', 
                        'true_states', 'policy_probs'
                    ]
                    for field in required_fields:
                        if field not in histories:
                            logger.error(f"Missing required field in histories.json: {field}")
                            return False
            except Exception as e:
                logger.error(f"Error validating histories.json: {str(e)}")
                return False
                
            # Check other required files
            required_files = {
                'data': ['simulation_metadata.json'],
                'results': ['simulation_results.json', 'simulation_summary.json'],
                'metrics': ['performance_metrics.json'],
                'checkpoints': ['final_state.json']
            }
            
            missing_files = []
            for dir_name, files in required_files.items():
                dir_path = simulation_dir / dir_name
                for file_name in files:
                    file_path = dir_path / file_name
                    if not file_path.exists():
                        missing_files.append(str(file_path))
                        
            if missing_files:
                logger.error("Missing required simulation files:")
                for f in missing_files:
                    logger.error(f"- {f}")
                return False
                
            logger.info("Validated simulation -> analysis transition")
            return True
            
        except Exception as e:
            logger.error(f"Error validating simulation transition: {str(e)}")
            return False