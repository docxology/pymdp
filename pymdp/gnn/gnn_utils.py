import numpy as np
import json
import os
import logging
from datetime import datetime
from pymdp import utils
from pymdp.maths import softmax, spm_dot, spm_norm, entropy
import glob
from typing import List

logger = logging.getLogger(__name__)

class GNNUtils:
    """Utility functions for GNN model handling, validation and inference"""
    
    @staticmethod
    def load_model(filepath):
        """Load GNN model from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
            
    @staticmethod
    def save_model(model, filepath):
        """Save GNN model to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(model, f, indent=2)

    @staticmethod
    def create_sandbox_dir(base_path):
        """Create timestamped sandbox directory for experiment outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_dir = os.path.join(base_path, f"run_{timestamp}")
        
        # Create directory structure
        for subdir in ['matrices', 'viz', 'logs']:
            os.makedirs(os.path.join(sandbox_dir, subdir), exist_ok=True)
            
        return sandbox_dir

    @staticmethod
    def setup_logging(sandbox_dir, logger_name='gnn_agent'):
        """Configure file and console logging"""
        logger = logging.getLogger(logger_name)
        log_file = os.path.join(sandbox_dir, "logs", "agent.log")
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        
        return logger

    @staticmethod
    def get_likelihood(obs, A_matrices, gnn_model):
        """Get likelihood arrays for observations"""
        likelihood = []
        
        try:
            for m, o in enumerate(obs):
                # Get likelihood array for this modality
                A_m = A_matrices[m]
                l_m = A_m[o]
                l_m = spm_norm(l_m)
                likelihood.append(l_m)
                
            return likelihood
                
        except Exception as e:
            logger.error(f"Error computing likelihood: {str(e)}")
            raise

    @staticmethod
    def infer_states(obs, qs, A_matrices, gnn_model):
        """Infer hidden states using message passing"""
        try:
            # Get likelihood for each modality
            likelihood = GNNUtils.get_likelihood(obs, A_matrices, gnn_model)
            
            # Initialize new beliefs
            qs_new = [np.zeros_like(q) for q in qs]
            
            # For each state factor
            for f in range(len(qs)):
                # Initialize message
                message = np.ones_like(qs[f])
                
                # Get messages from each modality that observes this factor
                for m, modality in enumerate(gnn_model['observations']['modalities']):
                    if f in modality['factors_observed']:
                        # Get likelihood and marginalize if needed
                        l_m = likelihood[m]
                        other_factors = [i for i in modality['factors_observed'] if i != f]
                        
                        if other_factors:
                            # Marginalize over other factors
                            factor_dims = [gnn_model['stateSpace']['factors'][i]['num_states'] 
                                         for i in modality['factors_observed']]
                            l_m = l_m.reshape(-1, *factor_dims)
                            
                            msg = l_m.copy()
                            for of in other_factors:
                                of_idx = modality['factors_observed'].index(of)
                                sum_axes = of_idx + 1
                                msg = np.tensordot(msg, qs[of], axes=([sum_axes], [0]))
                        else:
                            msg = l_m
                            
                        message *= msg[obs[m]]
                
                # Combine with prior and normalize
                qs_new[f] = softmax(np.log(message) + np.log(qs[f]))
                
            return qs_new
                
        except Exception as e:
            logger.error(f"Error during state inference: {str(e)}")
            raise

    @staticmethod
    def get_expected_states(qs, policies, B_matrices, parameters):
        """Get predicted states for each policy"""
        try:
            qs_pi = []
            num_policies = len(policies[0]) if policies[0] is not None else 0
            
            # For each policy
            for p in range(num_policies):
                # Initialize predicted state beliefs
                qs_p = [q.copy() for q in qs]
                
                # For each timestep
                for t in range(parameters['policy_len']):
                    # Get actions for this policy/timestep
                    actions = []
                    for f, pol in enumerate(policies):
                        if pol is not None:
                            actions.append(pol[p, t])
                        else:
                            actions.append(None)
                    
                    # Predict next states
                    for f, action in enumerate(actions):
                        if action is not None:
                            B = B_matrices[f]
                            qs_p[f] = spm_dot(B[:,:,int(action)], qs_p[f])
                
                qs_pi.append(qs_p)
            
            return qs_pi
            
        except Exception as e:
            logger.error(f"Error calculating expected states: {str(e)}")
            raise

    @staticmethod
    def save_experiment_summary(sandbox_dir, belief_history, parameters):
        """Save experiment summary to markdown file"""
        summary_path = os.path.join(sandbox_dir, "experiment_summary.md")
        with open(summary_path, 'w') as f:
            f.write("# GNN Agent Experiment Summary\n\n")
            f.write(f"- Total timesteps: {len(belief_history)}\n")
            
            if belief_history:
                states = np.array([np.argmax(beliefs[0]) for beliefs in belief_history])
                state_counts = np.bincount(states, minlength=3)
                state_percentages = state_counts / len(states) * 100
                
                f.write("\n## State Distribution\n")
                f.write(f"- LOW state: {state_percentages[0]:.1f}%\n")
                f.write(f"- HOMEO state: {state_percentages[1]:.1f}%\n")
                f.write(f"- HIGH state: {state_percentages[2]:.1f}%\n")
            
            f.write(f"\n## Agent Parameters\n")
            for param, value in parameters.items():
                f.write(f"- {param}: {value}\n")

    @staticmethod
    def load_rendered_matrices(matrices_dir):
        """Load pre-rendered matrices and GNN model from directory"""
        try:
            # Load matrices
            matrices = {}
            
            # Load A matrices
            A_dir = os.path.join(matrices_dir, "A_matrices")
            matrices['A'] = [np.load(f) for f in sorted(glob.glob(os.path.join(A_dir, "*.npy")))]
            
            # Load B matrices
            B_dir = os.path.join(matrices_dir, "B_matrices")
            matrices['B'] = [np.load(f) for f in sorted(glob.glob(os.path.join(B_dir, "*.npy")))]
            
            # Load C matrices if they exist
            C_dir = os.path.join(matrices_dir, "C_matrices")
            if os.path.exists(C_dir):
                matrices['C'] = [np.load(f) for f in sorted(glob.glob(os.path.join(C_dir, "*.npy")))]
            
            # Load D matrices
            D_dir = os.path.join(matrices_dir, "D_matrices")
            matrices['D'] = [np.load(f) for f in sorted(glob.glob(os.path.join(D_dir, "*.npy")))]
            
            # Load control indices
            control_file = os.path.join(matrices_dir, "control_indices.json")
            if os.path.exists(control_file):
                with open(control_file, 'r') as f:
                    matrices['control_fac_idx'] = json.load(f)
            
            # Load GNN model definition from config
            config_dir = os.path.join(os.path.dirname(matrices_dir), "config")
            model_file = os.path.join(config_dir, "model.json")
            if os.path.exists(model_file):
                with open(model_file, 'r') as f:
                    matrices['model'] = json.load(f)
                    
            return matrices
                
        except Exception as e:
            logger.error(f"Error loading rendered matrices: {str(e)}")
            raise

    @staticmethod
    def save_rendered_matrices(matrices, save_dir):
        """Save rendered matrices to directory in both .npy and readable formats"""
        try:
            # Create matrix subdirectories
            for matrix_type in ['A', 'B', 'C', 'D']:
                os.makedirs(os.path.join(save_dir, f"{matrix_type}_matrices"), exist_ok=True)
            
            # Save A matrices (Observation model)
            for i, A in enumerate(matrices['A']):
                base_path = os.path.join(save_dir, f"A_matrices/A_{i}")
                # Save .npy
                np.save(f"{base_path}.npy", A)
                # Save readable format
                GNNUtils._save_matrix_readable(
                    A, 
                    f"{base_path}.txt",
                    title="Observation Model Matrix",
                    row_labels=[f"Obs_{j}" for j in range(A.shape[0])],
                    col_labels=[f"State_{j}" for j in range(A.shape[1])],
                    description="Maps hidden states to observations. Values represent P(o|s)"
                )
                
            # Save B matrices (Transition model)
            for i, B in enumerate(matrices['B']):
                base_path = os.path.join(save_dir, f"B_matrices/B_{i}")
                # Save .npy
                np.save(f"{base_path}.npy", B)
                # Save readable format - handle 3D case
                if len(B.shape) == 3:
                    for a in range(B.shape[2]):
                        GNNUtils._save_matrix_readable(
                            B[:,:,a],
                            f"{base_path}_action_{a}.txt",
                            title=f"Transition Model Matrix (Action {a})",
                            row_labels=[f"Next_{j}" for j in range(B.shape[0])],
                            col_labels=[f"Current_{j}" for j in range(B.shape[1])],
                            description="Maps current states to next states. Values represent P(s'|s,a)"
                        )
                else:
                    GNNUtils._save_matrix_readable(
                        B,
                        f"{base_path}.txt",
                        title="Transition Model Matrix",
                        row_labels=[f"Next_{j}" for j in range(B.shape[0])],
                        col_labels=[f"Current_{j}" for j in range(B.shape[1])],
                        description="Maps current states to next states. Values represent P(s'|s)"
                    )
                
            # Save C matrices (Preferences) if they exist
            if 'C' in matrices:
                for i, C in enumerate(matrices['C']):
                    base_path = os.path.join(save_dir, f"C_matrices/C_{i}")
                    # Save .npy
                    np.save(f"{base_path}.npy", C)
                    # Save readable format
                    GNNUtils._save_matrix_readable(
                        C.reshape(-1, 1),
                        f"{base_path}.txt",
                        title="Preference Matrix",
                        row_labels=[f"Obs_{j}" for j in range(len(C))],
                        col_labels=["Preference"],
                        description="Preferences over observations. Positive values are preferred, negative are aversive"
                    )
                    
            # Save D matrices (Initial state priors)
            for i, D in enumerate(matrices['D']):
                base_path = os.path.join(save_dir, f"D_matrices/D_{i}")
                # Save .npy
                np.save(f"{base_path}.npy", D)
                # Save readable format
                GNNUtils._save_matrix_readable(
                    D.reshape(-1, 1),
                    f"{base_path}.txt",
                    title="Initial State Prior Matrix",
                    row_labels=[f"State_{j}" for j in range(len(D))],
                    col_labels=["Prior"],
                    description="Initial beliefs about hidden states. Values must sum to 1.0"
                )
                
            # Save control indices
            if 'control_fac_idx' in matrices:
                with open(os.path.join(save_dir, "control_indices.json"), 'w') as f:
                    json.dump(matrices['control_fac_idx'], f)
                    
        except Exception as e:
            logger.error(f"Error saving rendered matrices: {str(e)}")
            raise

    @staticmethod
    def _save_matrix_readable(matrix: np.ndarray, 
                            filepath: str,
                            title: str = "",
                            row_labels: List[str] = None,
                            col_labels: List[str] = None,
                            description: str = ""):
        """Save matrix in human-readable format with labels and description"""
        try:
            with open(filepath, 'w') as f:
                # Write header
                f.write(f"{title}\n{'='*len(title)}\n\n")
                
                # Write description
                if description:
                    f.write(f"Description:\n{description}\n\n")
                
                # Write shape
                f.write(f"Shape: {matrix.shape}\n\n")
                
                # Write column labels if provided
                if col_labels:
                    f.write("     " + "  ".join(f"{label:>8}" for label in col_labels) + "\n")
                    f.write("     " + "  ".join("-"*8 for _ in col_labels) + "\n")
                
                # Write matrix with row labels
                for i in range(matrix.shape[0]):
                    row = matrix[i]
                    row_str = "  ".join(f"{val:8.3f}" for val in row)
                    if row_labels:
                        f.write(f"{row_labels[i]:>4} {row_str}\n")
                    else:
                        f.write(f"{i:>4} {row_str}\n")
                        
        except Exception as e:
            logger.error(f"Error saving readable matrix to {filepath}: {str(e)}")
            raise

    @staticmethod
    def validate_rendered_matrices(matrices):
        """Validate loaded matrices for simulation"""
        try:
            # Check required matrices exist
            required = ['A', 'B', 'D']
            for req in required:
                if req not in matrices:
                    raise ValueError(f"Missing required matrix type: {req}")
                    
            # Validate shapes and normalization
            for A in matrices['A']:
                if not utils.is_normalized(A):
                    raise ValueError("A matrix not normalized")
                    
            for B in matrices['B']:
                if not utils.is_normalized(B):
                    raise ValueError("B matrix not normalized")
                    
            for D in matrices['D']:
                if not utils.is_normalized(D):
                    raise ValueError("D matrix not normalized")
                    
            return True
            
        except Exception as e:
            logger.error(f"Matrix validation failed: {str(e)}")
            raise