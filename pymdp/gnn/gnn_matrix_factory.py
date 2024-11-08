import numpy as np
import json
import os
from pymdp import utils
from pymdp.maths import softmax
import markdown
import yaml
from datetime import datetime
import logging
from typing import List
from pathlib import Path
import jsonschema

logger = logging.getLogger(__name__)

class GNNMatrixFactory:
    """
    Factory class for creating PyMDP-compatible matrices from GNN models.
    
    This class handles the conversion from GNN model definitions to the matrix 
    representations needed for active inference. The key matrices are:
    
    A (Observation Model):
        Maps hidden states to observations. Shape: (num_obs x num_states)
        - For single factors: P(o|s) 
        - For multiple factors: P(o|s1,s2)
        
    B (Transition Model):
        Defines state transitions. Shape: (num_states x num_states x num_actions)
        - For controlled factors: P(s'|s,a)
        - For fixed factors: P(s'|s)
        
    C (Preferences):
        Defines preferences over observations. Shape: (num_obs,)
        - Higher values indicate preferred observations
        - Lower values indicate aversive observations
        - C also can be see as defining the Pragmatic homeostatic constraint
        
    D (Initial State Priors):
        Initial beliefs about hidden states. Shape: (num_states,)
        - Must sum to 1.0 for each factor
    """
    
    def __init__(self, gnn_file: str, model_type: str = 'agent'):
        """Initialize factory with GNN model file
        
        Parameters
        ----------
        gnn_file : str
            Path to GNN model file
        model_type : str
            Type of model ('agent' or 'environment')
        """
        self.gnn_file = gnn_file
        self.model_type = model_type
        
        # Load schema
        schema_path = Path(__file__).parent / "gnn_schema.json"
        with open(schema_path) as f:
            self.schema = json.load(f)
            
        # Set required sections based on model type
        self.required_sections = self._get_required_sections(model_type)
        
        self.model = None
        self.load_model()
            
    def _get_required_sections(self, model_type: str) -> List[str]:
        """Get required sections based on model type"""
        # Base requirements for all models
        base_sections = [
            'stateSpace',
            'observations',
            'transitionModel'
        ]
        
        if model_type == 'agent':
            # Agent models need additional sections
            return base_sections + [
                'policies',
                'preferences'
            ]
        else:
            # Environment models only need base sections
            return base_sections
            
    def load_model(self):
        """Load and validate GNN model"""
        logger.debug(f"Loading GNN model from {self.gnn_file}")
        try:
            with open(self.gnn_file, 'r') as f:
                self.model = json.load(f)
                
            # Validate against schema
            jsonschema.validate(instance=self.model, schema=self.schema)
            
            # Add default sections based on model type
            if self.model_type == 'environment':
                self._add_environment_defaults()
            elif self.model_type == 'agent':
                self._add_agent_defaults()
                
            self.validate_gnn()
            
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"GNN model validation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading GNN model: {str(e)}")
            raise
            
    def _add_environment_defaults(self):
        """Add default sections to environment model if missing"""
        # Add or update state space section
        if 'stateSpace' not in self.model:
            self.model['stateSpace'] = {
                'factors': ['EcologicalState'],
                'sizes': [3],  # LOW, HOMEO, HIGH
                'labels': [['LOW', 'HOMEO', 'HIGH']]
            }
            logger.info("Added default state space section to environment model")
        else:
            state_space = self.model['stateSpace']
            if 'factors' not in state_space:
                state_space['factors'] = ['EcologicalState']
            if 'sizes' not in state_space:
                state_space['sizes'] = [3]  # LOW, HOMEO, HIGH
            if 'labels' not in state_space:
                state_space['labels'] = [['LOW', 'HOMEO', 'HIGH']]
                
        # Add or update observations section
        if 'observations' not in self.model:
            self.model['observations'] = {
                'modalities': ['StateObservation'],
                'sizes': [3],  # LOW, HOMEO, HIGH
                'labels': [['LOW', 'HOMEO', 'HIGH']]
            }
            logger.info("Added default observations section to environment model")
        else:
            obs = self.model['observations']
            if 'modalities' not in obs:
                obs['modalities'] = ['StateObservation']
            if 'sizes' not in obs:
                obs['sizes'] = [3]  # LOW, HOMEO, HIGH
            if 'labels' not in obs:
                obs['labels'] = [['LOW', 'HOMEO', 'HIGH']]

        # Add or update policies section
        if 'policies' not in self.model:
            self.model['policies'] = {
                'controlFactors': [0],  # First factor is controllable
                'numControls': [3],     # 3 possible actions
                'policyLen': 1,         # Single-step policies
                'control_fac_idx': [0], # Control factor indices
                'labels': [['DECREASE', 'MAINTAIN', 'INCREASE']]
            }
            logger.info("Added default policies section to environment model")
        else:
            policies = self.model['policies']
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

        # Add or update preferences section
        if 'preferences' not in self.model:
            self.model['preferences'] = {
                'modalities': ['StateObservation'],
                'values': [[0.0, 1.0, 0.0]],  # Prefer HOMEO state
                'labels': [['LOW', 'HOMEO', 'HIGH']]
            }
            logger.info("Added default preferences to environment model")
        else:
            prefs = self.model['preferences']
            if 'modalities' not in prefs:
                prefs['modalities'] = ['StateObservation']
            if 'values' not in prefs:
                prefs['values'] = [[0.0, 1.0, 0.0]]
            if 'labels' not in prefs:
                prefs['labels'] = [['LOW', 'HOMEO', 'HIGH']]

        # Add or update transition model section
        if 'transitionModel' not in self.model:
            self.model['transitionModel'] = {
                'EcologicalState': {
                    'controlled': True,
                    'control_factor_idx': 0,
                    'num_controls': 3,
                    'state_labels': ['LOW', 'HOMEO', 'HIGH'],
                    'action_labels': ['DECREASE', 'MAINTAIN', 'INCREASE']
                }
            }
            logger.info("Added default transition model to environment model")
            
    def _add_agent_defaults(self):
        """Add default sections to agent model if missing"""
        # Add or update state space section
        if 'stateSpace' not in self.model:
            self.model['stateSpace'] = {
                'factors': ['EcologicalState'],
                'sizes': [3],  # LOW, HOMEO, HIGH
                'labels': [['LOW', 'HOMEO', 'HIGH']]
            }
            logger.info("Added default state space section to agent model")
        else:
            state_space = self.model['stateSpace']
            if 'factors' not in state_space:
                state_space['factors'] = ['EcologicalState']
            if 'sizes' not in state_space:
                state_space['sizes'] = [3]
            if 'labels' not in state_space:
                state_space['labels'] = [['LOW', 'HOMEO', 'HIGH']]
                
        # Add or update observations section
        if 'observations' not in self.model:
            self.model['observations'] = {
                'modalities': ['StateObservation'],
                'sizes': [3],  # LOW, HOMEO, HIGH
                'labels': [['LOW', 'HOMEO', 'HIGH']]
            }
            logger.info("Added default observations section to agent model")
        else:
            obs = self.model['observations']
            if 'modalities' not in obs:
                obs['modalities'] = ['StateObservation']
            if 'sizes' not in obs:
                obs['sizes'] = [3]
            if 'labels' not in obs:
                obs['labels'] = [['LOW', 'HOMEO', 'HIGH']]

        # Add or update policies section
        if 'policies' not in self.model:
            self.model['policies'] = {
                'controlFactors': [0],  # First factor is controllable
                'numControls': [3],     # 3 possible actions
                'policyLen': 1,         # Single-step policies
                'control_fac_idx': [0], # Control factor indices
                'labels': [['DECREASE', 'MAINTAIN', 'INCREASE']]
            }
            logger.info("Added default policies section to agent model")
        else:
            policies = self.model['policies']
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
            
        # Add or update preferences section with strong preference for HOMEO
        if 'preferences' not in self.model:
            self.model['preferences'] = {
                'modalities': ['StateObservation'],
                'values': [[0.0, 4.0, 0.0]],  # Strong preference for HOMEO state
                'labels': [['LOW', 'HOMEO', 'HIGH']]
            }
            logger.info("Added default preferences to agent model")
        else:
            prefs = self.model['preferences']
            if 'modalities' not in prefs:
                prefs['modalities'] = ['StateObservation']
            if 'values' not in prefs:
                prefs['values'] = [[0.0, 4.0, 0.0]]  # Strong preference for HOMEO
            if 'labels' not in prefs:
                prefs['labels'] = [['LOW', 'HOMEO', 'HIGH']]

        # Add or update transition model section
        if 'transitionModel' not in self.model:
            self.model['transitionModel'] = {
                'EcologicalState': {
                    'controlled': True,
                    'control_factor_idx': 0,
                    'num_controls': 3,
                    'state_labels': ['LOW', 'HOMEO', 'HIGH'],
                    'action_labels': ['DECREASE', 'MAINTAIN', 'INCREASE']
                }
            }
            logger.info("Added default transition model to agent model")
            
    def validate_gnn(self):
        """Validate GNN model has required sections"""
        # First check required sections
        for section in self.required_sections:
            if section not in self.model:
                raise ValueError(f"Missing required section: {section}")
                
        # Validate section contents
        self._validate_state_space()
        self._validate_observations()
        self._validate_transition_model()
        
        # Additional validation for agent models
        if self.model_type == 'agent':
            self._validate_policies()
            self._validate_preferences()
            
        return True
        
    def _validate_state_space(self):
        """Validate state space section"""
        state_space = self.model['stateSpace']
        required = ['factors', 'sizes']
        for req in required:
            if req not in state_space:
                raise ValueError(f"State space missing required field: {req}")
                
    def _validate_observations(self):
        """Validate observations section"""
        obs = self.model['observations']
        required = ['modalities', 'sizes']
        for req in required:
            if req not in obs:
                raise ValueError(f"Observations missing required field: {req}")
                
    def _validate_transition_model(self):
        """Validate transition model section"""
        trans = self.model['transitionModel']
        if not isinstance(trans, dict):
            raise ValueError("Transition model must be a dictionary")
            
    def _validate_policies(self):
        """Validate policies section"""
        policies = self.model['policies']
        required = ['controlFactors', 'numControls', 'policyLen']
        for req in required:
            if req not in policies:
                raise ValueError(f"Policies missing required field: {req}")
                
    def _validate_preferences(self):
        """Validate preferences section"""
        prefs = self.model['preferences']
        required = ['modalities', 'values']
        for req in required:
            if req not in prefs:
                raise ValueError(f"Preferences missing required field: {req}")
    
    def create_matrices(self, output_dir=None):
        """
        Create all PyMDP matrices from GNN definition
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save matrix outputs
            
        Returns
        -------
        dict
            Dictionary containing all matrices needed for PyMDP agent
        """
        logger.info("Creating matrices from GNN model")
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            matrices = {}
            
            # Create A matrices (observation model)
            matrices['A'] = self._create_A_matrices()
            logger.debug(f"Created A matrices with shapes: {[A.shape for A in matrices['A']]}")
            
            # Create B matrices (transition model) 
            matrices['B'] = self._create_B_matrices()
            logger.debug(f"Created B matrices with shapes: {[B.shape for B in matrices['B']]}")
            
            # Create D matrices (initial state priors)
            matrices['D'] = self._create_D_matrices()
            logger.debug(f"Created D matrices with shapes: {[D.shape for D in matrices['D']]}")
            
            # Create initial observation if environment model
            if self.model_type == "environment":
                matrices['initial_obs'] = self._create_initial_obs()
                logger.debug(f"Created initial observation: {matrices['initial_obs']}")
            
            # Create control factor indices
            if 'policies' in self.model:
                matrices['control_fac_idx'] = self.model['policies']['control_fac_idx']
                logger.debug(f"Set control factor indices: {matrices['control_fac_idx']}")
            
            # Create C matrices (preferences)
            if self.model_type == "agent":
                matrices['C'] = self._create_C_matrices()
                logger.debug(f"Created C matrices with shapes: {[C.shape for C in matrices['C']]}")
            
            # Print matrices in readable format
            self.print_matrices(matrices)
            
            if output_dir:
                self.save_matrices(matrices)
                
            return matrices
            
        except Exception as e:
            logger.error(f"Error creating matrices: {str(e)}")
            raise
    
    def _create_A_matrices(self):
        """Create observation model matrices"""
        try:
            A = []
            
            # Get observation modalities from model
            modalities = self.model['observations']['modalities']
            sizes = self.model['observations']['sizes']
            
            # For Biofirm, we have a simple one-to-one mapping between states and observations
            # Create identity matrix for perfect observation
            A_m = np.eye(sizes[0])  # 3x3 identity matrix for LOW, HOMEO, HIGH states
            
            # Add small noise to make observations slightly imperfect
            noise = 0.1
            noise_matrix = noise * np.ones((sizes[0], sizes[0])) / sizes[0]
            A_m = (1 - noise) * A_m + noise_matrix
            
            # Normalize columns to ensure proper probability distribution
            A_m = A_m / A_m.sum(axis=0)
            
            # Log matrix info
            logger.debug(f"\nCreated A matrix for {modalities}:")
            logger.debug(f"Shape: {A_m.shape}")
            logger.debug(f"Values:\n{A_m}")
            
            A.append(A_m)
            
            return A
            
        except Exception as e:
            logger.error(f"Error creating A matrices: {str(e)}")
            raise

    def _create_B_matrices(self):
        """Create state transition matrices"""
        try:
            B = []
            
            # Get state space info
            state_factors = self.model['stateSpace']['factors']
            
            # For Biofirm, we have a single state factor (EcologicalState)
            factor_name = state_factors[0]  # Get first factor name
            factor_info = self.model['transitionModel'][factor_name]
            
            # Get transition matrices for each action
            transitions = factor_info['transitions']
            action_labels = factor_info['action_labels']
            
            # Create B matrix with shape (num_states, num_states, num_actions)
            num_states = self.model['stateSpace']['sizes'][0]
            num_actions = len(action_labels)
            B_f = np.zeros((num_states, num_states, num_actions))
            
            # Fill transition probabilities for each action
            for action_idx, action in enumerate(action_labels):
                B_f[:, :, action_idx] = np.array(transitions[action])
                
            # Validate normalization (columns should sum to 1)
            for action in range(num_actions):
                if not np.allclose(B_f[..., action].sum(axis=0), 1.0):
                    logger.warning(f"B matrix for action {action} not normalized")
                    B_f[..., action] = B_f[..., action] / B_f[..., action].sum(axis=0)
                    
            # Log matrix info
            logger.debug(f"\nCreated B matrix for {factor_name}:")
            logger.debug(f"Shape: {B_f.shape}")
            for action_idx, action in enumerate(action_labels):
                logger.debug(f"\nAction {action}:")
                logger.debug(f"\n{B_f[:, :, action_idx]}")
                
            B.append(B_f)
            
            return B
            
        except Exception as e:
            logger.error(f"Error creating B matrices: {str(e)}")
            raise
    
    def _create_C_matrices(self):
        """Create preference (C) matrices"""
        try:
            if self.model_type != "agent":
                return None
                
            C = []
            
            # Get preferences from model
            if 'preferences' not in self.model:
                raise ValueError("Agent model missing preferences section")
                
            # Get modalities and their sizes
            modalities = self.model['observations']['modalities']
            sizes = self.model['observations']['sizes']
            
            # Get preference values
            pref_values = self.model['preferences']['values']
            
            # Create C matrix for each modality
            for i, modality in enumerate(modalities):
                if i < len(pref_values):
                    C.append(np.array(pref_values[i]))
                else:
                    # Default to neutral preferences if not specified
                    C.append(np.zeros(sizes[i]))
                    logger.warning(f"Using default neutral preferences for modality {modality}")
                    
            # Log matrix info
            for i, (modality, C_m) in enumerate(zip(modalities, C)):
                logger.debug(f"\nCreated C matrix for {modality}:")
                logger.debug(f"Shape: {C_m.shape}")
                logger.debug(f"Values:\n{C_m}")
                
            return C
            
        except Exception as e:
            logger.error(f"Error creating C matrices: {str(e)}")
            raise

    def _create_D_matrices(self):
        """Create initial state prior matrices"""
        try:
            D = []
            
            # Get state space info
            state_factors = self.model['stateSpace']['factors']
            
            # For Biofirm, we have a single state factor (EcologicalState)
            factor_name = state_factors[0]
            
            # Get initial beliefs from model
            if 'initialState' in self.model:
                initial_beliefs = self.model['initialState'][factor_name]
            elif 'initialBeliefs' in self.model:
                initial_beliefs = self.model['initialBeliefs'][factor_name]
            else:
                # Default to uniform distribution if not specified
                num_states = self.model['stateSpace']['sizes'][0]
                initial_beliefs = np.ones(num_states) / num_states
                logger.info(f"Using uniform initial beliefs for {factor_name}")
                
            # Convert to numpy array and ensure normalization
            D_f = np.array(initial_beliefs)
            if not np.allclose(D_f.sum(), 1.0):
                logger.warning(f"Initial beliefs for {factor_name} not normalized")
                D_f = D_f / D_f.sum()
                
            # Log matrix info
            logger.debug(f"\nCreated D matrix for {factor_name}:")
            logger.debug(f"Shape: {D_f.shape}")
            logger.debug(f"Values:\n{D_f}")
            
            D.append(D_f)
            
            return D
            
        except Exception as e:
            logger.error(f"Error creating D matrices: {str(e)}")
            raise

    def _create_initial_obs(self):
        """
        Create initial observation for Biofirm environment model
        
        For Biofirm environment, we need 1 modality:
        1. StateObservation (3 states: LOW, HOMEO, HIGH)
        
        Returns
        -------
        list
            Initial observation for each modality
        """
        logger.debug("Creating initial observation")
        
        # Verify we have the required modality
        modalities = self.model['observations']['modalities']
        if len(modalities) != 1:
            raise ValueError(f"Biofirm requires 1 modality, found {len(modalities)}")
            
        # Initialize with default (HOMEO state)
        initial_obs = [1]  # Start in HOMEO state (index 1)
        
        # Override with values from GNN if specified
        if 'initial_obs' in self.model:
            for modality_name, value in self.model['initial_obs'].items():
                # Find modality index
                for i, modality in enumerate(modalities):
                    if modality['name'] == modality_name:
                        if value == "random":
                            initial_obs[i] = np.random.randint(modality['num_observations'])
                            logger.debug(f"Randomized {modality_name} observation: {initial_obs[i]}")
                        else:
                            initial_obs[i] = value
                            logger.debug(f"Set {modality_name} observation to {value}")
                        break
                        
        logger.info(f"Created initial observation: {initial_obs}")
        logger.debug(f"State: {['LOW', 'HOMEO', 'HIGH'][initial_obs[0]]}")
        
        return initial_obs

    def save_matrices(self, matrices: dict, output_dir: Path):
        """Save matrices to numpy files with proper directory structure"""
        try:
            # Create environment and agent directories
            env_dir = output_dir / "environment"
            agent_dir = output_dir / "agent"
            
            env_dir.mkdir(parents=True, exist_ok=True)
            agent_dir.mkdir(parents=True, exist_ok=True)
            
            # Save matrices based on model type
            save_dir = env_dir if self.model_type == "environment" else agent_dir
            logger.info(f"Saving {self.model_type} matrices to {save_dir}")
            
            # Save A matrices - handle list structure
            if 'A' in matrices:
                A_matrices = matrices['A']
                if isinstance(A_matrices, list) and len(A_matrices) > 0:
                    A = A_matrices[0]  # Get first A matrix
                    logger.info(f"Saving A matrix with shape {A.shape}")
                    np.save(save_dir / "A_matrices.npy", A)
                else:
                    raise ValueError("A matrices not in expected list format")
                    
            # Save B matrices - handle list structure
            if 'B' in matrices:
                B_matrices = matrices['B']
                if isinstance(B_matrices, list) and len(B_matrices) > 0:
                    B = B_matrices[0]  # Get first B matrix
                    logger.info(f"Saving B matrix with shape {B.shape}")
                    np.save(save_dir / "B_matrices.npy", B)
                else:
                    raise ValueError("B matrices not in expected list format")
                    
            # Save C matrices (agent only) - handle list structure
            if 'C' in matrices:
                C_matrices = matrices['C']
                if isinstance(C_matrices, list) and len(C_matrices) > 0:
                    C = C_matrices[0]  # Get first C matrix
                    logger.info(f"Saving C matrix with shape {C.shape}")
                    np.save(save_dir / "C_matrices.npy", C)
                else:
                    raise ValueError("C matrices not in expected list format")
                    
            # Save D matrices - handle list structure
            if 'D' in matrices:
                D_matrices = matrices['D']
                if isinstance(D_matrices, list) and len(D_matrices) > 0:
                    D = D_matrices[0]  # Get first D matrix
                    logger.info(f"Saving D matrix with shape {D.shape}")
                    np.save(save_dir / "D_matrices.npy", D)
                else:
                    raise ValueError("D matrices not in expected list format")
                    
            # Verify files were saved
            saved_files = list(save_dir.glob('*.npy'))
            logger.info(f"Saved {len(saved_files)} matrix files:")
            for f in saved_files:
                # Load and verify each saved matrix
                matrix = np.load(f)
                logger.info(f"- {f.name}: shape {matrix.shape}")
                
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'matrix_shapes': {
                    'A': [list(A.shape) for A in matrices['A']] if 'A' in matrices else [],
                    'B': [list(B.shape) for B in matrices['B']] if 'B' in matrices else [],
                    'C': [list(C.shape) for C in matrices['C']] if 'C' in matrices else [],
                    'D': [list(D.shape) for D in matrices['D']] if 'D' in matrices else []
                },
                'control_fac_idx': matrices.get('control_fac_idx', []),
                'initial_obs': matrices.get('initial_obs', [])
            }
            
            with open(output_dir / "matrices.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved matrix metadata to {output_dir}/matrices.json")
            
        except Exception as e:
            logger.error(f"Error saving matrices: {str(e)}")
            logger.error("Current directory contents:")
            for f in save_dir.glob('*'):
                logger.error(f"- {f.name}")
            raise

    def to_markdown(self, output_file=None):
        """
        Convert GNN model to markdown format
        
        Parameters
        ----------
        output_file : str, optional
            Path to save markdown file
            
        Returns
        -------
        str
            Markdown representation of model
        """
        logger.debug(f"Converting GNN model to markdown format")
        
        md_content = []
        
        # Add title and model type
        md_content.append(f"# {self.model['modelName']}")
        md_content.append(f"\nModel Type: {', '.join(self.model['modelType'])}")
        
        # Add description if available
        if 'description' in self.model:
            md_content.append(f"\n## Description\n{self.model['description']}")
        
        # Add state space information
        md_content.append("\n## State Space")
        for factor in self.model['stateSpace']['factors']:
            md_content.append(f"\n### {factor['name']}")
            md_content.append(f"- Number of states: {factor['num_states']}")
            md_content.append(f"- Controllable: {factor['controllable']}")
            if 'description' in factor:
                md_content.append(f"- Description: {factor['description']}")
            if 'labels' in factor:
                md_content.append(f"- Labels: {', '.join(factor['labels'])}")
        
        # Add observation modalities
        md_content.append("\n## Observations")
        for modality in self.model['observations']['modalities']:
            md_content.append(f"\n### {modality['name']}")
            md_content.append(f"- Number of observations: {modality['num_observations']}")
            md_content.append(f"- Factors observed: {modality['factors_observed']}")
            if 'description' in modality:
                md_content.append(f"- Description: {modality['description']}")
            if 'labels' in modality:
                md_content.append(f"- Labels: {', '.join(modality['labels'])}")
        
        # Add policies if present
        if 'policies' in self.model:
            md_content.append("\n## Policies")
            policies = self.model['policies']
            md_content.append(f"- Number of policies: {policies['num_policies']}")
            md_content.append(f"- Policy length: {policies['policy_len']}")
            md_content.append(f"- Control factors: {policies['control_fac_idx']}")
            if 'description' in policies:
                md_content.append(f"- Description: {policies['description']}")
        
        # Add parameters if present
        if 'parameters' in self.model:
            md_content.append("\n## Parameters")
            for param, value in self.model['parameters'].items():
                md_content.append(f"- {param}: {value}")
        
        # Add matrix shapes
        md_content.append("\n## Matrix Dimensions")
        if hasattr(self, 'A'):
            md_content.append("\n### A Matrices (Observation Model)")
            for i, A in enumerate(self.A):
                md_content.append(f"- A[{i}]: {A.shape}")
        if hasattr(self, 'B'):
            md_content.append("\n### B Matrices (Transition Model)")
            for i, B in enumerate(self.B):
                md_content.append(f"- B[{i}]: {B.shape}")
        
        # Combine content
        markdown_text = "\n".join(md_content)
        
        # Save if output file specified
        if output_file:
            logger.info(f"Saving markdown documentation to {output_file}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(markdown_text)
        
        return markdown_text

    def print_matrices(self, matrices):
        """Print all matrices of the generative model in readable format"""
        logger.info("\nGenerative Model Matrices:")
        
        # Print A matrices (observation model)
        logger.info("\nA Matrices (Observation Model):")
        for i, A in enumerate(matrices['A']):
            modality_name = self.model['observations']['modalities'][i]
            logger.info(f"\nModality: {modality_name}")
            logger.info(f"Shape: {A.shape}")
            logger.info("Values:")
            with np.printoptions(precision=3, suppress=True):
                logger.info(f"\n{A}")
        
        # Print B matrices (transition model)
        logger.info("\nB Matrices (Transition Model):")
        for i, B in enumerate(matrices['B']):
            factor_name = self.model['stateSpace']['factors'][i]
            logger.info(f"\nFactor: {factor_name}")
            logger.info(f"Shape: {B.shape}")
            logger.info("Values:")
            with np.printoptions(precision=3, suppress=True):
                if len(B.shape) == 3:
                    action_labels = self.model['transitionModel'][factor_name]['action_labels']
                    for action_idx, action in enumerate(action_labels):
                        logger.info(f"\nAction {action}:")
                        logger.info(f"{B[:,:,action_idx]}")
                else:
                    logger.info(f"\n{B}")
        
        # Print D matrices (initial state priors)
        logger.info("\nD Matrices (Initial State Priors):")
        for i, D in enumerate(matrices['D']):
            factor_name = self.model['stateSpace']['factors'][i]
            logger.info(f"\nFactor: {factor_name}")
            logger.info(f"Shape: {D.shape}")
            logger.info("Values:")
            with np.printoptions(precision=3, suppress=True):
                logger.info(f"\n{D}")
        
        # Print C matrices (preferences) if agent model
        if 'C' in matrices:
            logger.info("\nC Matrices (Preferences):")
            for i, C in enumerate(matrices['C']):
                modality_name = self.model['observations']['modalities'][i]
                logger.info(f"\nModality: {modality_name}")
                logger.info(f"Shape: {C.shape}")
                logger.info("Values:")
                with np.printoptions(precision=3, suppress=True):
                    logger.info(f"\n{C}")
        
        # Print control factor indices if present
        if 'control_fac_idx' in matrices:
            logger.info(f"\nControl Factor Indices: {matrices['control_fac_idx']}")
        
        # Print initial observations if present
        if 'initial_obs' in matrices:
            logger.info(f"\nInitial Observations: {matrices['initial_obs']}")