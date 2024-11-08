import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class BiofirmUtils:
    """Utility functions specific to Biofirm simulations"""
    
    @staticmethod
    def get_default_simulation_params():
        """Get default simulation parameters"""
        return {
            'n_timesteps': 1000,
            'progress_interval': 500,
            'initial_state': 1,  # Start in HOMEO state
            
            'active_inference': {
                'track_beliefs': True,
                'track_free_energy': True,
                'track_policy_probs': True,
                'track_observations': True,
                'track_actions': True,
                'track_states': True,
                'track_messages': True,
                'track_prediction_errors': True,
                'track_entropy': True,
            },
            
            'states': {
                'LOW': 0,
                'HOMEO': 1, 
                'HIGH': 2
            },
            'actions': {
                'DECREASE': 0,
                'MAINTAIN': 1,
                'INCREASE': 2
            }
        }

    @staticmethod
    def get_biofirm_parameters():
        """Get default parameters for Biofirm active inference"""
        return {
            'ecological_noise': 0.1,
            'controllability': 0.8,
            'policy_precision': 16.0,
            'use_states_info_gain': True,
            'num_states': 3,
            'num_actions': 3,
            'target_state': 1
        }

    @staticmethod
    def process_histories(histories: Dict) -> Dict:
        """Process raw histories into analysis-ready format"""
        try:
            processed = {}
            
            # Process states
            states = []
            for s in histories.get('true_states', []):
                try:
                    if isinstance(s, (list, tuple)) and len(s) > 0:
                        val = s[0]
                        if isinstance(val, np.ndarray):
                            val = val.item()
                        states.append(int(val))
                    elif isinstance(s, np.ndarray):
                        states.append(int(s.item()))
                    else:
                        states.append(int(s))
                except (ValueError, TypeError, IndexError):
                    states.append(np.nan)
            processed['states'] = np.array(states)
            
            # Process actions 
            actions = []
            for a in histories.get('actions', []):
                try:
                    if isinstance(a, (list, tuple)) and len(a) > 0:
                        val = a[0]
                        if isinstance(val, np.ndarray):
                            val = val.item()
                        actions.append(int(val))
                    elif isinstance(a, np.ndarray):
                        actions.append(int(a.item()))
                    else:
                        actions.append(int(a))
                except (ValueError, TypeError, IndexError):
                    actions.append(np.nan)
            processed['actions'] = np.array(actions)
            
            # Process beliefs
            beliefs = []
            for b in histories.get('beliefs', []):
                if isinstance(b, (list, tuple)) and len(b) > 0:
                    belief_probs = b[0]
                    if isinstance(belief_probs, np.ndarray):
                        beliefs.append(belief_probs.flatten())
            processed['beliefs'] = np.array(beliefs) if beliefs else np.array([])
            
            # Process policy probabilities
            policy_probs = []
            for p in histories.get('policy_probs', []):
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    probs = p[0]
                    if isinstance(probs, np.ndarray):
                        policy_probs.append(probs.flatten())
            processed['policy_probs'] = np.array(policy_probs) if policy_probs else np.array([])
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing histories: {str(e)}")
            return {}

    @staticmethod
    def calculate_performance_metrics(histories: Dict) -> Dict:
        """Calculate performance metrics from histories"""
        try:
            processed = BiofirmUtils.process_histories(histories)
            
            metrics = {}
            
            # Calculate time in homeostasis
            valid_states = ~np.isnan(processed['states'])
            if np.any(valid_states):
                metrics['time_in_homeo'] = float(np.mean(processed['states'][valid_states] == 1) * 100)
            else:
                metrics['time_in_homeo'] = 0.0
                
            # Calculate action frequencies
            valid_actions = ~np.isnan(processed['actions'])
            if np.any(valid_actions):
                action_counts = np.bincount(processed['actions'][valid_actions].astype(int), minlength=3)
                metrics['action_frequencies'] = {
                    'decrease': float(action_counts[0] / len(processed['actions'][valid_actions])),
                    'maintain': float(action_counts[1] / len(processed['actions'][valid_actions])),
                    'increase': float(action_counts[2] / len(processed['actions'][valid_actions]))
                }
            else:
                metrics['action_frequencies'] = {'decrease': 0.0, 'maintain': 0.0, 'increase': 0.0}
                
            # Calculate belief entropy
            if len(processed['beliefs']) > 0:
                belief_entropy = -np.sum(processed['beliefs'] * np.log2(processed['beliefs'] + 1e-10), axis=1)
                metrics['mean_belief_entropy'] = float(np.mean(belief_entropy))
            else:
                metrics['mean_belief_entropy'] = 0.0
                
            # Add final state and beliefs
            if len(processed['states']) > 0:
                metrics['final_state'] = int(processed['states'][-1])
            if len(processed['beliefs']) > 0:
                metrics['final_beliefs'] = processed['beliefs'][-1].tolist()
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    @staticmethod
    def save_numerical_results(histories: Dict, output_dir: Path, logger: Optional[logging.Logger] = None) -> None:
        """Save numerical results to JSON"""
        try:
            metrics = BiofirmUtils.calculate_performance_metrics(histories)
            
            results_file = output_dir / "simulation_results.json"
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            if logger:
                logger.info(f"Saved numerical results to {results_file}")
                
        except Exception as e:
            if logger:
                logger.error(f"Error saving numerical results: {str(e)}")

    @staticmethod
    def get_action_label(action: Union[int, np.ndarray]) -> str:
        """Convert action index to label"""
        try:
            if isinstance(action, (list, tuple)):
                action = action[0]
            if isinstance(action, np.ndarray):
                action = action.item()
            action = int(action)
            labels = ['DECREASE', 'MAINTAIN', 'INCREASE']
            return labels[action]
        except (IndexError, ValueError, TypeError):
            return 'UNKNOWN'

    @staticmethod
    def get_state_label(state: Union[int, np.ndarray]) -> str:
        """Convert state index to label"""
        try:
            if isinstance(state, (list, tuple)):
                state = state[0]
            if isinstance(state, np.ndarray):
                state = state.item()
            state = int(state)
            labels = ['LOW', 'HOMEO', 'HIGH']
            return labels[state]
        except (IndexError, ValueError, TypeError):
            return 'UNKNOWN'

    @staticmethod
    def validate_biofirm_matrices(matrices: Dict, model_type: str = "agent") -> bool:
        """Validate matrices specifically for Biofirm model
        
        Parameters
        ----------
        matrices : Dict
            Dictionary containing rendered matrices (A, B, C, D)
        model_type : str
            Either "agent" or "environment" to specify validation rules
            
        Returns
        -------
        bool
            True if validation passes, raises ValueError otherwise
        """
        try:
            # Check required matrices exist
            required = ['A', 'B', 'D']
            for req in required:
                if req not in matrices:
                    raise ValueError(f"Missing required matrix type: {req}")
                    
            # Validate A matrix shape and normalization
            A = matrices['A'][0]  # First modality
            if A.shape != (3, 3):
                raise ValueError(f"Invalid A matrix shape: {A.shape}, expected (3, 3)")
            if not np.allclose(A.sum(axis=0), 1.0):
                raise ValueError("A matrix not normalized (columns must sum to 1)")
                
            # Validate B matrix shape and normalization
            B = matrices['B'][0]  # First factor
            if B.shape != (3, 3, 3):
                raise ValueError(f"Invalid B matrix shape: {B.shape}, expected (3, 3, 3)")
            # Check B matrix normalization for each action
            for action in range(B.shape[-1]):
                if not np.allclose(B[..., action].sum(axis=0), 1.0):
                    logger.warning(f"B matrix for action {action} not normalized (columns should sum to 1)")
                    
            # Validate D matrix shape and normalization
            D = matrices['D'][0]
            if D.shape != (3,):
                raise ValueError(f"Invalid D matrix shape: {D.shape}, expected (3,)")
            if not np.allclose(D.sum(), 1.0):
                raise ValueError("D matrix not normalized (must sum to 1)")
                
            # Additional agent-specific validations
            if model_type == "agent":
                if 'C' not in matrices:
                    raise ValueError("Agent model missing C matrix (preferences)")
                C = matrices['C'][0]
                if C.shape != (3,):
                    raise ValueError(f"Invalid C matrix shape: {C.shape}, expected (3,)")
                    
            # Validate probability matrices are in valid range
            # Note: Only check A, B, D matrices which represent probabilities
            # C matrix (preferences) can have any real values
            prob_matrices = {
                'A': matrices['A'][0],
                'B': matrices['B'][0],
                'D': matrices['D'][0]
            }
            
            for name, matrix in prob_matrices.items():
                if np.any(matrix < 0) or np.any(matrix > 1):
                    raise ValueError(f"{name} matrix contains values outside [0,1] range")
                    
            # Validate special requirements for Biofirm
            if model_type == "agent":
                # Check preference structure favors homeostatic state
                C = matrices['C'][0]
                if not (C[1] > C[0] and C[1] > C[2]):
                    logger.warning("Agent preferences may not properly favor homeostatic state")
                    
                # Check B matrix promotes homeostasis
                B = matrices['B'][0]
                maintain_action = B[..., 1]
                if maintain_action[1,1] < 0.5:
                    logger.warning("MAINTAIN action may not effectively preserve homeostatic state")
                    
            return True
            
        except Exception as e:
            logger.error(f"Matrix validation failed: {str(e)}")
            raise ValueError(f"Matrix validation failed: {str(e)}")

    @staticmethod
    def validate_simulation_histories(histories: Dict) -> bool:
        """Validate simulation histories for completeness and consistency
        
        Parameters
        ----------
        histories : Dict
            Dictionary containing simulation histories
            
        Returns
        -------
        bool
            True if validation passes, raises ValueError otherwise
        """
        try:
            required = [
                'observations',
                'actions',
                'beliefs',
                'free_energies',
                'policy_probs',
                'true_states'
            ]
            
            # Check required histories exist
            for req in required:
                if req not in histories:
                    raise ValueError(f"Missing required history: {req}")
                    
            # Check all histories have same length
            lengths = [len(histories[key]) for key in required]
            if not all(l == lengths[0] for l in lengths):
                raise ValueError("Inconsistent history lengths")
                
            # Validate specific history formats
            for obs in histories['observations']:
                if not isinstance(obs, (list, np.ndarray)) or len(obs) == 0:
                    raise ValueError("Invalid observation format")
                    
            for action in histories['actions']:
                if not isinstance(action, (list, np.ndarray)) or len(action) == 0:
                    raise ValueError("Invalid action format")
                    
            for belief in histories['beliefs']:
                if not isinstance(belief, (list, np.ndarray)) or len(belief) == 0:
                    raise ValueError("Invalid belief format")
                if isinstance(belief[0], np.ndarray) and not np.allclose(belief[0].sum(), 1.0):
                    raise ValueError("Beliefs not normalized")
                    
            return True
            
        except Exception as e:
            logger.error(f"History validation failed: {str(e)}")
            raise ValueError(f"History validation failed: {str(e)}")

    @staticmethod
    def analyze_homeostatic_stability(histories: Dict) -> Dict:
        """Analyze homeostatic control performance
        
        Parameters
        ----------
        histories : Dict
            Simulation histories
            
        Returns
        -------
        Dict
            Stability metrics
        """
        try:
            # Process states
            states = []
            for s in histories.get('true_states', []):
                try:
                    if isinstance(s, (list, tuple)) and len(s) > 0:
                        val = s[0]
                        if isinstance(val, np.ndarray):
                            val = val.item()
                        states.append(int(val))
                except (ValueError, TypeError, IndexError):
                    continue
            states = np.array(states)
            
            # Calculate metrics
            metrics = {}
            
            # Time in each state
            state_counts = np.bincount(states, minlength=3)
            state_probs = state_counts / len(states)
            metrics['state_occupancy'] = {
                'LOW': float(state_probs[0]),
                'HOMEO': float(state_probs[1]),
                'HIGH': float(state_probs[2])
            }
            
            # Transition frequencies
            transitions = np.diff(states)
            metrics['transition_rates'] = {
                'total': float(np.sum(transitions != 0)) / len(states),
                'destabilizing': float(np.sum(transitions != 0) - np.sum(transitions == 0)) / len(states)
            }
            
            # Homeostatic stability
            homeo_runs = []
            current_run = 0
            for s in states:
                if s == 1:  # HOMEO state
                    current_run += 1
                else:
                    if current_run > 0:
                        homeo_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                homeo_runs.append(current_run)
                
            metrics['homeo_stability'] = {
                'mean_duration': float(np.mean(homeo_runs)) if homeo_runs else 0.0,
                'max_duration': float(np.max(homeo_runs)) if homeo_runs else 0.0,
                'num_episodes': len(homeo_runs)
            }
            
            # Recovery metrics
            non_homeo_mask = states != 1
            recovery_times = []
            current_time = 0
            for i, non_homeo in enumerate(non_homeo_mask):
                if non_homeo:
                    current_time += 1
                else:
                    if current_time > 0:
                        recovery_times.append(current_time)
                    current_time = 0
                    
            metrics['recovery'] = {
                'mean_time': float(np.mean(recovery_times)) if recovery_times else 0.0,
                'max_time': float(np.max(recovery_times)) if recovery_times else 0.0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing homeostatic stability: {str(e)}")
            return {}