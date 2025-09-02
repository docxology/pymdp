"""
Example Utilities
=================

Utilities for creating and running PyMDP textbook examples.
This module provides standardized functions for example creation,
execution, and analysis using real PyMDP methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pymdp_core import PyMDPCore
from visualization import apply_accessibility_enhancements
from pymdp import utils
from pymdp.utils import obj_array_zeros, obj_array_uniform, sample


class ExampleRunner:
    """
    Standardized example runner for PyMDP textbook examples.
    
    This class provides a consistent interface for running examples,
    generating outputs, and creating visualizations.
    """
    
    def __init__(self, example_name, output_dir=None):
        """
        Initialize example runner.
        
        Parameters
        ----------
        example_name : str
            Name of the example (e.g., "01_probability_basics")
        output_dir : str or Path, optional
            Output directory (defaults to outputs/{example_name})
        """
        self.example_name = example_name
        self.output_dir = Path(output_dir) if output_dir else Path("outputs") / example_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply consistent styling
        apply_accessibility_enhancements()
        
        # Initialize results storage
        self.results = {}
        self.visualizations = []
        
    def save_results(self, results_dict, filename="results.json"):
        """
        Save results to JSON file.
        
        Parameters
        ----------
        results_dict : dict
            Results to save
        filename : str
            Output filename
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"📋 Results saved: {output_path}")
    
    def save_visualization(self, fig, filename, title=None, description=None):
        """
        Save visualization with metadata.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Output filename
        title : str, optional
            Figure title
        description : str, optional
            Figure description
        """
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Store visualization metadata
        viz_info = {
            'filename': filename,
            'title': title,
            'description': description,
            'path': str(output_path)
        }
        self.visualizations.append(viz_info)
        
        print(f"📊 Visualization saved: {output_path}")
    
    def create_summary(self):
        """
        Create example summary with all results and visualizations.
        
        Returns
        -------
        summary : dict
            Complete example summary
        """
        summary = {
            'example_name': self.example_name,
            'output_directory': str(self.output_dir),
            'results': self.results,
            'visualizations': self.visualizations,
            'timestamp': str(np.datetime64('now'))
        }
        
        # Save summary
        self.save_results(summary, f"{self.example_name}_summary.json")
        
        return summary


class MatrixBuilder:
    """
    Utility class for building PyMDP matrices using real PyMDP methods.
    
    This class provides standardized methods for creating observation models (A),
    transition models (B), preferences (C), and priors (D).
    """
    
    @staticmethod
    def create_observation_model(num_obs, num_states, model_type="identity", noise=0.0):
        """
        Create observation model (A matrix) using PyMDP utilities.
        
        Parameters
        ----------
        num_obs : int
            Number of observations
        num_states : int
            Number of states
        model_type : str
            Type of model ("identity", "noisy", "random", "custom")
        noise : float
            Noise level for noisy models
            
        Returns
        -------
        A : obj_array
            Observation model matrix
        """
        A = obj_array_zeros([[num_obs, num_states]])
        
        if model_type == "identity":
            # Perfect observation model
            A[0] = np.eye(num_obs, num_states)
        elif model_type == "noisy":
            # Noisy observation model
            A[0] = np.eye(num_obs, num_states) * (1 - noise)
            A[0] += np.ones((num_obs, num_states)) * noise / num_obs
        elif model_type == "random":
            # Random observation model
            A[0] = np.random.rand(num_obs, num_states)
            # Normalize columns
            A[0] = A[0] / A[0].sum(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return A
    
    @staticmethod
    def create_transition_model(num_states, num_actions, model_type="random", **kwargs):
        """
        Create transition model (B matrix) using PyMDP utilities.
        
        Parameters
        ----------
        num_states : int
            Number of states
        num_actions : int
            Number of actions
        model_type : str
            Type of model ("random", "deterministic", "custom")
        **kwargs
            Additional parameters for specific model types
            
        Returns
        -------
        B : obj_array
            Transition model matrix
        """
        B = obj_array_zeros([[num_states, num_states, num_actions]])
        
        if model_type == "random":
            # Random transition model
            for a in range(num_actions):
                B[0][:, :, a] = np.random.rand(num_states, num_states)
                # Normalize columns
                B[0][:, :, a] = B[0][:, :, a] / B[0][:, :, a].sum(axis=0, keepdims=True)
        elif model_type == "deterministic":
            # Deterministic transition model
            for a in range(num_actions):
                for s in range(num_states):
                    # Simple deterministic transitions
                    next_s = (s + a) % num_states
                    B[0][next_s, s, a] = 1.0
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return B
    
    @staticmethod
    def create_preferences(num_obs, pref_type="uniform", **kwargs):
        """
        Create preferences (C vector) using PyMDP utilities.
        
        Parameters
        ----------
        num_obs : int
            Number of observations
        pref_type : str
            Type of preferences ("uniform", "linear", "custom")
        **kwargs
            Additional parameters for specific preference types
            
        Returns
        -------
        C : obj_array
            Preference vector
        """
        C = obj_array_zeros([[num_obs]])
        
        if pref_type == "uniform":
            # Uniform preferences
            C[0] = np.zeros(num_obs)
        elif pref_type == "linear":
            # Linear preferences
            C[0] = np.linspace(-1, 1, num_obs)
        elif pref_type == "custom":
            # Custom preferences
            if 'values' in kwargs:
                C[0] = np.array(kwargs['values'])
            else:
                C[0] = np.zeros(num_obs)
        else:
            raise ValueError(f"Unknown preference type: {pref_type}")
        
        return C
    
    @staticmethod
    def create_prior(num_states, prior_type="uniform", **kwargs):
        """
        Create prior beliefs (D vector) using PyMDP utilities.
        
        Parameters
        ----------
        num_states : int
            Number of states
        prior_type : str
            Type of prior ("uniform", "custom")
        **kwargs
            Additional parameters for specific prior types
            
        Returns
        -------
        D : obj_array
            Prior belief vector
        """
        D = obj_array_zeros([[num_states]])
        
        if prior_type == "uniform":
            # Uniform prior
            D[0] = np.ones(num_states) / num_states
        elif prior_type == "custom":
            # Custom prior
            if 'values' in kwargs:
                D[0] = np.array(kwargs['values'])
                # Normalize
                D[0] = D[0] / D[0].sum()
            else:
                D[0] = np.ones(num_states) / num_states
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        return D


class AnalysisUtils:
    """
    Utility class for analyzing PyMDP results using real PyMDP methods.
    
    This class provides standardized analysis functions for VFE, EFE,
    and other PyMDP computations.
    """
    
    @staticmethod
    def analyze_vfe_components(A, observations, priors, posteriors=None):
        """
        Analyze VFE components using real PyMDP methods.
        
        Parameters
        ----------
        A : obj_array
            Observation model
        observations : list
            List of observations
        priors : list
            List of prior beliefs
        posteriors : list, optional
            List of posterior beliefs (computed if not provided)
            
        Returns
        -------
        analysis : dict
            VFE analysis results
        """
        if posteriors is None:
            posteriors = []
            for obs, prior in zip(observations, priors):
                posterior = PyMDPCore.infer_states(
                    PyMDPCore.create_agent(A, np.eye(3)[:, :, np.newaxis]), 
                    obs
                )
                posteriors.append(posterior)
        
        vfe_results = []
        for obs, prior, posterior in zip(observations, priors, posteriors):
            vfe, components = PyMDPCore.compute_vfe(A, obs, prior, posterior)
            vfe_results.append({
                'observation': obs,
                'vfe': vfe,
                'complexity': components.get('complexity'),
                'accuracy': components.get('accuracy')
            })
        
        return {
            'vfe_results': vfe_results,
            'mean_vfe': np.mean([r['vfe'] for r in vfe_results]),
            'vfe_std': np.std([r['vfe'] for r in vfe_results])
        }
    
    @staticmethod
    def analyze_efe_components(A, B, C, beliefs, policies):
        """
        Analyze EFE components using real PyMDP methods.
        
        Parameters
        ----------
        A : obj_array
            Observation model
        B : obj_array
            Transition model
        C : obj_array
            Preferences
        beliefs : np.ndarray
            Current beliefs
        policies : list
            List of policies
            
        Returns
        -------
        analysis : dict
            EFE analysis results
        """
        efe_results = []
        for policy in policies:
            efe, components = PyMDPCore.compute_efe(A, B, C, beliefs, policy)
            efe_results.append({
                'policy': policy,
                'efe': efe,
                'pragmatic_value': components.get('pragmatic_value'),
                'epistemic_value': components.get('epistemic_value')
            })
        
        return {
            'efe_results': efe_results,
            'best_policy': min(efe_results, key=lambda x: x['efe']),
            'mean_efe': np.mean([r['efe'] for r in efe_results])
        }
    
    @staticmethod
    def compare_inference_methods(A, observations, priors):
        """
        Compare different inference methods using real PyMDP methods.
        
        Parameters
        ----------
        A : obj_array
            Observation model
        observations : list
            List of observations
        priors : list
            List of prior beliefs
            
        Returns
        -------
        comparison : dict
            Inference method comparison results
        """
        # Create agent for inference
        agent = PyMDPCore.create_agent(A, np.eye(3)[:, :, np.newaxis])
        
        results = []
        for obs, prior in zip(observations, priors):
            # PyMDP agent inference
            agent.qs = prior
            pymdp_posterior = PyMDPCore.infer_states(agent, obs)
            
            # Manual Bayes rule
            likelihood = A[0][obs, :]
            joint = likelihood * prior[0]
            manual_posterior = joint / joint.sum()
            
            # Compare results
            results.append({
                'observation': obs,
                'pymdp_posterior': pymdp_posterior[0],
                'manual_posterior': manual_posterior,
                'difference': np.abs(pymdp_posterior[0] - manual_posterior).sum()
            })
        
        return {
            'comparison_results': results,
            'mean_difference': np.mean([r['difference'] for r in results])
        }


def create_standard_example_setup(example_name, num_states=3, num_actions=2, num_obs=None):
    """
    Create standard example setup using real PyMDP methods.
    
    Parameters
    ----------
    example_name : str
        Name of the example
    num_states : int
        Number of states
    num_actions : int
        Number of actions
    num_obs : int, optional
        Number of observations (defaults to num_states)
        
    Returns
    -------
    setup : dict
        Standard example setup including matrices, agent, and runner
    """
    if num_obs is None:
        num_obs = num_states
    
    # Create matrices using PyMDP utilities
    A = MatrixBuilder.create_observation_model(num_obs, num_states, "identity")
    B = MatrixBuilder.create_transition_model(num_states, num_actions, "random")
    C = MatrixBuilder.create_preferences(num_obs, "uniform")
    D = MatrixBuilder.create_prior(num_states, "uniform")
    
    # Create agent using real PyMDP methods
    agent = PyMDPCore.create_agent(A, B, C, D)
    
    # Create example runner
    runner = ExampleRunner(example_name)
    
    return {
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'agent': agent,
        'runner': runner,
        'num_states': num_states,
        'num_actions': num_actions,
        'num_obs': num_obs
    }


def run_standard_analysis(setup, observations, priors=None):
    """
    Run standard analysis using real PyMDP methods.
    
    Parameters
    ----------
    setup : dict
        Example setup from create_standard_example_setup
    observations : list
        List of observations
    priors : list, optional
        List of prior beliefs (uniform if not provided)
        
    Returns
    -------
    analysis : dict
        Complete analysis results
    """
    if priors is None:
        priors = [setup['D']] * len(observations)
    
    # VFE analysis
    vfe_analysis = AnalysisUtils.analyze_vfe_components(
        setup['A'], observations, priors
    )
    
    # EFE analysis
    beliefs = np.ones(setup['num_states']) / setup['num_states']
    policies = [[0], [1]]  # Simple policies
    efe_analysis = AnalysisUtils.analyze_efe_components(
        setup['A'], setup['B'], setup['C'], beliefs, policies
    )
    
    # Inference comparison
    inference_comparison = AnalysisUtils.compare_inference_methods(
        setup['A'], observations, priors
    )
    
    return {
        'vfe_analysis': vfe_analysis,
        'efe_analysis': efe_analysis,
        'inference_comparison': inference_comparison
    }
