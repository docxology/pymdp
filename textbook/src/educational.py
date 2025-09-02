"""
Educational Utilities
=====================

Interactive demonstrations and step-by-step explanations of active inference concepts.
Designed for learning and teaching PyMDP and active inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from pymdp.utils import obj_array_zeros, obj_array_uniform
# Note: infer_states is a method on Agent class, not a standalone function
from pymdp.maths import softmax
try:
    from .visualization import plot_beliefs, plot_observation_model
except ImportError:
    from visualization import plot_beliefs, plot_observation_model


class StepByStepInference:
    """
    Step-by-step demonstration of active inference computations.
    """
    
    def __init__(self, A: np.ndarray, prior: np.ndarray, verbose: bool = True):
        """
        Initialize step-by-step inference demonstration.
        
        Parameters
        ----------
        A : np.ndarray
            Observation model
        prior : np.ndarray  
            Prior beliefs
        verbose : bool
            Whether to print explanations
        """
        self.A = A
        self.prior = prior
        self.verbose = verbose
        self.steps = []
    
    def observe(self, observation: int) -> np.ndarray:
        """
        Demonstrate state inference step by step.
        
        Parameters
        ----------
        observation : int
            Observed value
            
        Returns
        -------
        posterior : np.ndarray
            Posterior beliefs
        """
        
        if self.verbose:
            print("=" * 60)
            print("STEP-BY-STEP ACTIVE INFERENCE")
            print("=" * 60)
            print(f"\n1. OBSERVATION: {observation}")
            print("-" * 30)
        
        # Step 1: Get likelihood
        likelihood = self.A[observation, :]
        
        if self.verbose:
            print(f"Likelihood P(obs={observation} | state): {likelihood}")
            print(f"This tells us how likely each state is to produce observation {observation}")
        
        # Step 2: Apply Bayes rule
        if self.verbose:
            print(f"\n2. BAYES RULE COMPUTATION")
            print("-" * 30)
            print(f"Prior P(state): {self.prior}")
            print(f"Likelihood P(obs | state): {likelihood}")
        
        # Joint probability
        joint = likelihood * self.prior
        
        if self.verbose:
            print(f"Joint P(obs, state) = P(obs|state) × P(state): {joint}")
        
        # Evidence (normalization constant)
        evidence = np.sum(joint)
        
        if self.verbose:
            print(f"Evidence P(obs) = Σ P(obs, state): {evidence}")
        
        # Posterior
        posterior = joint / evidence
        
        if self.verbose:
            print(f"Posterior P(state | obs) = P(obs, state) / P(obs): {posterior}")
            print(f"\n3. INTERPRETATION")
            print("-" * 30)
            
            most_likely_state = np.argmax(posterior)
            confidence = posterior[most_likely_state]
            
            print(f"Most likely state: {most_likely_state} (confidence: {confidence:.3f})")
            print(f"Certainty increased from prior to posterior")
            
            # Information gained
            prior_entropy = -np.sum(self.prior * np.log(self.prior + 1e-16))
            posterior_entropy = -np.sum(posterior * np.log(posterior + 1e-16))
            info_gain = prior_entropy - posterior_entropy
            
            print(f"Information gained: {info_gain:.3f} bits")
            print(f"(Prior entropy: {prior_entropy:.3f}, Posterior entropy: {posterior_entropy:.3f})")
        
        # Store step
        step_info = {
            'observation': observation,
            'prior': self.prior.copy(),
            'likelihood': likelihood.copy(),
            'joint': joint.copy(), 
            'evidence': evidence,
            'posterior': posterior.copy()
        }
        self.steps.append(step_info)
        
        # Update prior for next step
        self.prior = posterior
        
        return posterior
    
    def visualize_last_step(self) -> plt.Figure:
        """
        Visualize the last inference step.
        
        Returns
        -------
        fig : plt.Figure
            Figure with visualizations
        """
        
        if not self.steps:
            raise ValueError("No inference steps to visualize")
        
        step = self.steps[-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Inference Step: Observation {step['observation']}")
        
        # Prior
        plot_beliefs(step['prior'], title="Prior P(state)", ax=axes[0, 0])
        
        # Likelihood  
        plot_beliefs(step['likelihood'], title=f"Likelihood P(obs={step['observation']} | state)", ax=axes[0, 1])
        
        # Joint
        plot_beliefs(step['joint'], title="Joint P(obs, state)", ax=axes[1, 0])
        
        # Posterior
        plot_beliefs(step['posterior'], title="Posterior P(state | obs)", ax=axes[1, 1])
        
        plt.tight_layout()
        return fig


def interactive_demo(
    model_type: str = "simple",
    **kwargs
) -> Dict[str, Any]:
    """
    Create an interactive demonstration of active inference.
    
    Parameters
    ----------
    model_type : str
        Type of model to demonstrate ("simple", "gridworld", "tmaze")
    **kwargs
        Additional parameters for model creation
        
    Returns
    -------
    demo : dict
        Dictionary containing demonstration components
    """
    
    demo = {}
    
    if model_type == "simple":
        # Simple 2-state, 3-observation model
        A = obj_array_zeros([[3, 2]])
        A[0] = np.array([[0.8, 0.2],    # obs 0: likely from state 0
                         [0.1, 0.9],    # obs 1: likely from state 1
                         [0.1, 0.1]])   # obs 2: uninformative
        
        prior = obj_array_uniform([2])
        
        demo['A'] = A
        demo['prior'] = prior[0]
        demo['model_type'] = 'simple'
        demo['description'] = "Simple 2-state model with 3 observations"
        
    elif model_type == "gridworld":
        # Create small gridworld
        from .model_utils import create_gridworld_model
        A, B, C, D = create_gridworld_model(height=3, width=3, **kwargs)
        
        demo['A'] = A
        demo['B'] = B  
        demo['C'] = C
        demo['prior'] = D[0]
        demo['model_type'] = 'gridworld'
        demo['description'] = "3x3 gridworld with goal and obstacles"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Add step-by-step inference helper
    demo['inference_helper'] = StepByStepInference(demo['A'][0], demo['prior'])
    
    # Add visualization helper
    def visualize_model():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Observation model
        plot_observation_model(demo['A'][0], ax=axes[0])
        
        # Prior beliefs
        plot_beliefs(demo['prior'], title="Prior Beliefs", ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    demo['visualize_model'] = visualize_model
    
    return demo


def concept_illustration(concept: str) -> Dict[str, Any]:
    """
    Create illustrations of key active inference concepts.
    
    Parameters
    ----------
    concept : str
        Concept to illustrate ("bayes_rule", "precision", "free_energy", etc.)
        
    Returns
    -------
    illustration : dict
        Dictionary containing illustration components
    """
    
    illustration = {}
    
    if concept == "bayes_rule":
        # Illustrate Bayes rule with simple example
        
        # Set up scenario
        prior = np.array([0.3, 0.7])  # Prior: more likely state 1
        likelihood_obs0 = np.array([0.9, 0.2])  # Obs 0 favors state 0
        likelihood_obs1 = np.array([0.1, 0.8])  # Obs 1 favors state 1
        
        # Compute posteriors
        joint_obs0 = prior * likelihood_obs0
        posterior_obs0 = joint_obs0 / np.sum(joint_obs0)
        
        joint_obs1 = prior * likelihood_obs1  
        posterior_obs1 = joint_obs1 / np.sum(joint_obs1)
        
        illustration['prior'] = prior
        illustration['posteriors'] = [posterior_obs0, posterior_obs1]
        illustration['observations'] = [0, 1]
        illustration['description'] = "How different observations update beliefs"
        
        def visualize():
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            plot_beliefs(prior, title="Prior P(state)", ax=axes[0])
            plot_beliefs(posterior_obs0, title="Posterior after obs 0", ax=axes[1]) 
            plot_beliefs(posterior_obs1, title="Posterior after obs 1", ax=axes[2])
            
            plt.suptitle("Bayes Rule: How Observations Update Beliefs")
            plt.tight_layout()
            return fig
            
        illustration['visualize'] = visualize
        
    elif concept == "precision":
        # Illustrate effect of precision on inference
        
        base_obs_model = np.array([[0.7, 0.3],
                                   [0.4, 0.6]])
        
        precisions = [0.5, 1.0, 2.0, 5.0]
        posteriors = []
        
        prior = np.array([0.5, 0.5])
        obs = 0  # Observe first observation
        
        for precision in precisions:
            # Apply precision by exponentiating likelihood
            likelihood = base_obs_model[obs, :]
            precise_likelihood = likelihood ** precision
            
            # Compute posterior
            joint = prior * precise_likelihood
            posterior = joint / np.sum(joint)
            posteriors.append(posterior)
        
        illustration['precisions'] = precisions
        illustration['posteriors'] = posteriors
        illustration['description'] = "How precision affects confidence in beliefs"
        
        def visualize():
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            for i, (precision, posterior) in enumerate(zip(precisions, posteriors)):
                plot_beliefs(posterior, title=f"Precision = {precision}", ax=axes[i])
                
            plt.suptitle("Effect of Precision on Posterior Beliefs")
            plt.tight_layout()
            return fig
            
        illustration['visualize'] = visualize
        
    elif concept == "free_energy":
        # Illustrate free energy computation
        
        # Simple scenario
        true_state = 1
        observations = [0, 1, 1, 0]  
        
        A = obj_array_zeros([[2, 2]])
        A[0] = np.array([[0.8, 0.2],
                         [0.2, 0.8]])
        
        prior = obj_array_uniform([2])
        
        # Compute free energy over time
        beliefs_history = []
        free_energies = []
        
        current_beliefs = prior[0]
        
        for obs in observations:
            # Inference step using PyMDP update_posterior_states
            from pymdp.inference import update_posterior_states
            posterior = update_posterior_states(A, None, [obs], None, prior=obj_array_zeros([2]))
            posterior[0][:] = current_beliefs  # Use current beliefs as prior
            posterior = update_posterior_states(A, None, [obs], None, prior=posterior)
            
            # Compute free energy (simplified)
            likelihood = A[0][obs, :]
            expected_log_likelihood = np.sum(posterior[0] * np.log(likelihood + 1e-16))
            kl_div = np.sum(posterior[0] * np.log(posterior[0] / current_beliefs + 1e-16))
            free_energy = kl_div - expected_log_likelihood
            
            beliefs_history.append(posterior[0].copy())
            free_energies.append(free_energy)
            current_beliefs = posterior[0]
        
        illustration['observations'] = observations
        illustration['beliefs_history'] = beliefs_history
        illustration['free_energies'] = free_energies
        illustration['description'] = "Free energy minimization over time"
        
    else:
        raise ValueError(f"Unknown concept: {concept}")
    
    return illustration


def create_tutorial_sequence() -> List[Dict[str, Any]]:
    """
    Create a sequence of tutorials building up to full active inference.
    
    Returns
    -------
    tutorials : list of dict
        Sequence of tutorial steps
    """
    
    tutorials = []
    
    # Tutorial 1: Probability basics
    tutorials.append({
        'title': 'Probability Distributions',
        'description': 'Understanding probability distributions and normalization',
        'concepts': ['probability', 'normalization', 'entropy'],
        'interactive': True
    })
    
    # Tutorial 2: Bayes rule
    tutorials.append({
        'title': 'Bayes Rule and Belief Update',
        'description': 'How observations update beliefs using Bayes rule',
        'concepts': ['bayes_rule', 'prior', 'likelihood', 'posterior'],
        'interactive': True
    })
    
    # Tutorial 3: Observation models
    tutorials.append({
        'title': 'Observation Models',
        'description': 'Specifying how hidden states generate observations',
        'concepts': ['observation_model', 'likelihood', 'generative_model'],
        'interactive': True
    })
    
    # Tutorial 4: State inference
    tutorials.append({
        'title': 'State Inference',
        'description': 'Inferring hidden states from observations',
        'concepts': ['inference', 'hidden_states', 'uncertainty'],
        'interactive': True
    })
    
    # Tutorial 5: Action and control
    tutorials.append({
        'title': 'Action Selection',
        'description': 'How agents select actions to minimize expected free energy',
        'concepts': ['action_selection', 'expected_free_energy', 'exploration'],
        'interactive': True
    })
    
    # Tutorial 6: Full active inference
    tutorials.append({
        'title': 'Full Active Inference',
        'description': 'Putting it all together: perception, action, and learning',
        'concepts': ['active_inference', 'perception_action_loop', 'learning'],
        'interactive': True
    })
    
    return tutorials
