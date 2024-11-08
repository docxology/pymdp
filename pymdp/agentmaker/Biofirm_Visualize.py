"""
    Biofirm Visualization Script
==========================

Script for generating visualizations from Biofirm simulation results.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import visualization functions
from pymdp.visualization.biofirm_visualization import (
    plot_active_inference_summary,
    plot_belief_dynamics,
    plot_action_frequencies,
    plot_state_transitions,
    plot_policy_probabilities
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default visualization parameters
VIZ_CONFIG = {
    'plot_types': [
        'belief_dynamics',
        'action_frequencies',
        'state_transitions',
        'policy_probabilities'
    ],
    'figure_size': (10, 6),
    'dpi': 300,
    'style': 'seaborn',
    'color_palette': 'husl'
}

class BiofirmVisualizer:
    """Generate visualizations from Biofirm simulation results"""
    
    def __init__(self, simulation_dir: Path, output_dir: Path):
        """Initialize visualizer with directories"""
        try:
            self.simulation_dir = Path(simulation_dir)
            self.output_dir = Path(output_dir)
            
            # Create output directory structure
            self.directories = {
                'root': self.output_dir,
                'core_analysis': self.output_dir / "core_analysis",
                'policy_analysis': self.output_dir / "policy_analysis",
                'inference_cycle': self.output_dir / "inference_cycle",
                'convergence': self.output_dir / "convergence",
                'belief_action': self.output_dir / "belief_action",
                'homeostasis': self.output_dir / "homeostasis"
            }
            
            # Create all directories
            for dir_path in self.directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
                
            # Load simulation data
            self.histories = self._load_histories()
            
            logger.info("BiofirmVisualizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing visualizer: {str(e)}")
            raise
            
    def _load_histories(self) -> dict:
        """Load simulation histories from file"""
        try:
            # Check both possible locations for histories.json
            data_dir = self.simulation_dir / "data"
            possible_locations = [
                data_dir / "histories.json",
                self.simulation_dir / "histories.json"
            ]
            
            histories_file = None
            for loc in possible_locations:
                if loc.exists():
                    histories_file = loc
                    break
                    
            if histories_file is None:
                logger.error("Could not find histories.json in expected locations:")
                for loc in possible_locations:
                    logger.error(f"- {loc}")
                logger.error("Directory contents:")
                self._print_directory_contents(self.simulation_dir)
                raise FileNotFoundError("Could not find histories.json")

            # Load histories
            with open(histories_file) as f:
                histories = json.load(f)

            # Convert lists to numpy arrays
            for key in ['beliefs', 'actions', 'observations', 'true_states', 'policy_probs', 
                       'belief_entropy', 'policy_entropy', 'accuracy']:
                if key in histories:
                    histories[key] = np.array(histories[key])

            # Load additional data files
            self._load_additional_data(histories)

            logger.info("Loaded simulation data successfully")
            return histories

        except Exception as e:
            logger.error(f"Error loading histories: {str(e)}")
            raise
            
    def _load_additional_data(self, histories: Dict):
        """Load additional data files"""
        try:
            # Load results summary
            results_file = self.simulation_dir / "results" / "simulation_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                    histories['summary'] = results['summary']

            # Load metadata
            metadata_file = self.simulation_dir / "data" / "simulation_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    histories['metadata'] = metadata

            # Load metrics
            metrics_file = self.simulation_dir / "metrics" / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    histories['metrics'] = metrics
                    
        except Exception as e:
            logger.warning(f"Error loading additional data: {str(e)}")

    def generate_visualizations(self):
        """Generate all visualizations"""
        try:
            # Create all required directories upfront
            for dir_path in self.directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)

            # Generate core analysis plots
            logger.info("Generating core analysis visualizations...")
            self._generate_core_analysis()

            # Generate policy analysis
            logger.info("Generating policy analysis...")
            self._generate_policy_analysis()

            # Generate homeostasis analysis
            logger.info("Generating homeostasis analysis...")
            self._generate_homeostasis_analysis()

            # Generate belief-action analysis
            logger.info("Generating belief-action analysis...")
            self._generate_belief_action_analysis()

            # Generate convergence analysis
            logger.info("Generating convergence analysis...")
            self._generate_convergence_analysis()

            # Generate inference cycle analysis
            logger.info("Generating inference cycle analysis...")
            self._generate_inference_cycle_analysis()

            # Generate summary time series
            logger.info("Generating summary time series visualization...")
            self._generate_summary_timeseries()

            # Save analysis summary
            self._save_analysis_summary()

            logger.info(f"Generated all visualizations in {self.output_dir}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _generate_core_analysis(self):
        """Generate core analysis visualizations"""
        try:
            core_dir = self.directories['core_analysis']

            # Plot belief dynamics
            plot_belief_dynamics(
                beliefs=self.histories['beliefs'],
                output_file=core_dir / "belief_dynamics.png"
            )

            # Plot action frequencies
            plot_action_frequencies(
                actions=self.histories['actions'],
                output_file=core_dir / "action_frequencies.png"
            )

            # Plot state transitions
            plot_state_transitions(
                states=self.histories['true_states'],
                output_file=core_dir / "state_transitions.png"
            )

            # Plot policy probabilities
            plot_policy_probabilities(
                policy_probs=self.histories['policy_probs'],
                output_file=core_dir / "policy_probabilities.png"
            )

            logger.info("Generated core analysis plots")

        except Exception as e:
            logger.error(f"Error generating core analysis: {str(e)}")
            raise

    def _generate_policy_analysis(self):
        """Generate policy analysis visualizations"""
        try:
            policy_dir = self.directories['policy_analysis']

            # Plot policy entropy over time
            plt.figure(figsize=(10, 6))
            plt.plot(self.histories['policy_entropy'])
            plt.title('Policy Entropy Over Time')
            plt.xlabel('Timestep')
            plt.ylabel('Entropy')
            plt.savefig(policy_dir / "policy_entropy.png")
            plt.close()

            # Plot action selection patterns
            plt.figure(figsize=(10, 6))
            plt.hist(self.histories['actions'], bins=3, density=True)
            plt.title('Action Selection Distribution')
            plt.xlabel('Action')
            plt.ylabel('Frequency')
            plt.savefig(policy_dir / "action_distribution.png")
            plt.close()

            logger.info("Generated policy analysis plots")

        except Exception as e:
            logger.error(f"Error generating policy analysis: {str(e)}")
            raise

    def _generate_homeostasis_analysis(self):
        """Generate homeostasis analysis visualizations"""
        try:
            homeo_dir = self.directories['homeostasis']

            # Plot time in each state
            plt.figure(figsize=(10, 6))
            state_counts = np.bincount(self.histories['true_states'], minlength=3)
            plt.bar(['LOW', 'HOMEO', 'HIGH'], state_counts / len(self.histories['true_states']))
            plt.title('Time Spent in Each State')
            plt.ylabel('Proportion of Time')
            plt.savefig(homeo_dir / "state_distribution.png")
            plt.close()

            # Plot state transitions over time with homeostasis highlighted
            plt.figure(figsize=(12, 6))
            plt.plot(self.histories['true_states'], 'b-', alpha=0.5)
            plt.axhline(y=1, color='g', linestyle='--', label='Homeostasis')
            plt.fill_between(range(len(self.histories['true_states'])), 
                           self.histories['true_states'] == 1, 
                           alpha=0.3, color='g')
            plt.title('State Trajectory with Homeostasis')
            plt.xlabel('Timestep')
            plt.ylabel('State')
            plt.legend()
            plt.savefig(homeo_dir / "homeostasis_trajectory.png")
            plt.close()

            logger.info("Generated homeostasis analysis plots")

        except Exception as e:
            logger.error(f"Error generating homeostasis analysis: {str(e)}")
            raise

    def _generate_belief_action_analysis(self):
        """Generate belief-action relationship visualizations"""
        try:
            belief_action_dir = self.directories['belief_action']

            # Plot belief-action correlation
            plt.figure(figsize=(10, 6))
            max_beliefs = np.argmax(self.histories['beliefs'], axis=1)
            timesteps = np.arange(len(max_beliefs))
            
            # Create separate scatter plots for each action
            for action in range(3):
                # Get indices where this action was taken
                action_indices = np.where(self.histories['actions'] == action)[0]
                if len(action_indices) > 0:
                    plt.scatter(
                        timesteps[action_indices],
                        max_beliefs[action_indices],
                        label=f'Action {action}',
                        alpha=0.5
                    )
                    
            plt.title('Belief-Action Relationship')
            plt.xlabel('Timestep')
            plt.ylabel('Most Likely State')
            plt.yticks([0, 1, 2], ['LOW', 'HOMEO', 'HIGH'])
            plt.legend()
            plt.savefig(belief_action_dir / "belief_action_correlation.png")
            plt.close()

            # Plot action selection confidence
            plt.figure(figsize=(10, 6))
            belief_certainty = np.max(self.histories['beliefs'], axis=1)
            
            # Create scatter plot with color mapping
            scatter = plt.scatter(
                timesteps,
                belief_certainty,
                c=self.histories['actions'],
                cmap='viridis',
                alpha=0.5
            )
            
            # Add colorbar with action labels
            cbar = plt.colorbar(scatter)
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(['DECREASE', 'MAINTAIN', 'INCREASE'])
            
            plt.title('Action Selection Confidence')
            plt.xlabel('Timestep')
            plt.ylabel('Belief Certainty')
            plt.savefig(belief_action_dir / "action_confidence.png")
            plt.close()

            logger.info("Generated belief-action analysis plots")

        except Exception as e:
            logger.error(f"Error generating belief-action analysis: {str(e)}")
            raise

    def _generate_convergence_analysis(self):
        """Generate convergence analysis visualizations"""
        try:
            conv_dir = self.directories['convergence']

            # Plot belief convergence
            plt.figure(figsize=(12, 6))
            belief_entropy = self.histories.get('belief_entropy', [])
            if len(belief_entropy) > 0:
                plt.plot(belief_entropy, label='Belief Entropy')
                plt.axhline(y=np.mean(belief_entropy), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(belief_entropy):.2f}')
                plt.xlabel('Time')
                plt.ylabel('Entropy')
                plt.title('Belief Convergence')
                plt.legend()
                plt.grid(True)
                plt.savefig(conv_dir / "belief_convergence.png")
            plt.close()

            # Plot policy convergence
            plt.figure(figsize=(12, 6))
            policy_entropy = self.histories.get('policy_entropy', [])
            if len(policy_entropy) > 0:
                plt.plot(policy_entropy, label='Policy Entropy')
                plt.axhline(y=np.mean(policy_entropy), color='r', linestyle='--',
                           label=f'Mean: {np.mean(policy_entropy):.2f}')
                plt.xlabel('Time')
                plt.ylabel('Entropy')
                plt.title('Policy Selection Convergence')
                plt.legend()
                plt.grid(True)
                plt.savefig(conv_dir / "policy_convergence.png")
            plt.close()

            logger.info("Generated convergence analysis plots")

        except Exception as e:
            logger.error(f"Error generating convergence analysis: {str(e)}")
            raise

    def _generate_inference_cycle_analysis(self):
        """Generate inference cycle analysis visualizations"""
        try:
            cycle_dir = self.directories['inference_cycle']

            # Plot belief updates vs observations
            plt.figure(figsize=(12, 6))
            max_beliefs = np.argmax(self.histories['beliefs'], axis=1)
            plt.plot(max_beliefs, label='Most Likely State', alpha=0.7)
            plt.plot(self.histories['observations'], label='Observations', alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel('State')
            plt.title('Belief Updates vs Observations')
            plt.yticks([0, 1, 2], ['LOW', 'HOMEO', 'HIGH'])
            plt.legend()
            plt.grid(True)
            plt.savefig(cycle_dir / "belief_observation_cycle.png")
            plt.close()

            # Plot prediction errors if available
            if 'prediction_errors' in self.histories:
                plt.figure(figsize=(12, 6))
                plt.plot(self.histories['prediction_errors'])
                plt.xlabel('Time')
                plt.ylabel('Prediction Error')
                plt.title('State Prediction Errors')
                plt.grid(True)
                plt.savefig(cycle_dir / "prediction_errors.png")
                plt.close()

            # Plot belief accuracy with dynamic window size
            if 'accuracy' in self.histories:
                plt.figure(figsize=(12, 6))
                accuracy = self.histories['accuracy']
                timesteps = len(accuracy)
                
                # Adjust window size based on total timesteps
                window = min(50, max(2, timesteps // 10))  # At least 2, at most 50
                
                if timesteps > window:  # Only compute rolling average if enough data
                    rolling_acc = np.convolve(accuracy, np.ones(window)/window, mode='valid')
                    valid_timesteps = range(window-1, timesteps)
                    
                    plt.plot(accuracy, alpha=0.3, label='Raw Accuracy')
                    plt.plot(valid_timesteps, rolling_acc, 
                            label=f'{window}-step Rolling Average')
                else:
                    plt.plot(accuracy, label='Accuracy')
                    
                plt.axhline(y=np.mean(accuracy), color='k', linestyle=':', 
                           label=f'Mean: {np.mean(accuracy):.2f}')
                plt.xlabel('Time')
                plt.ylabel('Accuracy')
                plt.title('Belief Accuracy Over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig(cycle_dir / "belief_accuracy.png")
                plt.close()

            logger.info("Generated inference cycle analysis plots")

        except Exception as e:
            logger.error(f"Error generating inference cycle analysis: {str(e)}")
            raise

    def _generate_summary_timeseries(self):
        """Generate comprehensive stacked time series visualization"""
        try:
            # Calculate number of subplots needed
            n_plots = 6  # State, Beliefs, Actions, Policy Probs, Entropy, Accuracy
            fig_height = 3 * n_plots  # 3 inches per subplot
            
            # Create figure with shared x-axis
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, fig_height), 
                                    sharex=True, constrained_layout=True)
            
            timesteps = range(len(self.histories['true_states']))
            
            # 1. State Transitions
            ax = axes[0]
            ax.plot(self.histories['true_states'], 'b-', alpha=0.7, label='True State')
            ax.plot(self.histories['observations'], 'r--', alpha=0.5, label='Observation')
            ax.axhline(y=1, color='g', linestyle=':', label='Homeostasis')
            ax.fill_between(timesteps, self.histories['true_states'] == 1, 
                           alpha=0.1, color='g', label='Homeo Zone')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['LOW', 'HOMEO', 'HIGH'])
            ax.set_title('Environmental State & Observations')
            ax.legend(loc='right')
            ax.grid(True, alpha=0.3)
            
            # 2. Belief Dynamics
            ax = axes[1]
            labels = ['LOW', 'HOMEO', 'HIGH']
            for i in range(self.histories['beliefs'].shape[1]):
                ax.plot(self.histories['beliefs'][:, i], label=labels[i])
            ax.set_title('Belief Dynamics')
            ax.set_ylabel('Probability')
            ax.legend(loc='right')
            ax.grid(True, alpha=0.3)
            
            # 3. Actions
            ax = axes[2]
            ax.plot(self.histories['actions'], 'k-', drawstyle='steps-post')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['DECREASE', 'MAINTAIN', 'INCREASE'])
            ax.set_title('Action Selection')
            ax.grid(True, alpha=0.3)
            
            # 4. Policy Probabilities
            ax = axes[3]
            labels = ['DECREASE', 'MAINTAIN', 'INCREASE']
            for i in range(self.histories['policy_probs'].shape[1]):
                ax.plot(self.histories['policy_probs'][:, i], label=labels[i])
            ax.set_title('Policy Probabilities')
            ax.set_ylabel('Probability')
            ax.legend(loc='right')
            ax.grid(True, alpha=0.3)
            
            # 5. Entropy
            ax = axes[4]
            belief_entropy = self.histories.get('belief_entropy', [])
            policy_entropy = self.histories.get('policy_entropy', [])
            
            if len(belief_entropy) > 0 and len(policy_entropy) > 0:
                # Determine appropriate window size based on data length
                data_length = len(belief_entropy)
                window = min(50, max(2, data_length // 10))  # At least 2, at most 50
                
                if data_length > window:
                    # Calculate rolling averages
                    belief_roll = np.convolve(belief_entropy, 
                                            np.ones(window)/window, mode='valid')
                    policy_roll = np.convolve(policy_entropy, 
                                            np.ones(window)/window, mode='valid')
                    valid_timesteps = range(window-1, data_length)
                    
                    # Plot raw and smoothed
                    ax.plot(belief_entropy, 'b-', alpha=0.2, label='Belief Entropy')
                    ax.plot(policy_entropy, 'r-', alpha=0.2, label='Policy Entropy')
                    ax.plot(valid_timesteps, belief_roll, 
                           'b-', label=f'Belief ({window}-step avg)')
                    ax.plot(valid_timesteps, policy_roll, 
                           'r-', label=f'Policy ({window}-step avg)')
                else:
                    # Just plot raw values for short sequences
                    ax.plot(belief_entropy, 'b-', label='Belief Entropy')
                    ax.plot(policy_entropy, 'r-', label='Policy Entropy')
                    
            ax.set_title('System Entropy')
            ax.legend(loc='right')
            ax.grid(True, alpha=0.3)
            
            # 6. Accuracy
            ax = axes[5]
            accuracy = self.histories.get('accuracy', [])
            if len(accuracy) > 0:
                # Determine window size based on data length
                data_length = len(accuracy)
                window = min(50, max(2, data_length // 10))
                
                if data_length > window:
                    # Calculate rolling average
                    acc_roll = np.convolve(accuracy, np.ones(window)/window, mode='valid')
                    valid_timesteps = range(window-1, data_length)
                    
                    ax.plot(accuracy, 'g-', alpha=0.2, label='Raw Accuracy')
                    ax.plot(valid_timesteps, acc_roll, 
                           'g-', label=f'{window}-step Average')
                else:
                    # Just plot raw values for short sequences
                    ax.plot(accuracy, 'g-', label='Accuracy')
                    
                ax.axhline(y=np.mean(accuracy), color='k', linestyle=':', 
                          label=f'Mean: {np.mean(accuracy):.2f}')
                
            ax.set_title('Belief Accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Timestep')
            ax.legend(loc='right')
            ax.grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle('Biofirm Active Inference Summary', 
                        fontsize=14, y=1.02)
            
            # Save figure
            summary_file = self.output_dir / "summary_timeseries.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Generated summary time series visualization")
            
        except Exception as e:
            logger.error(f"Error generating summary time series: {str(e)}")
            raise

    def _save_analysis_summary(self):
        """Save analysis summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'plots_generated': {
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
                    ],
                    'belief_action': [
                        'belief_action_correlation.png',
                        'action_confidence.png'
                    ]
                },
                'performance_metrics': {
                    'homeostasis': {
                        'time_in_homeo': float(np.mean(self.histories['true_states'] == 1)),
                        'transitions': int(len(np.where(np.diff(self.histories['true_states']))[0])),
                        'control_efficiency': float(np.mean(self.histories['actions'] == 1))
                    },
                    'belief': {
                        'mean_entropy': float(np.mean(self.histories['belief_entropy'])),
                        'mean_accuracy': float(np.mean(self.histories['accuracy']))
                    },
                    'policy': {
                        'mean_entropy': float(np.mean(self.histories['policy_entropy'])),
                        'action_distribution': np.bincount(self.histories['actions']).tolist()
                    }
                }
            }

            summary_file = self.output_dir / "analysis_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved analysis summary to {summary_file}")

        except Exception as e:
            logger.error(f"Error saving analysis summary: {str(e)}")
            raise

    def _print_directory_contents(self, directory: Path, indent: int = 0):
        """Print directory contents recursively"""
        for path in directory.iterdir():
            logger.error("  " * indent + f"- {path.name}")
            if path.is_dir():
                self._print_directory_contents(path, indent + 1)

    def validate_output(self) -> bool:
        """Validate all visualization outputs"""
        try:
            # Define required plots for each analysis type
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
                ],
                'belief_action': [
                    'belief_action_correlation.png',
                    'action_confidence.png'
                ]
            }
            
            # Check each required plot exists
            missing_plots = []
            for dir_name, plots in required_plots.items():
                plot_dir = self.directories[dir_name]
                for plot in plots:
                    plot_path = plot_dir / plot
                    if not plot_path.exists():
                        missing_plots.append(str(plot_path))
                        
            if missing_plots:
                logger.error("Missing required plots:")
                for plot in missing_plots:
                    logger.error(f"- {plot}")
                return False
                
            # Verify analysis summary exists
            summary_file = self.output_dir / "analysis_summary.json"
            if not summary_file.exists():
                logger.error("Missing analysis summary file")
                return False
                
            # Load and validate summary content
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                    
                required_fields = ['timestamp', 'plots_generated', 'performance_metrics']
                for field in required_fields:
                    if field not in summary:
                        logger.error(f"Missing required field in summary: {field}")
                        return False
                        
            except Exception as e:
                logger.error(f"Error validating summary file: {str(e)}")
                return False
                
            logger.info("Validated all visualization outputs successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating outputs: {str(e)}")
            return False

def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Generate Biofirm visualizations')
        parser.add_argument('--simulation_dir', type=str, required=True,
                          help='Path to simulation directory')
        parser.add_argument('--output_dir', type=str, required=True,
                          help='Path to output directory')
        args = parser.parse_args()
        
        # Create visualizer and generate plots
        visualizer = BiofirmVisualizer(
            simulation_dir=args.simulation_dir,
            output_dir=args.output_dir
        )
        
        # Generate visualizations
        visualizer.generate_visualizations()
        
        # Validate outputs
        if not visualizer.validate_output():
            logger.warning("Some visualizations may be missing but continuing...")
            
        logger.info(f"Visualizations saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
