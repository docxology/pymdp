"""
Biofirm Shapley Value Analysis
=============================

This script evaluates the marginal contributions of different Biofirm agents using Shapley value analysis.

Shapley Value Calculation Logic:
------------------------------
1. Agent Variations:
   - Base agent: Standard homeostatic preferences
   - Risk-averse agent: Higher preference for homeostatic state
   - Exploratory agent: Lower preference differential between states

2. Coalition Formation:
   - Generates all possible combinations of agents (2^n - 1 coalitions)
   - Each coalition represents a group of agents working together

3. Performance Evaluation:
   For each coalition:
   - Runs complete Biofirm simulation
   - Measures key metrics:
     * Homeostasis satisfaction (time in target state)
     * Expected free energy (uncertainty minimization)
     * Belief accuracy (state estimation)
     * Control efficiency (action selection)

4. Shapley Value Computation:
   For each agent i and metric m:
   Shapley_value(i,m) = Σ [|S|!*(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]
   Where:
   - S: Coalition subset not containing agent i
   - v(S): Performance value of coalition S
   - n: Total number of agents

5. Interpretation:
   - Positive values: Agent improves collective performance
   - Negative values: Agent potentially interferes with others
   - Magnitude: Relative importance of agent's contribution

Directory Structure:
------------------
shapley_experiment_TIMESTAMP/
├── 1_agents/                    # Agent variations
├── 2_coalitions/               # Coalition simulation results
│   └── coalition_N_[ids]/      # Results for each coalition
├── 3_analysis/                 # Shapley analysis outputs
│   ├── results/                # Numerical results
│   ├── plots/                  # Visualizations
│   └── metrics/                # Performance metrics
└── logs/                       # Execution logs

The script provides insights into:
1. Individual agent contributions to collective performance
2. Synergies between different agent types
3. Trade-offs between different performance metrics
4. Optimal agent combinations for specific goals
"""

import os
import sys
import json
import time
import shutil
import logging
import numpy as np
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, TextIO, Union
from concurrent.futures import ProcessPoolExecutor

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import local modules
from pymdp.agentmaker.Run_Biofirm import BiofirmExperiment
from pymdp.agentmaker.biofirm_execution_utils import BiofirmExecutionUtils
from pymdp.agentmaker.config import DEFAULT_PATHS, EXPERIMENT_CONFIG

# Move plot settings into a function to avoid import issues
def get_plot_settings():
    """Get plot settings after matplotlib is imported"""
    import matplotlib.pyplot as plt
    
    return {
        'figsize': {
            'shapley': (24, 18),  # Increased size for 5 agents
            'coalition': (28, 20),  # Larger for more combinations
            'synergy': (14, 12),   # Adjusted for 5x5 matrix
            'correlation': (10, 8)
        },
        'style': 'default',
        'dpi': 300,
        'colors': {
            'agents': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],  # Five colors
            'coalitions': plt.cm.viridis,
            'synergy': 'RdYlBu',
            'correlation': 'coolwarm'
        },
        'font': {
            'title': 16,
            'label': 12,
            'tick': 10,
            'annotation': 9
        }
    }

# Update SHAPLEY_CONFIG to remove matplotlib dependency
SHAPLEY_CONFIG = {
    # Base agent and environment paths
    'base_agent': Path(__file__).parent.parent / "gnn" / "models" / "gnn_agent_biofirm.gnn",
    'env_gnn': Path(__file__).parent.parent / "gnn" / "envs" / "gnn_env_biofirm.gnn",
    
    # Base output directory
    'output_base': Path(__file__).parent / "sandbox" / "Biofirm",
    
    # Updated agent variations for N=5
    'agent_variations': [
        {
            'name': 'base',
            'preferences': [[0.0, 4.0, 0.0]]  # Standard preferences
        },
        {
            'name': '0-6-0',
            'preferences': [[0.0, 6.0, 0.0]]  # Higher preference for HOMEO
        },
        {
            'name': '.1-2-.1',
            'preferences': [[0.1, 2.0, 0.1]]  # Lower preference differential
        },
        {
            'name': 'Extreme',
            'preferences': [[2, .1, 2]]  # Preferences for the Extremes. 
        },
        {
            'name': '2-2-2',
            'preferences': [[2.0, 2.0, 2.0]]  # Equal preferences across states
        }
    ],
    
    # Analysis parameters
    'n_iterations': 100,
    'metrics': [
        'homeostasis_satisfaction',
        'expected_free_energy', 
        'belief_accuracy',
        'control_efficiency'
    ],
    'log_level': logging.INFO,
    'random_seed': 42
}

# Configure logging
logging.basicConfig(
    level=SHAPLEY_CONFIG['log_level'],
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add custom JSON encoder for frozenset
class ShapleyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Shapley analysis results"""
    def default(self, obj):
        if isinstance(obj, frozenset):
            return f"coalition_{'_'.join(map(str, sorted(obj)))}"
        return super().default(obj)

class BiofirmShapleyAnalysis:
    """Analyze Biofirm agents using Shapley value calculations"""
    
    def __init__(self, 
                 agent_paths: List[Path],
                 output_dir: Optional[Path] = None,
                 metrics: Optional[List[str]] = None,
                 n_iterations: int = 100):
        """
        Initialize Shapley analysis for Biofirm agents
        
        Parameters
        ----------
        agent_paths : List[Path]
            Paths to agent GNN model files
        output_dir : Optional[Path]
            Base output directory (default: creates Shapley/ directory)
        metrics : Optional[List[str]]
            Metrics to calculate Shapley values for (default: all available metrics)
        n_iterations : int
            Number of simulation iterations per coalition (default: 100)
        """
        try:
            self.agent_paths = [Path(p) for p in agent_paths]
            self.n_agents = len(agent_paths)
            self.n_iterations = n_iterations
            
            # Validate agent paths
            for path in self.agent_paths:
                if not path.exists():
                    raise FileNotFoundError(f"Agent model not found: {path}")
            
            # Setup output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_dir is None:
                output_dir = SHAPLEY_CONFIG['output_base']
            self.output_dir = Path(output_dir) / f"shapley_experiment_{timestamp}"
            
            # Create complete directory structure
            self.directories = {
                'root': self.output_dir,
                'agents': self.output_dir / "1_agents",
                'coalitions': self.output_dir / "2_coalitions",
                'analysis': self.output_dir / "3_analysis",
                'results': self.output_dir / "3_analysis" / "results",
                'plots': self.output_dir / "3_analysis" / "plots",
                'metrics': self.output_dir / "3_analysis" / "metrics",
                'logs': self.output_dir / "logs"
            }
            
            # Create all directories
            for dir_path in self.directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                
            # Setup logging
            self._setup_logging()
            
            # Set default metrics if none provided
            self.metrics = metrics if metrics else SHAPLEY_CONFIG['metrics']
            
            logger.info(f"Initialized Shapley analysis for {self.n_agents} agents")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Metrics: {self.metrics}")
            
        except Exception as e:
            logger.error(f"Error initializing Shapley analysis: {str(e)}")
            raise

    def _setup_logging(self):
        """Configure logging for analysis"""
        log_file = self.directories['logs'] / f"shapley_analysis.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)7s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(file_handler)

    def run_analysis(self):
        """Execute complete Shapley value analysis"""
        try:
            analysis_start = time.time()
            logger.info("\nStarting Shapley value analysis...")
            
            # Generate all possible coalitions
            coalitions = self._generate_coalitions()
            logger.info(f"Generated {len(coalitions)} coalitions")
            
            # Run simulations for each coalition
            coalition_values = self._evaluate_coalitions(coalitions)
            
            # Calculate Shapley values
            shapley_values = self._calculate_shapley_values(coalition_values)
            
            # Save results
            self._save_results(shapley_values, coalition_values)
            
            analysis_time = time.time() - analysis_start
            logger.info(f"\nAnalysis completed in {analysis_time:.2f}s")
            
            return shapley_values
            
        except Exception as e:
            logger.error(f"Error during Shapley analysis: {str(e)}")
            raise

    def _generate_coalitions(self) -> List[Set[int]]:
        """Generate all possible agent coalitions"""
        agents = set(range(self.n_agents))
        coalitions = []
        
        # Generate all possible combinations (2^n - 1 coalitions)
        for r in range(1, self.n_agents + 1):
            for coalition in itertools.combinations(agents, r):
                coalitions.append(frozenset(coalition))
                
        logger.info(f"Generated {len(coalitions)} coalitions")
        
        # Log coalition structure
        for coalition in coalitions:
            agent_names = [SHAPLEY_CONFIG['agent_variations'][i]['name'] 
                          for i in coalition]
            logger.debug(f"Coalition: {' + '.join(agent_names)}")
            
        return coalitions

    def _evaluate_coalitions(self, coalitions: List[Set[int]]) -> Dict[str, Dict[frozenset, float]]:
        """Evaluate performance metrics for each coalition"""
        coalition_values = {metric: {} for metric in self.metrics}
        
        for coalition in coalitions:
            logger.info(f"\nEvaluating coalition: {coalition}")
            
            # Get agent paths for this coalition
            coalition_agents = [self.agent_paths[i] for i in coalition]
            
            # Run simulations and collect metrics
            metrics = self._run_coalition_simulations(coalition_agents)
            
            # Store results
            coalition_key = frozenset(coalition)
            for metric in self.metrics:
                coalition_values[metric][coalition_key] = metrics[metric]
                
            logger.info(f"Coalition {coalition} metrics:")
            for metric, value in metrics.items():
                logger.info(f"- {metric}: {value:.4f}")
                
        return coalition_values

    def _run_coalition_simulations(self, agent_paths: List[Path]) -> Dict[str, float]:
        """Run simulations for a coalition of agents and compute their collective performance"""
        try:
            # Create unique name for this coalition
            coalition_name = f"coalition_{len(agent_paths)}_{''.join(str(i) for i in range(len(agent_paths)))}"
            coalition_dir = self.directories['coalitions'] / coalition_name
            coalition_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure experiment with correct paths and output directory
            config = EXPERIMENT_CONFIG.copy()
            config.update({
                'name': coalition_name,
                'n_iterations': self.n_iterations,
                'agent_paths': [str(p) for p in agent_paths],
                'env_gnn': str(SHAPLEY_CONFIG['env_gnn']),
                'output_base': str(coalition_dir)  # Set output base to coalition directory
            })
            
            # Run experiment
            experiment = BiofirmExperiment(
                name=coalition_name,
                config=config
            )
            experiment.run_experiment()
            
            # Get metrics from simulation results
            metrics_file = Path(experiment.output_dir) / "2_simulation" / "metrics" / "performance_metrics.json"
            
            if not metrics_file.exists():
                raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
                
            with open(metrics_file) as f:
                results = json.load(f)
                
            # Calculate Shapley-relevant metrics
            metrics = {}
            metrics['homeostasis_satisfaction'] = results['homeostatic_metrics']['time_in_homeo']
            metrics['belief_accuracy'] = results['belief_metrics']['accuracy']
            metrics['control_efficiency'] = results['homeostatic_metrics']['control_efficiency']
            metrics['expected_free_energy'] = -np.mean(results['belief_metrics']['entropy'])
            
            # Log coalition performance with clear interpretation
            logger.info(f"\nCoalition Performance Summary ({coalition_name}):")
            logger.info("-" * 40)
            logger.info(f"Agents: {[p.stem for p in agent_paths]}")
            logger.info("Metrics:")
            logger.info(f"- Homeostasis: {metrics['homeostasis_satisfaction']:.2%} time in target range")
            logger.info(f"- Belief Accuracy: {metrics['belief_accuracy']:.2%} correct state estimation")
            logger.info(f"- Control Efficiency: {metrics['control_efficiency']:.2f} optimal actions")
            logger.info(f"- Expected Free Energy: {metrics['expected_free_energy']:.2f} (lower is better)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running coalition simulations: {str(e)}")
            raise

    def _calculate_shapley_values(self, 
                                coalition_values: Dict[str, Dict[frozenset, float]]
                                ) -> Dict[str, Dict[int, float]]:
        """Calculate Shapley values for each agent and metric"""
        shapley_values = {metric: {} for metric in self.metrics}
        
        for metric in self.metrics:
            values = coalition_values[metric]
            
            for i in range(self.n_agents):
                shapley_sum = 0.0
                n = self.n_agents
                
                # Calculate marginal contributions
                for coalition in [c for c in values.keys() if i not in c]:
                    coalition_with_i = frozenset(list(coalition) + [i])
                    marginal = values.get(coalition_with_i, 0) - values.get(coalition, 0)
                    
                    # Calculate Shapley weight
                    weight = (len(coalition) * np.math.factorial(len(coalition)) * 
                            np.math.factorial(n - len(coalition) - 1) / np.math.factorial(n))
                    
                    shapley_sum += weight * marginal
                    
                shapley_values[metric][i] = shapley_sum
                
        return shapley_values

    def _save_results(self, 
                     shapley_values: Dict[str, Dict[int, float]], 
                     coalition_values: Dict[str, Dict[frozenset, float]]):
        """Save comprehensive Shapley analysis results with visualizations"""
        try:
            # 1. Process coalition values for serialization
            processed_coalition_values = {
                metric: {
                    f"coalition_{'_'.join(map(str, sorted(c)))}": value 
                    for c, value in values.items()
                }
                for metric, values in coalition_values.items()
            }
            
            # 2. Create results dictionary with complete structure
            results = {
                'shapley_values': {
                    metric: {
                        'values': {str(agent): value for agent, value in values.items()},
                        'interpretation': self._interpret_shapley_values(metric, values),
                        'ranking': self._rank_agents(values)
                    }
                    for metric, values in shapley_values.items()
                },
                'coalition_performances': processed_coalition_values,
                'summary_statistics': self._calculate_summary_statistics(shapley_values, coalition_values),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'n_agents': self.n_agents,
                    'n_iterations': self.n_iterations,
                    'agent_types': [v['name'] for v in SHAPLEY_CONFIG['agent_variations']]
                }
            }
            
            # Save detailed results
            results_file = self.directories['results'] / "shapley_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=ShapleyJSONEncoder)
                
            # Generate visualizations
            self._plot_shapley_results(shapley_values)
            self._plot_coalition_performance(coalition_values)
            self._plot_agent_synergies(coalition_values)
            self._plot_metric_correlations(shapley_values)
            
            # Generate summary report
            self._generate_analysis_report(results, coalition_values)
            
            logger.info("\nShapley Analysis Results:")
            for metric, metric_results in results['shapley_values'].items():
                logger.info(f"\n{metric}:")
                logger.info(metric_results['interpretation'])
                
            # Log detailed Shapley analysis
            logger.info("\n" + "="*80)
            logger.info("SHAPLEY ANALYSIS RESULTS")
            logger.info("="*80 + "\n")
            
            # 1. Individual Agent Contributions
            logger.info("Individual Agent Contributions:")
            logger.info("-"*50)
            for metric in shapley_values:
                logger.info(f"\n{metric.upper()}:")
                ranked_agents = sorted(
                    shapley_values[metric].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                for rank, (agent_id, value) in enumerate(ranked_agents, 1):
                    agent_name = SHAPLEY_CONFIG['agent_variations'][agent_id]['name']
                    relative = abs(value) / sum(abs(v) for v in shapley_values[metric].values()) * 100
                    logger.info(f"Rank {rank}: {agent_name:12} | Value: {value:+.4f} | Relative Impact: {relative:5.1f}%")
            
            # 2. Coalition Performance
            logger.info("\nCoalition Performance:")
            logger.info("-"*50)
            for metric in coalition_values:
                logger.info(f"\n{metric.upper()}:")
                sorted_coalitions = sorted(
                    coalition_values[metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                logger.info("\nTop 3 Coalitions:")
                for coalition, value in sorted_coalitions[:3]:
                    agents = [SHAPLEY_CONFIG['agent_variations'][i]['name'] for i in coalition]
                    logger.info(f"- {' + '.join(agents):40} | Performance: {value:.4f}")
                
                logger.info("\nWorst 3 Coalitions:")
                for coalition, value in sorted_coalitions[-3:]:
                    agents = [SHAPLEY_CONFIG['agent_variations'][i]['name'] for i in coalition]
                    logger.info(f"- {' + '.join(agents):40} | Performance: {value:.4f}")
            
            # 3. Synergy Analysis
            logger.info("\nSynergy Analysis:")
            logger.info("-"*50)
            for metric in self.metrics:
                synergy_scores = self._calculate_synergy_scores(coalition_values)[metric]
                logger.info(f"\n{metric.upper()} Synergies:")
                
                # Log strongest positive synergies
                pos_pairs = []
                neg_pairs = []
                for i in range(self.n_agents):
                    for j in range(i+1, self.n_agents):
                        score = synergy_scores[i,j]
                        pair = (
                            SHAPLEY_CONFIG['agent_variations'][i]['name'],
                            SHAPLEY_CONFIG['agent_variations'][j]['name'],
                            score
                        )
                        if score > 0:
                            pos_pairs.append(pair)
                        else:
                            neg_pairs.append(pair)
                
                pos_pairs.sort(key=lambda x: x[2], reverse=True)
                neg_pairs.sort(key=lambda x: x[2])
                
                logger.info("\nStrongest Positive Synergies:")
                for a1, a2, score in pos_pairs[:3]:
                    logger.info(f"- {a1} + {a2:12} | Synergy: +{score:.4f}")
                    
                logger.info("\nStrongest Negative Synergies:")
                for a1, a2, score in neg_pairs[:3]:
                    logger.info(f"- {a1} + {a2:12} | Synergy: {score:.4f}")
            
            # 4. Generate and save all visualizations
            logger.info("\nGenerating Visualizations...")
            logger.info("-"*50)
            
            # Shapley value plots
            self._plot_shapley_results(shapley_values)
            logger.info("✓ Generated Shapley value plots")
            
            # Coalition performance plots
            self._plot_coalition_performance(coalition_values)
            logger.info("✓ Generated coalition performance plots")
            
            # Synergy matrices
            self._plot_agent_synergies(coalition_values)
            logger.info("✓ Generated synergy matrices")
            
            # Metric correlations
            self._plot_metric_correlations(shapley_values)
            logger.info("✓ Generated metric correlations")
            
            # Verify all plots were created
            expected_plots = [
                "shapley_analysis.png",
                "coalition_performance.png",
                *[f"synergy_matrix_{m}.png" for m in self.metrics],
                "metric_correlations.png"
            ]
            
            missing_plots = []
            for plot in expected_plots:
                if not (self.directories['plots'] / plot).exists():
                    missing_plots.append(plot)
            
            if missing_plots:
                logger.warning(f"Warning: Missing plots: {', '.join(missing_plots)}")
            else:
                logger.info("\n✓ All visualizations generated successfully")
                logger.info(f"Plots saved to: {self.directories['plots']}")
            
            # Generate comprehensive report
            self._generate_analysis_report(results, coalition_values)
            logger.info(f"✓ Generated analysis report: {self.directories['results']}/shapley_analysis_report.md")
            
            logger.info("\nAnalysis Summary:")
            logger.info("-"*50)
            for metric, metric_results in results['shapley_values'].items():
                logger.info(f"\n{metric}:")
                logger.info(metric_results['interpretation'])
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def _generate_analysis_report(self, results: Dict, coalition_values: Dict[str, Dict[frozenset, float]]):
        """Generate comprehensive Shapley analysis report"""
        report_path = self.directories['results'] / "shapley_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Biofirm Shapley Analysis Report\n\n")
            
            # Overall Summary
            f.write("## Overall Performance Summary\n")
            f.write("### Key Findings\n")
            for metric, stats in results['summary_statistics'].items():
                f.write(f"\n#### {metric.replace('_', ' ').title()}\n")
                
                # Access correct stats structure
                shapley_stats = stats['shapley_stats']
                coalition_stats = stats['coalition_stats']
                perf_range = stats['performance_range']
                
                f.write(f"- Mean Shapley Contribution: {shapley_stats['mean']:.4f}\n")
                f.write(f"- Mean Coalition Performance: {coalition_stats['mean']:.4f}\n")
                f.write(f"- Best Coalition: {' + '.join(str(i) for i in coalition_stats['best_coalition'][0])} ({coalition_stats['best_coalition'][1]:.4f})\n")
                f.write(f"- Performance Range: {perf_range['range']:.4f}\n")
                
                # Add strongest contributor
                max_contributor = shapley_stats['max_contributor']
                agent_name = SHAPLEY_CONFIG['agent_variations'][max_contributor[0]]['name']
                f.write(f"- Strongest Contributor: {agent_name} ({max_contributor[1]:.4f})\n")
            
            # Individual Agent Contributions
            f.write("\n## Individual Agent Contributions\n")
            for metric, metric_results in results['shapley_values'].items():
                f.write(f"\n### {metric.replace('_', ' ').title()}\n")
                f.write("| Agent | Shapley Value | Relative Contribution | Rank |\n")
                f.write("|-------|---------------|---------------------|------|\n")
                
                # Sort by absolute Shapley value
                ranked_agents = sorted(
                    metric_results['ranking'],
                    key=lambda x: abs(x['value']),
                    reverse=True
                )
                
                for agent_info in ranked_agents:
                    f.write(
                        f"| {agent_info['name']} | {agent_info['value']:.4f} | "
                        f"{agent_info['relative_contribution']:.1f}% | "
                        f"{agent_info['rank']} |\n"
                    )
            
            # Coalition Analysis
            f.write("\n## Coalition Analysis\n")
            for metric, stats in results['summary_statistics'].items():
                f.write(f"\n### {metric.replace('_', ' ').title()}\n")
                
                coalition_stats = stats['coalition_stats']
                f.write(f"- Average Performance: {coalition_stats['mean']:.4f}\n")
                f.write(f"- Variability (std): {coalition_stats['std']:.4f}\n")
                f.write(f"- Performance Range: {stats['performance_range']['range']:.4f}\n")
                
                # Best and Worst Coalitions
                f.write("\nTop Performing Coalitions:\n")
                sorted_coalitions = sorted(
                    coalition_values[metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for coalition, value in sorted_coalitions[:3]:
                    agents = [SHAPLEY_CONFIG['agent_variations'][i]['name'] 
                             for i in sorted(coalition)]
                    f.write(f"- {' + '.join(agents)}: {value:.4f}\n")
                    
                # Add worst performing coalitions
                f.write("\nWorst Performing Coalitions:\n")
                for coalition, value in sorted_coalitions[-3:]:
                    agents = [SHAPLEY_CONFIG['agent_variations'][i]['name'] 
                             for i in sorted(coalition)]
                    f.write(f"- {' + '.join(agents)}: {value:.4f}\n")
            
            # Synergy Analysis
            f.write("\n## Synergy Analysis\n")
            for metric, stats in results['summary_statistics'].items():
                f.write(f"\n### {metric.replace('_', ' ').title()}\n")
                
                synergy = stats['synergy_analysis']
                
                # Best Synergies
                f.write("\nMost Synergistic Pairs:\n")
                for pair, info in synergy['best_pairs']:
                    f.write(
                        f"- {pair}: {info['synergy_score']:.4f} "
                        f"(Joint: {info['joint_performance']:.4f}, "
                        f"Individual: {info['individual_performances'][0]:.4f}, "
                        f"{info['individual_performances'][1]:.4f})\n"
                    )
                
                # Worst Synergies
                f.write("\nMost Antagonistic Pairs:\n")
                for pair, info in synergy['worst_pairs']:
                    f.write(
                        f"- {pair}: {info['synergy_score']:.4f} "
                        f"(Joint: {info['joint_performance']:.4f}, "
                        f"Individual: {info['individual_performances'][0]:.4f}, "
                        f"{info['individual_performances'][1]:.4f})\n"
                    )
            
            # Methodology
            f.write("\n## Methodology\n")
            f.write("\n### Shapley Value Calculation\n")
            f.write("Shapley values were calculated using the formula:\n")
            f.write("```\nφᵢ(v) = Σ [|S|!*(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]\n```\n")
            f.write("Where:\n")
            f.write("- S: Coalition subset not containing agent i\n")
            f.write("- v(S): Performance value of coalition S\n")
            f.write("- n: Total number of agents\n")
            
        logger.info(f"Generated comprehensive analysis report: {report_path}")

    def _interpret_shapley_values(self, metric: str, values: Dict[int, float]) -> str:
        """Interpret Shapley values for a given metric"""
        # Sort agents by contribution
        sorted_agents = sorted(values.items(), key=lambda x: x[1], reverse=True)
        
        # Generate interpretation
        if metric == 'homeostasis_satisfaction':
            interpretation = "Contribution to maintaining homeostasis:\n"
        elif metric == 'expected_free_energy':
            interpretation = "Contribution to minimizing uncertainty:\n"
        elif metric == 'belief_accuracy':
            interpretation = "Contribution to accurate state estimation:\n"
        else:
            interpretation = "Contribution to control efficiency:\n"
            
        # Add ranked contributions
        for agent, value in sorted_agents:
            agent_name = SHAPLEY_CONFIG['agent_variations'][agent]['name']
            interpretation += f"- Agent {agent} ({agent_name}): {value:.4f}\n"
            
        return interpretation

    def _plot_shapley_results(self, shapley_values: Dict[str, Dict[int, float]]):
        """Enhanced visualization of Shapley analysis results for 5 agents"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            settings = get_plot_settings()
            
            # Create figure with adjusted size for 5 agents
            fig, axes = plt.subplots(2, 2, figsize=settings['figsize']['shapley'])
            fig.suptitle('Shapley Value Analysis: Agent Contributions', 
                        fontsize=settings['font']['title'], 
                        y=1.02)
            
            for (metric, values), ax in zip(shapley_values.items(), axes.flat):
                # Get agent names and values
                agent_names = [SHAPLEY_CONFIG['agent_variations'][i]['name'] 
                             for i in range(self.n_agents)]
                agent_values = list(values.values())
                
                # Create bar plot with custom styling
                bars = ax.bar(
                    range(len(agent_names)),
                    agent_values,
                    color=settings['colors']['agents'],
                    width=0.7
                )
                
                # Add value labels with conditional positioning
                for bar in bars:
                    height = bar.get_height()
                    va = 'bottom' if height >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va=va,
                           fontsize=settings['font']['annotation'])
                
                # Customize plot
                ax.set_title(metric.replace('_', ' ').title(),
                            fontsize=settings['font']['title'])
                ax.set_xlabel('Agent Type', fontsize=settings['font']['label'])
                ax.set_ylabel('Shapley Value', fontsize=settings['font']['label'])
                
                # Adjust tick labels for better readability with 5 agents
                ax.set_xticks(range(len(agent_names)))
                ax.set_xticklabels(agent_names, 
                                 rotation=45, 
                                 ha='right',
                                 fontsize=settings['font']['tick'])
                
                # Add grid and adjust limits
                ax.grid(True, linestyle='--', alpha=0.7)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(min(ymin * 1.1, 0), max(ymax * 1.1, 0))
                
            plt.tight_layout()
            
            # Save plot with high resolution
            plot_file = self.directories['plots'] / "shapley_analysis.png"
            plt.savefig(plot_file, 
                       dpi=settings['dpi'],
                       bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved Shapley analysis plot to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error creating Shapley visualizations: {str(e)}")

    def _rank_agents(self, values: Dict[int, float]) -> List[Dict]:
        """Rank agents by their Shapley values"""
        # Sort agents by absolute contribution
        ranked_agents = sorted(
            values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Create ranked list with all required information
        ranked_list = []
        for rank, (agent_id, value) in enumerate(ranked_agents, 1):
            ranked_list.append({
                'agent_id': agent_id,
                'name': SHAPLEY_CONFIG['agent_variations'][agent_id]['name'],
                'value': value,
                'relative_contribution': abs(value) / sum(abs(v) for v in values.values()) * 100,
                'rank': rank  # Add explicit rank
            })
        
        return ranked_list

    def _calculate_summary_statistics(self, 
                                   shapley_values: Dict[str, Dict[int, float]], 
                                   coalition_values: Dict[str, Dict[frozenset, float]]) -> Dict:
        """Calculate comprehensive summary statistics for Shapley analysis"""
        stats = {}
        
        for metric in self.metrics:
            metric_stats = {
                'shapley_stats': {
                    'mean': np.mean(list(shapley_values[metric].values())),
                    'std': np.std(list(shapley_values[metric].values())),
                    'max_contributor': max(shapley_values[metric].items(), key=lambda x: x[1]),
                    'min_contributor': min(shapley_values[metric].items(), key=lambda x: x[1])
                },
                'coalition_stats': {
                    'mean': np.mean(list(coalition_values[metric].values())),
                    'std': np.std(list(coalition_values[metric].values())),
                    'best_coalition': max(coalition_values[metric].items(), key=lambda x: x[1]),
                    'worst_coalition': min(coalition_values[metric].items(), key=lambda x: x[1])
                },
                'synergy_analysis': self._analyze_synergies(coalition_values[metric]),
                'performance_range': {
                    'min': min(coalition_values[metric].values()),
                    'max': max(coalition_values[metric].values()),
                    'range': max(coalition_values[metric].values()) - min(coalition_values[metric].values())
                }
            }
            stats[metric] = metric_stats
            
        return stats

    def _analyze_synergies(self, coalition_values: Dict[frozenset, float]) -> Dict:
        """Analyze synergistic and antagonistic effects between agents for N=5"""
        synergy_analysis = {
            'pairwise_synergies': {},
            'best_pairs': [],
            'worst_pairs': [],
            'triplet_synergies': {},
            'quartet_synergies': {},  # Added for N=5 analysis
            'synergy_scores': {}
        }
        
        # Analyze pairs (same as before)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                i_perf = coalition_values.get(frozenset([i]), 0)
                j_perf = coalition_values.get(frozenset([j]), 0)
                joint_perf = coalition_values.get(frozenset([i, j]), 0)
                
                synergy = joint_perf - (i_perf + j_perf)
                
                pair_key = (
                    f"{SHAPLEY_CONFIG['agent_variations'][i]['name']}-"
                    f"{SHAPLEY_CONFIG['agent_variations'][j]['name']}"
                )
                synergy_analysis['pairwise_synergies'][pair_key] = {
                    'synergy_score': synergy,
                    'joint_performance': joint_perf,
                    'individual_performances': (i_perf, j_perf)
                }
        
        # Analyze triplets (enhanced for N=5)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                for k in range(j + 1, self.n_agents):
                    perfs = [coalition_values.get(frozenset([x]), 0) for x in (i, j, k)]
                    pair_perfs = [
                        coalition_values.get(frozenset([i, j]), 0),
                        coalition_values.get(frozenset([j, k]), 0),
                        coalition_values.get(frozenset([i, k]), 0)
                    ]
                    triplet_perf = coalition_values.get(frozenset([i, j, k]), 0)
                    
                    # Enhanced synergy calculation
                    triplet_synergy = (
                        triplet_perf - 
                        sum(pair_perfs) + 
                        2 * sum(perfs)
                    )
                    
                    triplet_key = "-".join([
                        SHAPLEY_CONFIG['agent_variations'][x]['name']
                        for x in (i, j, k)
                    ])
                    
                    synergy_analysis['triplet_synergies'][triplet_key] = {
                        'synergy_score': triplet_synergy,
                        'joint_performance': triplet_perf,
                        'pair_performances': pair_perfs,
                        'individual_performances': perfs
                    }
        
        # Add quartet analysis (new for N=5)
        for combo in itertools.combinations(range(self.n_agents), 4):
            i, j, k, l = combo
            perfs = [coalition_values.get(frozenset([x]), 0) for x in combo]
            
            # Get all possible triplet performances
            triplet_perfs = [
                coalition_values.get(frozenset(t), 0)
                for t in itertools.combinations(combo, 3)
            ]
            
            # Get all possible pair performances
            pair_perfs = [
                coalition_values.get(frozenset(p), 0)
                for p in itertools.combinations(combo, 2)
            ]
            
            quartet_perf = coalition_values.get(frozenset(combo), 0)
            
            # Calculate higher-order synergy for quartets
            quartet_synergy = (
                quartet_perf -
                sum(triplet_perfs) +
                2 * sum(pair_perfs) -
                3 * sum(perfs)
            )
            
            quartet_key = "-".join([
                SHAPLEY_CONFIG['agent_variations'][x]['name']
                for x in combo
            ])
            
            synergy_analysis['quartet_synergies'][quartet_key] = {
                'synergy_score': quartet_synergy,
                'joint_performance': quartet_perf,
                'triplet_performances': triplet_perfs,
                'pair_performances': pair_perfs,
                'individual_performances': perfs
            }
        
        # Sort and store best/worst combinations
        for category in ['pairwise_synergies', 'triplet_synergies', 'quartet_synergies']:
            sorted_combos = sorted(
                synergy_analysis[category].items(),
                key=lambda x: x[1]['synergy_score'],
                reverse=True
            )
            
            synergy_analysis[f'best_{category}'] = sorted_combos[:3]
            synergy_analysis[f'worst_{category}'] = sorted_combos[-3:]
        
        return synergy_analysis

    def _calculate_synergy_scores(self, coalition_values: Dict[str, Dict[frozenset, float]]) -> Dict[str, np.ndarray]:
        """Calculate synergy score matrix for each metric"""
        synergy_scores = {}
        
        for metric, values in coalition_values.items():
            scores = np.zeros((self.n_agents, self.n_agents))
            
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        # Individual performances
                        i_perf = values.get(frozenset([i]), 0)
                        j_perf = values.get(frozenset([j]), 0)
                        
                        # Joint performance
                        joint_perf = values.get(frozenset([i, j]), 0)
                        
                        # Synergy score
                        scores[i, j] = joint_perf - (i_perf + j_perf)
                    
            synergy_scores[metric] = scores
            
        return synergy_scores

    def _plot_coalition_performance(self, coalition_values: Dict[str, Dict[frozenset, float]]):
        """Enhanced visualization of coalition performance for 4 agents"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            settings = get_plot_settings()
            
            # Create larger figure for more coalitions
            fig, axes = plt.subplots(2, 2, figsize=settings['figsize']['coalition'])
            fig.suptitle('Coalition Performance Analysis', 
                        fontsize=settings['font']['title'], 
                        y=1.02)
            
            for (metric, values), ax in zip(coalition_values.items(), axes.flat):
                # Process coalitions
                coalition_data = []
                for coalition, value in values.items():
                    size = len(coalition)
                    agents = [SHAPLEY_CONFIG['agent_variations'][i]['name'][:3] 
                             for i in sorted(coalition)]
                    coalition_data.append({
                        'size': size,
                        'label': '+'.join(agents),
                        'value': value
                    })
                
                # Sort by size then value
                coalition_data.sort(key=lambda x: (x['size'], x['value']), reverse=True)
                
                # Create color palette based on coalition sizes
                unique_sizes = len(set(d['size'] for d in coalition_data))
                colors = settings['colors']['coalitions'](np.linspace(0, 1, unique_sizes))
                size_to_color = {s: c for s, c in zip(range(1, unique_sizes + 1), colors)}
                
                # Create bars
                bars = ax.bar(
                    range(len(coalition_data)),
                    [d['value'] for d in coalition_data],
                    color=[size_to_color[d['size']] for d in coalition_data]
                )
                
                # Add value labels
                for bar, data in zip(bars, coalition_data):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom',
                           fontsize=settings['font']['annotation'])
                
                # Customize plot
                ax.set_title(metric.replace('_', ' ').title(),
                            fontsize=settings['font']['title'])
                ax.set_xlabel('Agent Coalition',
                            fontsize=settings['font']['label'])
                ax.set_ylabel('Performance',
                            fontsize=settings['font']['label'])
                
                # Set tick labels
                ax.set_xticks(range(len(coalition_data)))
                ax.set_xticklabels([d['label'] for d in coalition_data],
                                  rotation=45,
                                  ha='right',
                                  fontsize=settings['font']['tick'])
                
                # Add size legend
                legend_elements = [plt.Rectangle((0,0),1,1, 
                                              facecolor=size_to_color[s],
                                              label=f'{s} Agents')
                                 for s in range(1, unique_sizes + 1)]
                ax.legend(handles=legend_elements,
                         loc='upper right',
                         fontsize=settings['font']['annotation'])
                
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save plot
            plot_file = self.directories['plots'] / "coalition_performance.png"
            plt.savefig(plot_file,
                       dpi=settings['dpi'],
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting coalition performance: {str(e)}")

    def _plot_agent_synergies(self, coalition_values: Dict[str, Dict[frozenset, float]]):
        """Enhanced visualization of agent synergies for 4 agents"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            settings = get_plot_settings()
            
            # Calculate synergy scores
            synergy_scores = self._calculate_synergy_scores(coalition_values)
            
            for metric, scores in synergy_scores.items():
                plt.figure(figsize=settings['figsize']['synergy'])
                
                # Create mask for diagonal
                mask = np.eye(len(scores), dtype=bool)
                
                # Get agent labels
                agent_labels = [v['name'] for v in SHAPLEY_CONFIG['agent_variations']]
                
                # Plot heatmap with enhanced styling
                sns.heatmap(
                    scores,
                    annot=True,
                    fmt='.3f',
                    cmap=settings['colors']['synergy'],
                    center=0,
                    mask=mask,
                    xticklabels=agent_labels,
                    yticklabels=agent_labels,
                    annot_kws={'size': settings['font']['annotation']},
                    cbar_kws={'label': 'Synergy Score'}
                )
                
                plt.title(f'Agent Synergies: {metric.replace("_", " ").title()}',
                         fontsize=settings['font']['title'],
                         pad=20)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                
                plt.tight_layout()
                
                # Save plot
                plot_file = self.directories['plots'] / f"synergy_matrix_{metric}.png"
                plt.savefig(plot_file,
                           dpi=settings['dpi'],
                           bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting agent synergies: {str(e)}")

    def _plot_metric_correlations(self, shapley_values: Dict[str, Dict[int, float]]):
        """Visualize correlations between different performance metrics"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Create correlation matrix
            metric_data = {}
            for metric, values in shapley_values.items():
                metric_data[metric] = list(values.values())
                
            corr_df = pd.DataFrame(metric_data)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                corr_df.corr(),
                annot=True,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1
            )
            
            plt.title('Metric Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.directories['plots'] / "metric_correlations.png",
                       dpi=SHAPLEY_CONFIG['plot_settings']['dpi'])
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting metric correlations: {str(e)}")

def main():
    """Main execution function"""
    try:
        # Set random seed
        np.random.seed(SHAPLEY_CONFIG['random_seed'])
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = SHAPLEY_CONFIG['output_base'] / f"shapley_experiment_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        agents_dir = experiment_dir / "1_agents"
        coalitions_dir = experiment_dir / "2_coalitions"
        analysis_dir = experiment_dir / "3_analysis"
        logs_dir = experiment_dir / "logs"
        
        for dir_path in [agents_dir, coalitions_dir, analysis_dir, logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load base agent config
        base_agent = SHAPLEY_CONFIG['base_agent']
        if not base_agent.exists():
            raise FileNotFoundError(f"Base agent model not found: {base_agent}")
            
        with open(base_agent) as f:
            base_config = json.load(f)
        
        # Create agent variations in agents directory
        agent_paths = []
        for variation in SHAPLEY_CONFIG['agent_variations']:
            # Create new config with modified preferences
            agent_config = base_config.copy()
            agent_config['preferences']['values'] = variation['preferences']
            
            # Save to agents directory
            agent_path = agents_dir / f"gnn_agent_biofirm_{variation['name']}.gnn"
            with open(agent_path, 'w') as f:
                json.dump(agent_config, f, indent=4)
            agent_paths.append(agent_path)
            
            logger.info(f"Created agent variation: {agent_path}")
        
        # Create analyzer with experiment directory structure
        analyzer = BiofirmShapleyAnalysis(
            agent_paths=agent_paths,
            output_dir=experiment_dir,
            metrics=SHAPLEY_CONFIG['metrics'],
            n_iterations=SHAPLEY_CONFIG['n_iterations']
        )
        
        # Run analysis
        shapley_values = analyzer.run_analysis()
        
        # Print summary
        print("\nShapley Value Analysis Summary:")
        print("=" * 40)
        for metric in shapley_values:
            print(f"\n{metric}:")
            for agent, value in shapley_values[metric].items():
                agent_name = SHAPLEY_CONFIG['agent_variations'][agent]['name']
                print(f"Agent {agent} ({agent_name}): {value:.4f}")
                
        # Save experiment config
        config_file = experiment_dir / "shapley_config.json"
        with open(config_file, 'w') as f:
            config_dict = SHAPLEY_CONFIG.copy()
            config_dict['base_agent'] = str(config_dict['base_agent'])
            config_dict['env_gnn'] = str(config_dict['env_gnn'])
            config_dict['output_base'] = str(config_dict['output_base'])
            json.dump(config_dict, f, indent=4)
            
        logger.info(f"\nExperiment completed. Results saved to: {experiment_dir}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
