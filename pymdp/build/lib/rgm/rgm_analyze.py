"""
RGM Analysis Module
==================

Analyzes RGM experiment results and generates comprehensive reports.

This module:
1. Loads and processes experiment results
2. Generates performance visualizations
3. Creates detailed experiment summaries
4. Validates model behavior
5. Exports analysis artifacts
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

from .utils.rgm_experiment_utils import RGMExperimentUtils
from .utils.rgm_metrics_utils import RGMMetricsUtils

class RGMAnalyzer:
    """Analyzes RGM experiment results and generates reports"""
    
    def __init__(self, results_dir: Path):
        """
        Initialize analyzer with experiment results.
        
        Args:
            results_dir: Path to experiment results directory
        """
        try:
            # Get experiment state
            self.experiment = RGMExperimentUtils.get_experiment()
            self.logger = RGMExperimentUtils.get_logger('analyzer')
            self.logger.info("Initializing RGM analyzer...")
            
            # Load results
            self.results = self._load_results(results_dir)
            self.metrics = RGMMetricsUtils()
            
            # Setup output directories
            self.output_dirs = {
                'analysis': self.experiment['dirs']['analysis'],
                'figures': self.experiment['dirs']['analysis'] / "figures",
                'reports': self.experiment['dirs']['analysis'] / "reports",
                'validation': self.experiment['dirs']['analysis'] / "validation"
            }
            
            # Create directories
            for path in self.output_dirs.values():
                path.mkdir(exist_ok=True)
                
            self.logger.info("RGM analyzer initialization complete")
            
        except Exception as e:
            logging.error(f"Failed to initialize analyzer: {str(e)}")
            raise
            
    def analyze_results(self) -> Path:
        """
        Run complete analysis of experiment results.
        
        Returns:
            Path to analysis directory
        """
        try:
            self.logger.info("Starting experiment analysis...")
            
            # Generate performance analysis
            self._analyze_performance()
            
            # Analyze learning dynamics
            self._analyze_learning()
            
            # Analyze belief states
            self._analyze_beliefs()
            
            # Generate experiment summary
            self._generate_summary()
            
            # Validate results
            self._validate_results()
            
            self.logger.info(f"Analysis complete. Results in: {self.output_dirs['analysis']}")
            return self.output_dirs['analysis']
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise
            
    def _analyze_performance(self):
        """Analyze model performance metrics"""
        try:
            self.logger.info("Analyzing model performance...")
            
            # Create performance figures directory
            perf_dir = self.output_dirs['figures'] / "performance"
            perf_dir.mkdir(exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                self.results['confusion_matrix'],
                annot=True,
                fmt='.2%',
                cmap='YlOrRd'
            )
            plt.title('Confusion Matrix')
            plt.savefig(perf_dir / "confusion_matrix.png")
            plt.close()
            
            # Plot class accuracies
            plt.figure(figsize=(10, 5))
            sns.barplot(
                x=range(10),
                y=self.results['class_accuracies']
            )
            plt.title('Per-Class Accuracy')
            plt.xlabel('Digit')
            plt.ylabel('Accuracy')
            plt.savefig(perf_dir / "class_accuracies.png")
            plt.close()
            
            # Generate performance report
            report = {
                'overall_accuracy': float(self.results['accuracy']),
                'class_accuracies': self.results['class_accuracies'].tolist(),
                'class_counts': self.results['class_counts'].tolist(),
                'confusion_matrix': self.results['confusion_matrix'].tolist()
            }
            
            report_path = self.output_dirs['reports'] / "performance_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            raise
            
    def _analyze_learning(self):
        """Analyze learning dynamics"""
        try:
            self.logger.info("Analyzing learning dynamics...")
            
            # Create learning figures directory
            learn_dir = self.output_dirs['figures'] / "learning"
            learn_dir.mkdir(exist_ok=True)
            
            # Plot learning curves
            metrics = ['elbo', 'accuracy', 'precision']
            for metric in metrics:
                history = self.results['learning'][f'{metric}_history']
                
                plt.figure(figsize=(10, 5))
                plt.plot(history)
                plt.title(f'{metric.capitalize()} History')
                plt.xlabel('Iteration')
                plt.ylabel(metric.capitalize())
                plt.savefig(learn_dir / f"{metric}_history.png")
                plt.close()
                
            # Analyze convergence
            convergence = {
                'final_elbo': float(self.results['learning']['elbo_history'][-1]),
                'final_accuracy': float(self.results['learning']['accuracy_history'][-1]),
                'final_precision': float(self.results['learning']['precision']),
                'convergence_iteration': self._find_convergence_point()
            }
            
            # Save convergence analysis
            conv_path = self.output_dirs['reports'] / "convergence_analysis.json"
            with open(conv_path, 'w') as f:
                json.dump(convergence, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error in learning analysis: {str(e)}")
            raise
            
    def _analyze_beliefs(self):
        """Analyze belief state dynamics"""
        try:
            self.logger.info("Analyzing belief states...")
            
            # Create beliefs figures directory
            belief_dir = self.output_dirs['figures'] / "beliefs"
            belief_dir.mkdir(exist_ok=True)
            
            # Analyze each level
            n_levels = len(self.results['beliefs']['states'])
            for level in range(n_levels):
                # Get level beliefs
                states = self.results['beliefs']['states'][level]
                factors = self.results['beliefs']['factors'][level]
                
                # Plot state distribution
                plt.figure(figsize=(10, 5))
                sns.histplot(states, bins=30)
                plt.title(f'Level {level} State Distribution')
                plt.savefig(belief_dir / f"level_{level}_state_dist.png")
                plt.close()
                
                # Plot factor distribution
                plt.figure(figsize=(10, 5))
                sns.histplot(factors, bins=30)
                plt.title(f'Level {level} Factor Distribution')
                plt.savefig(belief_dir / f"level_{level}_factor_dist.png")
                plt.close()
                
                # Compute belief statistics
                stats = {
                    'states': {
                        'mean': float(np.mean(states)),
                        'std': float(np.std(states)),
                        'entropy': float(self.metrics.compute_entropy(states))
                    },
                    'factors': {
                        'mean': float(np.mean(factors)),
                        'std': float(np.std(factors)),
                        'entropy': float(self.metrics.compute_entropy(factors))
                    }
                }
                
                # Save level statistics
                stats_path = self.output_dirs['reports'] / f"level_{level}_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error in belief analysis: {str(e)}")
            raise
            
    def _generate_summary(self):
        """Generate comprehensive experiment summary"""
        try:
            self.logger.info("Generating experiment summary...")
            
            summary = {
                'experiment': {
                    'name': self.experiment['name'],
                    'timestamp': self.experiment['timestamp'],
                    'duration': self._compute_duration()
                },
                'performance': {
                    'accuracy': float(self.results['accuracy']),
                    'elbo': float(self.results['elbo']),
                    'class_accuracies': self.results['class_accuracies'].tolist()
                },
                'learning': {
                    'convergence': self._analyze_convergence(),
                    'final_precision': float(self.results['learning']['precision'])
                },
                'model': {
                    'hierarchy': self._analyze_hierarchy(),
                    'complexity': self._compute_model_complexity()
                },
                'validation': {
                    'checks': self._run_validation_checks(),
                    'warnings': self._get_validation_warnings()
                }
            }
            
            # Save summary
            summary_path = self.output_dirs['reports'] / "experiment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Generate summary plots
            self._generate_summary_plots()
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise
            
    def _validate_results(self):
        """Run validation checks on results"""
        try:
            self.logger.info("Validating results...")
            
            # Define validation checks
            checks = {
                'belief_normalization': self._check_belief_normalization(),
                'performance_consistency': self._check_performance_consistency(),
                'learning_stability': self._check_learning_stability(),
                'numerical_stability': self._check_numerical_stability()
            }
            
            # Generate validation report
            report = {
                'timestamp': datetime.now().isoformat(),
                'checks': checks,
                'warnings': self._get_validation_warnings(),
                'recommendations': self._get_recommendations()
            }
            
            # Save validation report
            report_path = self.output_dirs['validation'] / "validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            # Log validation results
            self._log_validation_results(checks)
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise
