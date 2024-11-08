#!/usr/bin/env python3
"""
Simple script to run Biofirm experiment with default configuration
"""

from Run_Biofirm import BiofirmExperiment, DEFAULT_CONFIG
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Biofirm experiment')
    parser.add_argument('--config', type=str, help='Custom config file')
    args = parser.parse_args()
    
    # Create experiment
    experiment = BiofirmExperiment()
    
    # Run with default or custom config
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
        
    # Run experiment
    experiment.run_experiment(**config)

if __name__ == "__main__":
    main() 