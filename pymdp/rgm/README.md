# RGM MNIST Pipeline

This directory contains the Recursive Generative Models (RGM) implementation focused on MNIST digit recognition and generation.

## Overview

The RGM pipeline implements a hierarchical generative model that learns to:
1. Recognize MNIST digits through inference
2. Generate digit samples through top-down processing
3. Learn hierarchical representations through matrix factorization

## Directory Structure

```
rgm/
├── __init__.py              # Package initialization
├── Run_RGM.py              # Main pipeline runner
├── rgm_render.py           # Matrix generation and visualization
├── rgm_execute.py          # Inference engine
├── rgm_analyze.py          # Results analysis
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── rgm_experiment_utils.py          # Experiment management
│   ├── rgm_matrix_visualization_utils.py # Visualization tools
│   ├── rgm_pipeline_manager.py          # Pipeline orchestration
│   └── rgm_config_loader.py             # Configuration handling
├── configs/                # Configuration files
│   ├── default_config.json    # Default parameters
│   └── mnist_config.json      # MNIST-specific settings
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── rgm_base.py           # Base RGM implementation
│   └── rgm_mnist.py          # MNIST-specific model
└── experiments/            # Generated experiment data
    └── README.md             # Experiment documentation
```

## Components

### 1. Pipeline Runner (`Run_RGM.py`)
- Main entry point for RGM pipeline execution
- Handles experiment initialization and logging
- Orchestrates pipeline stages
- Manages error handling and checkpointing

### 2. Matrix Generation (`rgm_render.py`)
- Generates RGM matrices from GNN specifications
- Validates matrix properties
- Provides visualization utilities
- Handles matrix storage and versioning

### 3. Inference Engine (`rgm_execute.py`)
- Implements RGM inference algorithms
- Processes MNIST data
- Manages model state
- Handles checkpointing

### 4. Results Analysis (`rgm_analyze.py`)
- Analyzes inference results
- Generates visualizations
- Computes performance metrics
- Produces analysis reports

## Pipeline Stages

1. **Initialization**
   - Create experiment directory
   - Set up logging
   - Load configuration
   - Initialize components

2. **Matrix Generation**
   - Load GNN specifications
   - Generate matrices
   - Validate properties
   - Save and visualize

3. **MNIST Processing**
   - Load MNIST dataset
   - Preprocess images
   - Generate batches
   - Handle data augmentation

4. **Inference**
   - Initialize model state
   - Run inference
   - Save checkpoints
   - Monitor convergence

5. **Analysis**
   - Compute metrics
   - Generate visualizations
   - Save results
   - Create reports

## Usage

1. **Basic Usage**
   ```bash
   python3 Run_RGM.py
   ```

2. **Resume from Checkpoint**
   ```bash
   python3 Run_RGM.py /path/to/checkpoint
   ```

3. **Custom Configuration**
   ```bash
   python3 Run_RGM.py --config custom_config.json
   ```

## Configuration

The pipeline is configured through JSON files in the `configs/` directory:

1. `default_config.json`: Base configuration
2. `mnist_config.json`: MNIST-specific settings

Key configuration sections:
- Model parameters
- Training settings
- MNIST preprocessing
- Visualization options
- Logging configuration

## Experiment Results

Experiment results are stored in timestamped directories under `experiments/`:
```
experiments/
└── rgm_mnist_YYYYMMDD_HHMMSS/
    ├── config/          # Configuration snapshot
    ├── logs/           # Experiment logs
    ├── matrices/       # Generated matrices
    ├── checkpoints/    # Model checkpoints
    ├── visualizations/ # Generated plots
    └── results/        # Analysis results
```

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- PyTorch (for MNIST data handling)
- Logging
- Pathlib 