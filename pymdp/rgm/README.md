# Renormalization Generative Model (RGM) Pipeline

## Overview
This repository implements the Renormalization Generative Model, a hierarchical generative model that leverages renormalization group principles for robust pattern recognition and generation.

## Architecture
The Renormalization Generative Model consists of:
- Hierarchical layers with bidirectional connections
- State and factor nodes at each level
- Message passing between levels
- Precision-weighted updates

## Components
- `rgm_model_state.py`: Manages Renormalization Generative Model state
- `rgm_message_utils.py`: Handles message passing between levels
- `rgm_matrix_normalizer.py`: Normalizes connection matrices

## Directory Structure

```
rgm/
├── __init__.py              # Package initialization
├── Run_RGM.py               # Main pipeline runner
├── rgm_render.py            # Matrix generation and visualization
├── rgm_execute.py           # Inference engine
├── rgm_analyze.py           # Results analysis
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── rgm_experiment_utils.py          # Experiment management
│   ├── rgm_matrix_visualization_utils.py # Visualization tools
│   ├── rgm_pipeline_manager.py          # Pipeline orchestration
│   └── rgm_config_loader.py             # Configuration handling
├── configs/                 # Configuration files
│   ├── default_config.json    # Default parameters
│   └── mnist_config.json      # MNIST-specific settings
├── models/                  # Model definitions
│   ├── __init__.py
│   ├── rgm_base.py           # Base RGM implementation
│   └── rgm_mnist.py          # MNIST-specific model
├── gnn_specs/               # GNN specification files
│   ├── rgm_base.gnn          # Base GNN specification
│   ├── rgm_mnist.gnn         # MNIST-specific GNN specification
│   ├── rgm_message_passing.gnn # Message passing GNN specification
│   └── rgm_hierarchical_level.gnn # Hierarchical level GNN specification
├── data/                    # Data storage
│   └── mnist/               # MNIST dataset
├── experiments/             # Experiment results
│   └── ...
├── README.md                # Pipeline documentation
└── requirements.txt         # Dependencies
```

## Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the pipeline by modifying the configuration files in the `configs/` directory.

3. Run the pipeline:
   ```
   python Run_RGM.py
   ```

4. Monitor the progress and logs during the pipeline execution.

5. Analyze the experiment results in the `experiments/` directory.

## Configuration

The pipeline can be configured using the JSON files in the `configs/` directory:
- `default_config.json`: Contains the default parameters for the RGM pipeline.
- `mnist_config.json`: Contains MNIST-specific settings and overrides.

## Experiment Results

The experiment results, including logs, generated matrices, model checkpoints, and visualizations, are stored in timestamped directories under `experiments/`.

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- tqdm 