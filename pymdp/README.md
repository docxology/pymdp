# Renormalization Generative Model (RGM)

A PyTorch implementation of the Renormalization Generative Model for hierarchical pattern recognition and generation.

## Features

- Matrix-based hierarchical generative model
- Message passing inference
- MNIST training pipeline
- Visualization and analysis tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rgm.git
cd rgm
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Running the MNIST Pipeline

1. Configure the model in `rgm/config.json` (a default configuration is provided)

2. Run the pipeline:
```bash
cd rgm
python Run_RGM.py
```

The pipeline will:
1. Prepare MNIST data
2. Verify GNN specifications
3. Generate and visualize matrices
4. Train the model
5. Save results in the experiments directory

### Configuration

The model can be configured through `config.json` with the following sections:

```json
{
    "architecture": {
        "layer_dims": [784, 500, 200],
        "association_dims": []
    },
    "initialization": {
        "recognition": {
            "method": "xavier_uniform",
            "gain": 1.4
        },
        "generative": {
            "method": "xavier_uniform",
            "gain": 0.8
        },
        "lateral": {
            "method": "identity_with_noise",
            "noise_std": 0.01
        }
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5
    },
    "data": {
        "dataset": "MNIST",
        "validation_split": 0.1,
        "normalization": {
            "mean": [0.1307],
            "std": [0.3081]
        }
    }
}
```

### Project Structure

```
rgm/
├── __init__.py           # Package initialization
├── Run_RGM.py           # Main pipeline runner
├── rgm_render.py        # Matrix generation and visualization
├── models/              # GNN model specifications
├── utils/               # Utility modules
│   ├── rgm_logging.py   # Logging configuration
│   ├── rgm_data_loader.py  # Data loading utilities
│   └── rgm_config_validator.py  # Configuration validation
└── experiments/         # Generated experiment directories
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. Use black for formatting:

```bash
black rgm/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rgm2023,
    title={Renormalization Generative Model Implementation},
    author={RGM Team},
    year={2023},
    publisher={GitHub},
    howpublished={\url{https://github.com/yourusername/rgm}}
}
``` 