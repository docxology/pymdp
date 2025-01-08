# PyMDP: Active Inference Framework

## Overview
PyMDP is a Python implementation of Active Inference, including Renormalization Generative Models (RGM) and other advanced inference methods.

## Installation

```bash
# Clone repository
git clone https://github.com/versesresearch/pymdp.git
cd pymdp

# Install package
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## RGM MNIST Example
The RGM implementation includes a complete example of MNIST digit classification:

```bash
# Run complete pipeline
cd pymdp/rgm
python Run_RGM.py

# Or run stages individually
python rgm_render.py
python rgm_execute.py
python rgm_analyze.py
```

## Project Structure
```
pymdp/
├── rgm/                    # RGM Implementation
│   ├── models/            # GNN Model Specifications
│   ├── utils/             # Utility Modules
│   ├── rgm_render.py      # Matrix Generation
│   ├── rgm_execute.py     # Model Execution
│   ├── rgm_analyze.py     # Analysis & Visualization
│   └── Run_RGM.py         # Pipeline Runner
├── tests/                 # Test Suite
├── docs/                  # Documentation
└── examples/              # Usage Examples
```

## Documentation
- [RGM Technical Specification](rgm/docs/RGM_GNN_Tech_Spec.md)
- [RGM MNIST Example](rgm/docs/RGM_MNIST.md)
- [GNN Framework](gnn/docs/README_GNN.md)

## Contributing
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details. 