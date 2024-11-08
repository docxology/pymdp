# GNN (Generalized Notation Notation) System

## Overview
GNN is a domain-specific language and toolset for defining Active Inference generative models in PyMDP. It provides:

1. Structured JSON model definitions
2. Matrix generation for PyMDP
3. Visualization pipeline
4. Documentation generation
5. Experiment management

## Core Components

### 1. Model Definition (`.gnn` files)
```json
{
    "modelName": "example_model",
    "modelType": ["Dynamic", "POMDP"],
    "stateSpace": { ... },
    "observations": { ... },
    "matrices": { ... },
    "visualization": { ... },
    "markdown": { ... }
}
```

### 2. Matrix Generation
- A matrices (observation models)
- B matrices (transition dynamics)
- C matrices (preferences)
- D vectors (initial beliefs)
- E vectors (policy priors)

### 3. Visualization Pipeline
- Factor graphs
- Matrix heatmaps
- Belief evolution
- Learning curves
- LaTeX equations

### 4. Documentation
- Markdown generation
- Model summaries
- Equation rendering
- Implementation notes

## System Architecture

```
Model Definition → Matrix Generation → Visualization → Deployment
     (.gnn)      →  (PyMDP Format)   →   (Plots)    → (AgentMaker)
```

### Directory Structure
```
pymdp/
├── gnn/
│   ├── models/              # Model definitions
│   │   └── *.gnn           # GNN files
│   └── sandbox/            # Generated outputs
│       └── {model}_{timestamp}/
│           ├── matrices/   # Generated matrices
│           ├── viz/        # Visualizations
│           └── docs/       # Documentation
└── agentmaker/            # Deployment system
```

## Implementation Details

### 1. Model Validation
- JSON schema validation
- Matrix dimension checking
- Consistency verification
- Type checking

### 2. Matrix Generation
- Automatic normalization
- Dimension validation
- Sparse matrix support
- Default initialization

### 3. Visualization Engine
- NetworkX for graphs
- Matplotlib for plots
- LaTeX integration
- Interactive displays

### 4. Documentation Pipeline
- Markdown generation
- LaTeX equation rendering
- Automatic summaries
- Implementation notes

## Development Roadmap

### Current Features
1. Basic model definition
2. Matrix generation
3. Simple visualization
4. Basic documentation

### In Progress
1. Enhanced validation
2. Advanced visualization
3. Interactive editing
4. Real-time preview

### Future Plans
1. GUI interface
2. Version control
3. Model composition
4. Template library

## Best Practices

### 1. Model Definition
- Use descriptive names
- Include documentation
- Validate matrices
- Test thoroughly

### 2. Visualization
- Clear layouts
- Consistent styling
- Informative labels
- Interactive when possible

### 3. Documentation
- Complete descriptions
- Clear equations
- Usage examples
- Implementation notes

## Integration Notes

### With PyMDP
- Direct matrix compatibility
- Standard agent interface
- Consistent validation
- Performance optimization

### With AgentMaker
- Seamless deployment
- Experiment management
- Result analysis
- Visualization tools



Ensure that common and transferable methods for the GNN rendering are in proper methods and utils scripts. 

Whereas the Render script should fully form all matrices. 

Whereas the Execute script picks up right there from saved files, and runs the simulation according to situational parameters at the top of the file. 