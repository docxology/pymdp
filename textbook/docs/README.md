# PyMDP Textbook Documentation

This directory contains comprehensive documentation for the PyMDP textbook project, focusing on active inference, partially observable Markov decision processes (POMDPs), and related computational methods.

## Documentation Structure

### Core Concepts
- [`active_inference_basics.md`](active_inference_basics.md) - Fundamental concepts of active inference
- [`pomdp_theory.md`](pomdp_theory.md) - Mathematical foundation of POMDPs
- [`free_energy_principle.md`](free_energy_principle.md) - The free energy principle explained
- [`bayesian_brain.md`](bayesian_brain.md) - Bayesian brain hypothesis and predictive processing

### Mathematical Foundations
- [`mathematical_notation.md`](mathematical_notation.md) - Notation and conventions used
- [`probability_theory.md`](probability_theory.md) - Essential probability theory
- [`variational_inference.md`](variational_inference.md) - Variational methods for approximate inference
- [`message_passing.md`](message_passing.md) - Message passing algorithms

### Implementation Guides
- [`pymdp_overview.md`](pymdp_overview.md) - Overview of PyMDP package structure
- [`model_specification.md`](model_specification.md) - How to specify generative models
- [`inference_algorithms.md`](inference_algorithms.md) - Available inference methods
- [`control_algorithms.md`](control_algorithms.md) - Action selection and planning

### Advanced Topics
- [`hierarchical_models.md`](hierarchical_models.md) - Multi-level generative models
- [`learning_dynamics.md`](learning_dynamics.md) - Parameter learning and adaptation
- [`multi_agent_systems.md`](multi_agent_systems.md) - Multi-agent active inference
- [`continuous_time.md`](continuous_time.md) - Continuous-time formulations

### Applications
- [`cognitive_modeling.md`](cognitive_modeling.md) - Modeling cognitive processes
- [`robotics_applications.md`](robotics_applications.md) - Active inference in robotics
- [`neuroscience_applications.md`](neuroscience_applications.md) - Computational neuroscience

## Reading Path

### For Beginners
1. Start with `active_inference_basics.md`
2. Read `pomdp_theory.md` for mathematical background
3. Follow `pymdp_overview.md` for implementation details
4. Work through examples in the `../examples/` directory

### For Advanced Users
1. Review `free_energy_principle.md` for theoretical depth
2. Study `variational_inference.md` for algorithmic details
3. Explore `hierarchical_models.md` and `learning_dynamics.md`
4. Check application-specific documents

## Contributing

When adding new documentation:
1. Follow the established structure and naming conventions
2. Include mathematical formulations where appropriate
3. Provide code examples that link to the examples directory
4. Cross-reference related concepts
5. Keep explanations clear and progressive

## References

Each document includes its own reference list, but key general references include:

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., & Friston, K. J. (2017). Uncertainty, epistemics and active inference
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces
- Fountas, Z., et al. (2020). Deep active inference agents using Monte-Carlo methods

## Notation Conventions

- Consistent mathematical notation across all documents
- Clear distinction between random variables and their realizations
- Standard probability theory symbols and conventions
- PyMDP-specific variable names and structures

See [`mathematical_notation.md`](mathematical_notation.md) for complete details.
