# PyMDP Textbook Examples

This directory contains step-by-step examples that build up from basic concepts to full POMDP implementations using PyMDP and active inference.

## Learning Path

The examples are designed to be worked through in order, with each building upon previous concepts:

### Foundation (Examples 1-3)
1. **[`01_probability_basics.py`](01_probability_basics.py)** - Probability distributions and basic operations
2. **[`02_bayes_rule.py`](02_bayes_rule.py)** - Bayes rule and belief updating  
3. **[`03_observation_models.py`](03_observation_models.py)** - Building observation models (A matrices)

### State Inference (Examples 4-6)
4. **[`04_state_inference.py`](04_state_inference.py)** - Inferring hidden states from observations
5. **[`05_sequential_inference.py`](05_sequential_inference.py)** - Inference over time sequences
6. **[`06_multi_factor_models.py`](06_multi_factor_models.py)** - Models with multiple state factors

### Dynamics and Control (Examples 7-9)
7. **[`07_transition_models.py`](07_transition_models.py)** - Building transition models (B matrices)
8. **[`08_preferences_and_control.py`](08_preferences_and_control.py)** - Preferences (C vectors) and action selection
9. **[`09_policy_inference.py`](09_policy_inference.py)** - Planning and policy inference

### Complete POMDP Examples (Examples 10-12)
10. **[`10_simple_pomdp.py`](10_simple_pomdp.py)** - Complete POMDP with inference and control
11. **[`11_gridworld_pomdp.py`](11_gridworld_pomdp.py)** - Grid world navigation with active inference
12. **[`12_tmaze_pomdp.py`](12_tmaze_pomdp.py)** - T-maze decision making under uncertainty

## Usage

Each example is self-contained and can be run independently:

```bash
cd textbook/examples/
python 01_probability_basics.py
```

However, we recommend working through them in order as concepts build upon each other.

## Interactive Mode

Many examples include interactive components. Run with the `--interactive` flag for step-by-step execution:

```bash
python 04_state_inference.py --interactive
```

## Key Concepts Covered

### Mathematical Foundations
- Probability distributions and normalization
- Bayes rule and belief updating
- Information theory (entropy, KL divergence)
- Matrix operations and tensor manipulations

### Generative Models
- Observation models (A matrices): P(observation | state)
- Transition models (B matrices): P(next_state | current_state, action)  
- Preference models (C vectors): Prior preferences over observations
- Prior beliefs (D vectors): Initial state distributions

### Inference and Control
- State inference using variational message passing
- Policy inference and action selection
- Expected free energy minimization
- Exploration vs exploitation trade-offs

### Learning and Adaptation
- Parameter learning (updating A, B matrices)
- Structure learning and model selection
- Online adaptation and plasticity

## Tips for Learning

1. **Start Simple**: Begin with the foundational examples even if you're familiar with the concepts
2. **Run the Code**: Execute each example and examine the outputs carefully
3. **Modify Parameters**: Change model parameters to see how behavior changes
4. **Visualize Results**: Use the visualization tools to understand what's happening
5. **Read Documentation**: Refer to the `../docs/` folder for theoretical background

## Getting Help

- Check the `../docs/` directory for detailed explanations
- Look at the `../tests/` directory for more code examples
- Use the visualization tools in `../src/visualization.py`
- Run the setup script if you encounter import errors: `bash ../setup.sh`

## Contributing

When adding new examples:
1. Follow the naming convention: `##_descriptive_name.py`
2. Include docstrings and comments explaining key concepts
3. Add visualization where helpful
4. Test that examples run without errors
5. Update this README with the new example

## References
The primary reference for this textbook is:

Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. The MIT Press. DOI: https://doi.org/10.7551/mitpress/12441.001.0001. ISBN electronic: 9780262369978. Available at: https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind

Additional relevant references include:

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., & Friston, K. J. (2017). Uncertainty, epistemics and active inference  
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces
- Heins, C., et al. (2022). pymdp: A Python library for active inference
