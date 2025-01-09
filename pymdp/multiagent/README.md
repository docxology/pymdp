# AWS (Administrator-Website-Consumer) Multi-Agent System

## Overview
This repository implements a multi-agent active inference system modeling the interactions between an Administrator (Adm), Website (Web), and Consumer (Csr) in an advertising context. The system demonstrates how multiple agents can coordinate and interact through active inference principles.

## Theory Background

### Active Inference Framework
The system is built on the Active Inference framework, which unifies:
- Perception (state inference)
- Action (policy selection)
- Learning (model parameter updates)

Through the minimization of variational free energy, agents:
1. Form beliefs about hidden states
2. Select actions to minimize expected free energy
3. Update their generative models based on observations

### Multi-Agent Architecture
The AWS system implements a hierarchical structure where:
- Administrator agent optimizes ad campaign settings
- Website agent manages user interface and targeting
- Consumer agent represents end-user behavior and preferences

## System Architecture

### Core Components
The system consists of three interacting agents, each implemented using Active Inference principles with:
- Generative Model (A, B, C matrices)
- State Space
- Action Space 
- Observation Space

### Agent Specifications

#### Administrator Agent
```
States (sWeb):
- AD_TYPE: [EXPANDED_TEXT_ADS, RESPONSIVE_SEARCH_ADS]
- AD_COPY_CREATION: [DESCRIPTION_LINES, DISPLAY_URL, CALL_TO_ACTION]

Actions (aAdm):
- NULL_ACT
- SET_MATCH_TYPE_ACTION: [BROAD_MATCH_ACT, PHRASE_MATCH_ACT, EXACT_MATCH_ACT]

Observations (yWeb):
- AD_TYPE_OBS: [EXPANDED_TEXT_ADS_OBS, RESPONSIVE_SEARCH_ADS_OBS, UNKNOWN_OBS]
- AD_COPY_CREATION_OBS: [DESCRIPTION_LINES_OBS, DISPLAY_URL_OBS, CALL_TO_ACTION_OBS]
- AD_EXTENSIONS_OBS: [SITE_LINKS_OBS, CALLOUTS_OBS, STRUCTURED_SNIPPETS_OBS]
```

#### Website Agent
```
States (sCsr):
- AD_SCHEDULE: [BUSINESS_HOURS, AFTER_HOURS]
- LOCATION_TARGET: [EAST, CENTRAL, WEST]

Actions (aWeb):
- NULL_ACT
- SET_DEVICE_TARGET_ACTION: [MOBILE_ACT, DESKTOP_ACT, TABLET_ACT]

Observations (yCsr):
- AD_SCHEDULE_OBS: [BUSINESS_HOURS_OBS, OVERLAP_HOURS_OBS, AFTER_HOURS_OBS]
- LOCATION_TARGET_OBS: [EAST_OBS, CENTRAL_OBS, WEST_OBS]
- LANGUAGE_OBS: [ENGLISH_OBS, SPANISH_OBS, FRENCH_OBS]
```

### Matrix Specifications

#### A Matrices (Observation Models)
```python
# Administrator
_Aᴬᵈᵐ: Observation likelihood mapping
- Dimensions: [observation_modalities] × [states]
- Example: P(yᵂᵉᵇ₁ | sᵂᵉᵇ₁, sᵂᵉᵇ₂)

# Website
_Aᵂᵉᵇ: Observation likelihood mapping
- Dimensions: [observation_modalities] × [states]
- Example: P(yᶜˢʳ₁ | sᶜˢʳ₁, sᶜˢʳ₂)
```

#### B Matrices (Transition Models)
```python
# Administrator
_Bᴬᵈᵐ: State transition mapping
- Dimensions: [states_t+1] × [states_t] × [actions]
- Stochastic parameter: _p_stochAdm

# Website
_Bᵂᵉᵇ: State transition mapping
- Dimensions: [states_t+1] × [states_t] × [actions]
- Stochastic parameter: _p_stochWeb
```

#### C Matrices (Preference Models)
```python
# Preference distributions over observations
_Cᴬᵈᵐ, _Cᵂᵉᵇ: Preference mappings
- Encode desired outcomes
- Shape policy selection
```

### Data Structures

#### State and Action Tracking
```python
# Administrator tracking
_aAdm_facs = {'aᴬᵈᵐ₁': [], 'aᴬᵈᵐ₂': []}  # Action history
_sAdm_facs = {'sᵂᵉᵇ₁': [], 'sᵂᵉᵇ₂': []}  # State beliefs
_s̆Web_facs = {'s̆ᵂᵉᵇ₁': [], 's̆ᵂᵉᵇ₂': []}  # True states
_yWeb_mods = {'yᵂᵉᵇ₁': [], 'yᵂᵉᵇ₂': [], 'yᵂᵉᵇ₃': []}  # Observations

# Website tracking
_aWeb_facs = {'aᵂᵉᵇ₁': [], 'aᵂᵉᵇ₂': []}  # Action history
_sWeb_facs = {'sᶜˢʳ₁': [], 'sᶜˢʳ₂': []}  # State beliefs
_s̆Csr_facs = {'s̆ᶜˢʳ₁': [], 's̆ᶜˢʳ₂': []}  # True states
_yCsr_mods = {'yᶜˢʳ₁': [], 'yᶜˢʳ₂': [], 'yᶜˢʳ₃': []}  # Observations
```

#### Policy and Free Energy Tracking
```python
# Administrator inference history
_qAdmIpiIs = []  # Posterior over policies
_GAdmNegs = []   # Negative expected free energy

# Website inference history
_qWebIpiIs = []  # Posterior over policies
_GWebNegs = []   # Negative expected free energy
```

### Initial Configuration

#### Administrator Initial State
```python
_s̆ᵂᵉᵇ = [  # True initial state
    _labAdm['s̆']['s̆ᵂᵉᵇ₁'].index('EXPANDED_TEXT_ADS'),
    _labAdm['s̆']['s̆ᵂᵉᵇ₂'].index('DESCRIPTION_LINES')
]

_yᵂᵉᵇ = [  # Initial observation
    _labAdm['y']['yᵂᵉᵇ₁'].index('UNKNOWN_OBS'),
    _labAdm['y']['yᵂᵉᵇ₂'].index('CALL_TO_ACTION_OBS'),
    _labAdm['y']['yᵂᵉᵇ₃'].index('SITE_LINKS_OBS')
]
```

#### Website Initial State
```python
_s̆ᶜˢʳ = [  # True initial state
    _labWeb['s̆']['s̆ᶜˢʳ₁'].index('BUSINESS_HOURS'),
    _labWeb['s̆']['s̆ᶜˢʳ₂'].index('EAST')
]

_yᶜˢʳ = [  # Initial observation
    _labWeb['y']['yᶜˢʳ₁'].index('BUSINESS_HOURS_OBS'),
    _labWeb['y']['yᶜˢʳ₂'].index('EAST_OBS'),
    _labWeb['y']['yᶜˢʳ₃'].index('ENGLISH_OBS')
]
```

## Implementation Details

### File Structure
- `LL_AWC_Run.py`: Main simulation loop
- `LL_GM_Admin.py`: Administrator agent definition
- `LL_GM_Website.py`: Website agent definition
- `LL_Methods.py`: Shared utility functions
- `LL_AWS_Visualization.py`: Visualization tools

### Key Methods
```python
act():      # Action selection based on policy inference
future():   # Policy inference and expected free energy computation
next():     # State transition computation
observe():  # Observation generation
infer():    # State inference using variational inference
```

### Simulation Parameters
- Default timesteps: 25
- Configurable stochastic transition parameters
- Initial states and observations defined for each agent

### Simulation Flow
1. **Initialization**
   ```python
   # Initialize agents
   _agtAdm = Agent(A=_Aᴬᵈᵐ, B=_Bᴬᵈᵐ, C=_Cᴬᵈᵐ)
   _agtWeb = Agent(A=_Aᵂᵉᵇ, B=_Bᵂᵉᵇ, C=_Cᵂᵉᵇ)
   
   # Set initial states
   _s̆ᵂᵉᵇ = [initial_states]
   _s̆ᶜˢʳ = [initial_states]
   ```

2. **Main Loop**
   ```python
   for t in range(_T):
       # Action selection
       actionAdm = act(_agtAdm, ...)
       actionWeb = act(_agtWeb, ...)
       
       # State transitions
       next(_s̆Web_facs, actionAdm, ...)
       next(_s̆Csr_facs, actionWeb, ...)
       
       # Observation generation
       observe(_yWeb_mods, _yᵂᵉᵇ, ...)
       observe(_yCsr_mods, _yᶜˢʳ, ...)
       
       # State inference
       infer(_agtAdm, _sAdm_facs, ...)
       infer(_agtWeb, _sWeb_facs, ...)
   ```

### Advanced Features

#### Softmax Mappings
```python
# Administrator mappings
_EXPANDED_TEXT_ADS_MAPPING_ADM = softmax([1.0, 0])
_RESPONSIVE_SEARCH_ADS_MAPPING_ADM = softmax([0.0, 1.0])

# Website mappings
_BUSINESS_HOURS_MAPPING_WEB = softmax([1.0, 0])
_AFTER_HOURS_MAPPING_WEB = softmax([0.0, 1.0])
```

#### Visualization Configuration
```python
# Color schemes for different variables
colors = [
    {'NULL_ACT':'black'},
    {'BROAD_MATCH_ACT':'red', 'PHRASE_MATCH_ACT':'green'},
    # ... additional color mappings
]

# Plot parameters
ylabel_size = 12
marker_size = 7
```

## Interaction Dynamics

### State Transitions
- Administrator's actions influence Website states
- Website's actions influence Consumer states
- Stochastic transitions controlled by probability parameters

### Observation Generation
- Probabilistic mappings between true states and observations
- Uncertainty modeled through softmax-transformed distributions
- Special mappings for different state combinations

### Preferences
- Encoded in C matrices
- Drive goal-directed behavior
- Customizable for different scenarios

## Visualization

The system includes comprehensive visualization tools for:
- Actions over time
- True state evolution
- Inferred state beliefs
- Observations
- Color-coded time series with legends

## Extension Points

### Possible Extensions
1. Additional state factors
2. More complex action spaces
3. Hierarchical agent structures
4. Learning dynamics
5. Enhanced preference learning

### Integration Points
1. External API connections
2. Real-time data processing
3. Advanced visualization tools
4. Performance monitoring
5. A/B testing capabilities

## Development Guidelines

### Best Practices
1. Use descriptive variable names with clear prefixes
2. Maintain consistent matrix dimensions
3. Document probability distributions
4. Handle edge cases in state transitions
5. Validate observation generation

### Contributing
1. Fork the repository
2. Create a feature branch
3. Follow the coding style
4. Add tests if applicable
5. Submit a pull request

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- PyMDP

## Installation
```bash
git clone [repository-url]
cd pymdp/multiagent
pip install -r requirements.txt
```

## Usage
```python
# Run the simulation
python LL_AWC_Run.py

# Visualize results
python LL_AWS_Visualization.py
```

## License
[Specify License]

## Contact
[Specify Contact Information]

## Performance Optimization

### Memory Management
- Use NumPy arrays for efficient matrix operations
- Implement sparse matrices for large state spaces
- Cache frequently accessed computations

### Computational Efficiency
- Vectorized operations where possible
- Parallel processing for independent agent computations
- Optimized belief updating algorithms

## Debugging and Testing

### Debugging Tools
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Matrix dimension verification
def verify_dimensions(A, B, C):
    # Implementation
    pass

# State-space consistency checks
def validate_state_space(states, actions):
    # Implementation
    pass
```

### Testing Framework
```python
# Unit tests
def test_state_transitions():
    # Test B matrix properties
    pass

def test_observation_generation():
    # Test A matrix properties
    pass

# Integration tests
def test_agent_interaction():
    # Test multi-agent dynamics
    pass
```

## Examples

### Basic Usage
```python
# Initialize system
from aws_system import AWSystem
aws = AWSystem()

# Run simulation
results = aws.run_simulation(timesteps=25)

# Analyze results
aws.plot_results(results)
```

### Custom Agent Configuration
```python
# Modify agent preferences
custom_C = create_custom_preferences()
aws.set_agent_preferences(agent_type='Admin', preferences=custom_C)

# Add new state factors
new_states = define_new_states()
aws.extend_state_space(new_states)
```

### Data Analysis
```python
# Extract belief trajectories
beliefs = aws.get_belief_trajectories()

# Compute metrics
performance = aws.compute_metrics(beliefs)

# Export results
aws.export_results('results.csv')
```

## Troubleshooting

### Common Issues
1. Matrix dimension mismatches
   - Solution: Use `verify_dimensions()` helper
2. Convergence problems
   - Solution: Adjust learning rates
3. Memory constraints
   - Solution: Implement sparse representations

### Error Messages
- `InvalidDimensionError`: Matrix dimensions don't match
- `ConvergenceWarning`: Belief updating not converging
- `InvalidPolicyError`: Policy specification incorrect

## Advanced Configuration

### Environment Variables
```bash
export AWS_DEBUG_LEVEL=INFO
export AWS_MAX_TIMESTEPS=100
export AWS_RANDOM_SEED=42
```

### Configuration File
```yaml
# config.yaml
simulation:
  timesteps: 25
  random_seed: 42
  
agents:
  administrator:
    learning_rate: 0.1
    temperature: 1.0
  website:
    learning_rate: 0.1
    temperature: 1.0
```




NOTE ON MULTI-AGENT GRAPHICS
Each entity has
an agent
an environment
The name/identifier of an entity always starts with a single capital letter, e.g.
‘Administrator’, or in abbreviated form, say ‘Adm’
‘Website’, or in abbreviated form, say ‘Web’ or ‘Wsi’
‘Consumer’, or in abbreviated form, say ‘Csr’ or ‘Con’
In general, the environment for a specific entity agent consists of multiple environment segments
Each entity has one colocated environment segment, e.g.
An ant entity’s agent (inside its brain) is colocated with the ant’s own environment segment which contains, for example, a state variable capturing the ant’s location. This ant agent may also be interested in influencing other environment segments.
An entity’s environment consists of its colocated environment segment as well as (potentially) all other entities’ colocated environment segments
An entity’s agent may be interested in emitted observations from a subset of other entities’ environment segments
An entity’s agent may be interested in emitting actions to a subset of other entities’ environment segments
The graphic of an ent has a vertical dotted line separating its agt from its colocated environment
If an entity mainly interacts with one other entity, their graphics are usually placed next to each other horizontally
All actions flow at the top
All observations flow at the bottom
When too many actions and observations need to be shown, one action busbar is drawn at the top, and one observation busbar at the bottom
An entity’s agent usually attempts to steer or influence at least one environment segment
An entity’s agent is usually informed by observations from at least one environment segment
An entity may be impacted by exogenous information/signals, indicated by 
All variables/signals 
 are identified by 
The standard symbols for the identification of variables/signals are:
observation 
 where Ent is the emitter of the observation
action 
 where Ent is the emitter of the action
state 
 where Ent is the owner of the inferred state
parameter 
 where Ent is the owner of the inferred parameter
control 
 where Ent is the owner of the inferred control
“True” (i.e. non-inferred) symbols have a breve symbol on top, e.g.
 is the true parameters of the Website entity
 is the true state of the Website entity
 is the true parameters of the Consumer entity
 is the true state of the Consumer entity
The Administrator agent has symbols:

 is the inferred control states of the Website entity
 is the inferred parameters of the Website entity
 is the inferred state of the Website entity
 is the observation from the Website entity
The Website environment has symbols:

 is the action from the Administrator entity
 is the true parameters of the Website entity
 is the true state of the Website entity
 is the exogenous information impacting the Website entity
 is the observation from the Website entity
The Website agent has symbols:

 is the inferred control states of the Consumer entity
 is the inferred parameters of the Consumer entity
 is the inferred state of the Consumer entity
 is the observation from the Consumer entity
The Consumer environment has symbols:

 is the action from the Website entity
 is the true parameters of the Consumer entity
 is the true state of the Consumer entity
 is the exogenous information impacting the Consumer entity
 is the observation from the Consumer entity