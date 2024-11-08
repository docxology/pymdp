# Biofirm Shapley Value Analysis

## Overview
The Shapley value analysis module (`Biofirm_Shapley.py`) provides a comprehensive framework for evaluating agent contributions in multi-agent Biofirm systems. This analysis helps understand agent synergies, optimal combinations, and individual strengths using cooperative game theory principles.

## Core Concepts

### Agent Types
1. **Base Agent**
   - Standard homeostatic preferences: `[0.0, 4.0, 0.0]`
   - Balanced performance characteristics
   - Reference point for comparisons

2. **Risk-Averse Agent**
   - Higher homeostatic preference: `[0.0, 6.0, 0.0]`
   - Prioritizes stability over exploration
   - Strong homeostatic maintenance

3. **Exploratory Agent**
   - Lower preference differential: `[0.1, 2.0, 0.1]`
   - Enhanced environmental sampling
   - Greater adaptability to changes

4. **Balanced Agent**
   - Moderate with exploration: `[1.0, 3.0, 1.0]`
   - Compromise between stability and adaptation
   - Robust general performance

## Performance Metrics

### 1. Homeostasis Satisfaction
- Primary metric for system stability
- Calculated as: `time_in_homeostasis / total_time`
- Weighted by state proximity to target
- Success threshold: > 0.75

### 2. Expected Free Energy
- Measures predictive performance
- Components:
  - Ambiguity term (epistemic value)
  - Risk term (pragmatic value)
- Normalized to [-1, 0] range

### 3. Belief Accuracy
- State estimation precision
- Calculated using KL divergence
- Includes:
  - State belief accuracy
  - Observation model accuracy
  - Transition model accuracy

### 4. Control Efficiency
- Action selection optimality
- Factors:
  - Action cost
  - State transition efficiency
  - Policy entropy

## Analysis Framework

### Coalition Analysis
```python
class CoalitionAnalysis:
    def __init__(self, agents, metrics):
        self.agents = agents
        self.metrics = metrics
        self.coalitions = self._generate_coalitions()
    
    def _generate_coalitions(self):
        """Generate all possible agent combinations"""
        return [combo for n in range(1, len(self.agents) + 1)
                for combo in combinations(self.agents, n)]
```

### Performance Evaluation
```python
def evaluate_coalition(coalition, environment, timesteps=1000):
    """Evaluate coalition performance across metrics"""
    results = {
        'homeostasis': measure_homeostasis(coalition, environment),
        'free_energy': calculate_free_energy(coalition),
        'belief_acc': assess_belief_accuracy(coalition),
        'control_eff': measure_control_efficiency(coalition)
    }
    return results
```

### Shapley Calculation
```python
def calculate_shapley_values(coalition_values):
    """Calculate Shapley values for each agent"""
    n = len(agents)
    shapley_values = {}
    
    for agent in agents:
        value = sum(
            len(S) * math.factorial(n-len(S)-1) * math.factorial(len(S)) / 
            math.factorial(n) * (coalition_values[S + [agent]] - coalition_values[S])
            for S in powerset(agents - {agent})
        )
        shapley_values[agent] = value
    
    return shapley_values
```

## Output Structure

### Directory Organization
```
shapley_experiment_TIMESTAMP/
├── 1_agents/                    # Agent definitions
│   ├── base_agent.gnn
│   ├── risk_averse_agent.gnn
│   ├── exploratory_agent.gnn
│   └── balanced_agent.gnn
├── 2_coalitions/               # Coalition results
│   └── coalition_N_[ids]/
│       ├── simulation_data/
│       ├── performance_metrics/
│       └── analysis_results/
├── 3_analysis/                # Shapley analysis
│   ├── results/
│   │   ├── raw_data/
│   │   ├── processed_results/
│   │   └── statistical_analysis/
│   └── visualizations/
└── metadata/
```

### Data Formats

#### Shapley Analysis JSON
```json
{
  "agents": {
    "base": {"shapley_value": 0.25, "metrics": {...}},
    "risk_averse": {"shapley_value": 0.30, "metrics": {...}},
    "exploratory": {"shapley_value": 0.20, "metrics": {...}},
    "balanced": {"shapley_value": 0.25, "metrics": {...}}
  },
  "coalitions": {
    "base_risk": {"performance": 0.65, "synergy": 0.1},
    "base_exploratory": {"performance": 0.60, "synergy": 0.05}
  }
}
```

## Visualization Suite

### 1. Agent Contributions
- Individual Shapley values
- Contribution breakdowns
- Temporal evolution

### 2. Coalition Performance
- Performance heatmaps
- Synergy matrices
- Size-performance relationships

### 3. Metric Analysis
- Correlation matrices
- Trade-off plots
- Performance distributions

### 4. Advanced Visualizations
- Network diagrams
- Dynamic trajectories
- Uncertainty quantification

## Usage Guide

### Basic Usage
```bash
# Run complete analysis
python3 Biofirm_Shapley.py --config standard_config.yaml

# Analysis with custom parameters
python3 Biofirm_Shapley.py --agents custom_agents.json --metrics custom_metrics.yaml
```

### Configuration Options
```yaml
analysis_config:
  iterations: 1000
  confidence_level: 0.95
  bootstrap_samples: 500
  
visualization_config:
  style: 'publication'
  format: 'svg'
  dpi: 300
```

## Advanced Features

### 1. Statistical Analysis
- Bootstrap confidence intervals
- Significance testing
- Effect size calculations

### 2. Sensitivity Analysis
- Parameter perturbation
- Robustness testing
- Uncertainty propagation

### 3. Custom Metrics
- Metric definition framework
- Weighting schemes
- Composite indicators

## Best Practices

### 1. Experimental Design
- Use sufficient iterations
- Implement proper controls
- Account for stochasticity

### 2. Analysis Workflow
- Validate assumptions
- Check for convergence
- Document parameters

### 3. Reporting
- Include all configurations
- Report uncertainty
- Provide raw data

## Extensions and Future Work

### 1. Enhanced Metrics
- Information-theoretic measures
- Multi-objective optimization
- Context-specific indicators

### 2. Advanced Analysis
- Temporal Shapley values
- Dynamic coalition formation
- Hierarchical analysis

### 3. Integration
- Real-time analysis
- Distributed computation
- Interactive visualization

## References
- Shapley, L.S. (1953). A Value for n-Person Games
- Active Inference: Theory and Implementation
- Cooperative Game Theory in Multi-Agent Systems 