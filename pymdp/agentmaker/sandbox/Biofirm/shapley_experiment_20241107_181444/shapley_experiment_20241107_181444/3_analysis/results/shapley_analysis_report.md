# Biofirm Shapley Analysis Report

## Overall Performance Summary
### Key Findings

#### Homeostasis Satisfaction
- Mean Shapley Contribution: 0.0012
- Mean Coalition Performance: 0.3069
- Best Coalition: 0 (0.3470)
- Performance Range: 0.0800
- Strongest Contributor: 1-3-1 (0.0286)

#### Expected Free Energy
- Mean Shapley Contribution: 0.0022
- Mean Coalition Performance: -0.0353
- Best Coalition: 2 (-0.0034)
- Performance Range: 0.0755
- Strongest Contributor: 0-6-0 (0.0386)

#### Belief Accuracy
- Mean Shapley Contribution: 0.0039
- Mean Coalition Performance: 0.3654
- Best Coalition: 0 + 1 + 2 + 4 (0.3890)
- Performance Range: 0.0470
- Strongest Contributor: base (0.0293)

#### Control Efficiency
- Mean Shapley Contribution: -0.0056
- Mean Coalition Performance: 0.2548
- Best Coalition: 0 + 2 (0.2810)
- Performance Range: 0.0800
- Strongest Contributor: 2-2-2 (-0.0012)

## Individual Agent Contributions

### Homeostasis Satisfaction
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| 1-3-1 | 0.0286 | 42.9% | 1 |
| .1-2-.1 | -0.0176 | 26.3% | 2 |
| base | -0.0081 | 12.1% | 3 |
| 2-2-2 | 0.0078 | 11.6% | 4 |
| 0-6-0 | -0.0047 | 7.1% | 5 |

### Expected Free Energy
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| 0-6-0 | 0.0386 | 42.7% | 1 |
| 1-3-1 | -0.0296 | 32.8% | 2 |
| 2-2-2 | 0.0121 | 13.4% | 3 |
| .1-2-.1 | -0.0069 | 7.7% | 4 |
| base | -0.0031 | 3.4% | 5 |

### Belief Accuracy
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| base | 0.0293 | 45.8% | 1 |
| 1-3-1 | -0.0197 | 30.8% | 2 |
| .1-2-.1 | 0.0123 | 19.3% | 3 |
| 2-2-2 | -0.0021 | 3.4% | 4 |
| 0-6-0 | -0.0005 | 0.7% | 5 |

### Control Efficiency
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| base | -0.0108 | 38.6% | 1 |
| 0-6-0 | -0.0094 | 33.8% | 2 |
| 1-3-1 | -0.0035 | 12.5% | 3 |
| .1-2-.1 | -0.0030 | 10.9% | 4 |
| 2-2-2 | -0.0012 | 4.2% | 5 |

## Coalition Analysis

### Homeostasis Satisfaction
- Average Performance: 0.3069
- Variability (std): 0.0192
- Performance Range: 0.0800

Top Performing Coalitions:
- base: 0.3470
- 0-6-0 + .1-2-.1: 0.3440
- 1-3-1 + 2-2-2: 0.3370

Worst Performing Coalitions:
- base + .1-2-.1 + 2-2-2: 0.2760
- base + 0-6-0 + .1-2-.1: 0.2740
- 0-6-0 + .1-2-.1 + 1-3-1: 0.2670

### Expected Free Energy
- Average Performance: -0.0353
- Variability (std): 0.0172
- Performance Range: 0.0755

Top Performing Coalitions:
- .1-2-.1: -0.0034
- base + .1-2-.1 + 2-2-2: -0.0035
- base + .1-2-.1: -0.0179

Worst Performing Coalitions:
- base + .1-2-.1 + 1-3-1: -0.0598
- base + .1-2-.1 + 1-3-1 + 2-2-2: -0.0782
- base + 1-3-1: -0.0789

### Belief Accuracy
- Average Performance: 0.3654
- Variability (std): 0.0108
- Performance Range: 0.0470

Top Performing Coalitions:
- base + 0-6-0 + .1-2-.1 + 2-2-2: 0.3890
- base + 0-6-0 + 1-3-1: 0.3840
- .1-2-.1 + 2-2-2: 0.3800

Worst Performing Coalitions:
- base + 0-6-0: 0.3510
- 0-6-0: 0.3460
- 0-6-0 + .1-2-.1 + 1-3-1 + 2-2-2: 0.3420

### Control Efficiency
- Average Performance: 0.2548
- Variability (std): 0.0166
- Performance Range: 0.0800

Top Performing Coalitions:
- base + .1-2-.1: 0.2810
- base + .1-2-.1 + 1-3-1: 0.2780
- 0-6-0 + .1-2-.1 + 1-3-1 + 2-2-2: 0.2730

Worst Performing Coalitions:
- base + 0-6-0 + 1-3-1: 0.2270
- 1-3-1 + 2-2-2: 0.2240
- 0-6-0 + .1-2-.1: 0.2010

## Synergy Analysis

### Homeostasis Satisfaction

Most Synergistic Pairs:

Most Antagonistic Pairs:

### Expected Free Energy

Most Synergistic Pairs:

Most Antagonistic Pairs:

### Belief Accuracy

Most Synergistic Pairs:

Most Antagonistic Pairs:

### Control Efficiency

Most Synergistic Pairs:

Most Antagonistic Pairs:

## Methodology

### Shapley Value Calculation
Shapley values were calculated using the formula:
```
φᵢ(v) = Σ [|S|!*(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]
```
Where:
- S: Coalition subset not containing agent i
- v(S): Performance value of coalition S
- n: Total number of agents
