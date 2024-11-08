# Biofirm Shapley Analysis Report

## Overall Performance Summary
### Key Findings

#### Homeostasis Satisfaction
- Mean Shapley Contribution: 0.0197
- Mean Coalition Performance: 0.3095
- Best Coalition: 0 + 1 + 2 (0.3620)
- Performance Range: 0.1010
- Strongest Contributor: .1-2-.1 (0.0519)

#### Expected Free Energy
- Mean Shapley Contribution: -0.0210
- Mean Coalition Performance: -0.0340
- Best Coalition: 0 + 1 + 3 + 4 (-0.0050)
- Performance Range: 0.0611
- Strongest Contributor: base (0.0151)

#### Belief Accuracy
- Mean Shapley Contribution: -0.0089
- Mean Coalition Performance: 0.3665
- Best Coalition: 2 + 3 (0.4090)
- Performance Range: 0.0680
- Strongest Contributor: base (-0.0010)

#### Control Efficiency
- Mean Shapley Contribution: 0.0354
- Mean Coalition Performance: 0.2472
- Best Coalition: 0 + 1 + 2 + 3 + 4 (0.2900)
- Performance Range: 0.1060
- Strongest Contributor: 0-6-0 (0.0625)

## Individual Agent Contributions

### Homeostasis Satisfaction
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| .1-2-.1 | 0.0519 | 52.8% | 1 |
| 0-6-0 | 0.0156 | 15.9% | 2 |
| base | 0.0153 | 15.6% | 3 |
| Extreme | 0.0102 | 10.4% | 4 |
| 2-2-2 | 0.0053 | 5.4% | 5 |

### Expected Free Energy
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| .1-2-.1 | -0.0404 | 29.9% | 1 |
| 0-6-0 | -0.0390 | 28.8% | 2 |
| 2-2-2 | -0.0245 | 18.2% | 3 |
| Extreme | -0.0162 | 12.0% | 4 |
| base | 0.0151 | 11.2% | 5 |

### Belief Accuracy
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| .1-2-.1 | -0.0181 | 40.8% | 1 |
| Extreme | -0.0111 | 25.1% | 2 |
| 2-2-2 | -0.0075 | 17.0% | 3 |
| 0-6-0 | -0.0066 | 14.9% | 4 |
| base | -0.0010 | 2.2% | 5 |

### Control Efficiency
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| 0-6-0 | 0.0625 | 35.4% | 1 |
| 2-2-2 | 0.0333 | 18.8% | 2 |
| Extreme | 0.0284 | 16.0% | 3 |
| base | 0.0283 | 16.0% | 4 |
| .1-2-.1 | 0.0243 | 13.8% | 5 |

## Coalition Analysis

### Homeostasis Satisfaction
- Average Performance: 0.3095
- Variability (std): 0.0222
- Performance Range: 0.1010

Top Performing Coalitions:
- base + 0-6-0 + .1-2-.1: 0.3620
- 0-6-0 + .1-2-.1: 0.3420
- 0-6-0 + Extreme + 2-2-2: 0.3380

Worst Performing Coalitions:
- base + 0-6-0 + Extreme: 0.2690
- base + 0-6-0 + 2-2-2: 0.2690
- base + Extreme + 2-2-2: 0.2610

### Expected Free Energy
- Average Performance: -0.0340
- Variability (std): 0.0158
- Performance Range: 0.0611

Top Performing Coalitions:
- base + 0-6-0 + Extreme + 2-2-2: -0.0050
- base + 0-6-0 + .1-2-.1: -0.0085
- base + .1-2-.1 + Extreme + 2-2-2: -0.0140

Worst Performing Coalitions:
- base + 0-6-0 + .1-2-.1 + Extreme + 2-2-2: -0.0598
- 0-6-0 + .1-2-.1 + Extreme + 2-2-2: -0.0634
- 0-6-0: -0.0661

### Belief Accuracy
- Average Performance: 0.3665
- Variability (std): 0.0157
- Performance Range: 0.0680

Top Performing Coalitions:
- .1-2-.1 + Extreme: 0.4090
- base + 0-6-0 + 2-2-2: 0.3970
- base + Extreme + 2-2-2: 0.3920

Worst Performing Coalitions:
- base + .1-2-.1: 0.3450
- 0-6-0 + .1-2-.1: 0.3450
- .1-2-.1 + 2-2-2: 0.3410

### Control Efficiency
- Average Performance: 0.2472
- Variability (std): 0.0201
- Performance Range: 0.1060

Top Performing Coalitions:
- base + 0-6-0 + .1-2-.1 + Extreme + 2-2-2: 0.2900
- 0-6-0 + .1-2-.1 + 2-2-2: 0.2710
- 0-6-0 + 2-2-2: 0.2700

Worst Performing Coalitions:
- base + .1-2-.1 + Extreme + 2-2-2: 0.2220
- Extreme: 0.2190
- base + 0-6-0 + .1-2-.1: 0.1840

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
