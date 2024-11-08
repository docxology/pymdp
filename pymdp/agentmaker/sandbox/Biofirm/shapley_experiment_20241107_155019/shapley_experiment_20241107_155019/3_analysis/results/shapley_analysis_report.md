# Biofirm Shapley Analysis Report

## Overall Performance Summary
### Key Findings

#### Homeostasis Satisfaction
- Mean Shapley Contribution: -0.0024
- Mean Coalition Performance: 0.3035
- Best Coalition: 0 (0.3380)
- Performance Range: 0.0710
- Strongest Contributor: exploratory (0.0182)

#### Expected Free Energy
- Mean Shapley Contribution: 0.0170
- Mean Coalition Performance: -0.0333
- Best Coalition: 0 + 1 + 2 + 3 (-0.0134)
- Performance Range: 0.0540
- Strongest Contributor: balanced (0.0439)

#### Belief Accuracy
- Mean Shapley Contribution: -0.0124
- Mean Coalition Performance: 0.3711
- Best Coalition: 1 + 3 (0.4090)
- Performance Range: 0.0670
- Strongest Contributor: risk_averse (0.0083)

#### Control Efficiency
- Mean Shapley Contribution: -0.0024
- Mean Coalition Performance: 0.2493
- Best Coalition: 2 + 3 (0.2710)
- Performance Range: 0.0630
- Strongest Contributor: balanced (0.0030)

## Individual Agent Contributions

### Homeostasis Satisfaction
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| exploratory | 0.0182 | 39.7% | 1 |
| risk_averse | -0.0125 | 27.3% | 2 |
| base | -0.0102 | 22.2% | 3 |
| balanced | -0.0049 | 10.7% | 4 |

### Expected Free Energy
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| balanced | 0.0439 | 64.6% | 1 |
| exploratory | 0.0114 | 16.8% | 2 |
| base | 0.0090 | 13.2% | 3 |
| risk_averse | 0.0037 | 5.4% | 4 |

### Belief Accuracy
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| balanced | -0.0294 | 44.3% | 1 |
| exploratory | -0.0150 | 22.6% | 2 |
| base | -0.0136 | 20.5% | 3 |
| risk_averse | 0.0083 | 12.6% | 4 |

### Control Efficiency
| Agent | Shapley Value | Relative Contribution | Rank |
|-------|---------------|---------------------|------|
| exploratory | -0.0054 | 34.4% | 1 |
| risk_averse | -0.0051 | 32.3% | 2 |
| balanced | 0.0030 | 19.0% | 3 |
| base | -0.0023 | 14.3% | 4 |

## Coalition Analysis

### Homeostasis Satisfaction
- Average Performance: 0.3035
- Variability (std): 0.0224
- Performance Range: 0.0710

Top Performing Coalitions:
- base: 0.3380
- base + exploratory: 0.3330
- risk_averse + exploratory: 0.3320

Worst Performing Coalitions:
- base + risk_averse + exploratory: 0.2800
- exploratory + balanced: 0.2780
- base + risk_averse + balanced: 0.2670

### Expected Free Energy
- Average Performance: -0.0333
- Variability (std): 0.0157
- Performance Range: 0.0540

Top Performing Coalitions:
- base + risk_averse + exploratory + balanced: -0.0134
- risk_averse + balanced: -0.0164
- base + exploratory + balanced: -0.0182

Worst Performing Coalitions:
- risk_averse: -0.0521
- base + risk_averse + exploratory: -0.0552
- exploratory: -0.0673

### Belief Accuracy
- Average Performance: 0.3711
- Variability (std): 0.0178
- Performance Range: 0.0670

Top Performing Coalitions:
- risk_averse + balanced: 0.4090
- base + risk_averse + exploratory: 0.3970
- balanced: 0.3840

Worst Performing Coalitions:
- base + exploratory + balanced: 0.3550
- risk_averse + exploratory: 0.3480
- base + balanced: 0.3420

### Control Efficiency
- Average Performance: 0.2493
- Variability (std): 0.0171
- Performance Range: 0.0630

Top Performing Coalitions:
- exploratory + balanced: 0.2710
- balanced: 0.2670
- risk_averse + exploratory + balanced: 0.2640

Worst Performing Coalitions:
- risk_averse: 0.2270
- risk_averse + balanced: 0.2270
- risk_averse + exploratory: 0.2080

## Synergy Analysis

### Homeostasis Satisfaction

Most Synergistic Pairs:
- risk_averse-exploratory: -0.2770 (Joint: 0.3320, Individual: 0.2870, 0.3220)
- risk_averse-balanced: -0.2990 (Joint: 0.3170, Individual: 0.2870, 0.3290)
- base-exploratory: -0.3270 (Joint: 0.3330, Individual: 0.3380, 0.3220)

Most Antagonistic Pairs:
- base-risk_averse: -0.3430 (Joint: 0.2820, Individual: 0.3380, 0.2870)
- base-balanced: -0.3660 (Joint: 0.3010, Individual: 0.3380, 0.3290)
- exploratory-balanced: -0.3730 (Joint: 0.2780, Individual: 0.3220, 0.3290)

### Expected Free Energy

Most Synergistic Pairs:
- risk_averse-exploratory: 0.0866 (Joint: -0.0329, Individual: -0.0521, -0.0673)
- base-exploratory: 0.0796 (Joint: -0.0385, Individual: -0.0508, -0.0673)
- base-risk_averse: 0.0743 (Joint: -0.0286, Individual: -0.0508, -0.0521)

Most Antagonistic Pairs:
- exploratory-balanced: 0.0600 (Joint: -0.0274, Individual: -0.0673, -0.0200)
- risk_averse-balanced: 0.0557 (Joint: -0.0164, Individual: -0.0521, -0.0200)
- base-balanced: 0.0513 (Joint: -0.0195, Individual: -0.0508, -0.0200)

### Belief Accuracy

Most Synergistic Pairs:
- risk_averse-balanced: -0.3470 (Joint: 0.4090, Individual: 0.3720, 0.3840)
- base-risk_averse: -0.3670 (Joint: 0.3790, Individual: 0.3740, 0.3720)
- exploratory-balanced: -0.3790 (Joint: 0.3840, Individual: 0.3790, 0.3840)

Most Antagonistic Pairs:
- base-exploratory: -0.3970 (Joint: 0.3560, Individual: 0.3740, 0.3790)
- risk_averse-exploratory: -0.4030 (Joint: 0.3480, Individual: 0.3720, 0.3790)
- base-balanced: -0.4160 (Joint: 0.3420, Individual: 0.3740, 0.3840)

### Control Efficiency

Most Synergistic Pairs:
- base-risk_averse: -0.2360 (Joint: 0.2410, Individual: 0.2500, 0.2270)
- base-exploratory: -0.2440 (Joint: 0.2600, Individual: 0.2500, 0.2540)
- exploratory-balanced: -0.2500 (Joint: 0.2710, Individual: 0.2540, 0.2670)

Most Antagonistic Pairs:
- base-balanced: -0.2540 (Joint: 0.2630, Individual: 0.2500, 0.2670)
- risk_averse-balanced: -0.2670 (Joint: 0.2270, Individual: 0.2270, 0.2670)
- risk_averse-exploratory: -0.2730 (Joint: 0.2080, Individual: 0.2270, 0.2540)

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
