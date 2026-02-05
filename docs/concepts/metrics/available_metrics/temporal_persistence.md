# Temporal Persistence

Temporal Persistence measures stability of a metric across time splits (baseline vs target period).

## Formula
$$
Persistence = target_score / baseline_score
$$

## Inputs
- `baseline_score`
- `target_score`

## Output
- Ratio in [0, 1] in typical cases. Higher is better.

## Examples
```python
from tempoeval.metrics import TemporalPersistence

metric = TemporalPersistence()
score = metric.compute(baseline_score=0.8, target_score=0.6)
```
