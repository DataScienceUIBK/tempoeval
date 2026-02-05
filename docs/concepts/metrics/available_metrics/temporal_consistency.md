# Temporal Consistency

Temporal Consistency measures whether multiple sampled answers agree on temporal information.

## Formula
$$
Consistency = consistent_pairs / total_pairs
$$

## Inputs
- `answers` (list of answer strings)

## Output
- Range: [0, 1], higher is better.

## Examples
```python
from tempoeval.metrics import TemporalConsistency

metric = TemporalConsistency()
score = metric.compute(
    answers=["Python was created in 1991", "Python was created in 1991"]
)
```
