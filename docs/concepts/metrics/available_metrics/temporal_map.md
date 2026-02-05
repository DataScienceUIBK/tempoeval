# Temporal MAP@K

<span class="metric-badge retrieval">Retrieval Metric</span>

Temporal Mean Average Precision (MAP@K) measures precision across all relevant documents in the ranking, rewarding systems that rank relevant documents higher.

## Formula

$$
\text{TemporalMAP@K} = \frac{1}{R} \sum_{r=1}^{K} P@r \cdot \text{rel}(d_r, q)
$$

Where:

- $R$ = total number of relevant documents in the collection
- $P@r$ = precision at rank $r$ (fraction of relevant docs in top-$r$)
- $\text{rel}(d_r, q) = 1$ if $|QFT \cap DFT_{d_r}| > 0$, otherwise 0

**In simple terms**: Average of precision values at each rank where a relevant document appears.

## Inputs
- `qft` and `dfts` (Focus Time mode)
- `k` (cutoff)

## Output
- Range: [0, 1], higher is better.

## Examples
```python
from tempoeval.metrics import TemporalMAP

metric = TemporalMAP()
score = metric.compute(qft={2020, 2021}, dfts=[{2020}, {2019}, {2021}], k=3)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
