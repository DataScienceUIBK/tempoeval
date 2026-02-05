# Temporal MRR

<span class="metric-badge retrieval">Retrieval Metric</span>

Temporal Mean Reciprocal Rank (MRR) measures how quickly the first temporally relevant document appears in the ranking. A higher rank (closer to position 1) produces a higher score.

## Formula

$$
\text{TemporalMRR} = \frac{1}{\min\{r : \text{rel}(d_r, q) = 1\}}
$$

Where:

- $r$ = rank position (1-indexed)
- $d_r$ = document at rank $r$
- $\text{rel}(d_r, q) = 1$ if $|QFT \cap DFT_{d_r}| > 0$, otherwise 0

**In simple terms**: If the first relevant document is at rank 1, MRR = 1.0. At rank 2, MRR = 0.5. At rank 3, MRR = 0.33, etc.

## Inputs
- `retrieved_ids` and `gold_ids` (gold mode)
- `query` and `retrieved_docs` (LLM mode)
- `qft` and `dfts` (Focus Time mode)
- `k` (cutoff)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Judge if this document is temporally relevant to answering the query.

## Query
{query}

## Document
{document}

## Criteria
A document is temporally relevant if it:
1. Contains temporal information (dates, periods, durations) needed to answer the query
2. Discusses events/facts from the time period the query asks about
3. Provides temporal context that helps answer the query

## Output (JSON)
{
    "is_relevant": true or false,
    "confidence": 0.0 to 1.0
}
```

## Examples
### Focus Time
```python
from tempoeval.metrics import TemporalMRR

metric = TemporalMRR(use_focus_time=True)
score = metric.compute(qft={2020}, dfts=[{2019}, {2020}], k=2)
```

### LLM
```python
from tempoeval.metrics import TemporalMRR

metric = TemporalMRR(use_llm=True)
metric.llm = llm
score = await metric.acompute(
    query="When did X happen?",
    retrieved_docs=["..."],
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
