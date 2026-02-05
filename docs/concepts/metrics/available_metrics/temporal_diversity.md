# Temporal Diversity

Temporal Diversity measures how many distinct time periods are covered by the retrieved documents.

## Formula
$$
Diversity@K = unique_periods / expected_periods
$$

In Focus Time mode, unique_periods is the number of unique years in union(DFT_1..K).

## Inputs
- `retrieved_docs` (LLM or regex mode)
- `dfts` (Focus Time mode)
- `total_scope` or `expected_periods` (optional)
- `k` (cutoff)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Identify distinct HIGHLIGHTS of the timeline in these documents.

## Documents
{documents}

## Instructions
Extract distinct historical eras or time clusters.
Cluster close dates together (e.g., "2018, 2019" -> "Late 2010s").
Do not list every single year. We want to see DIVERSITY of coverage.

Examples of distinct periods:
- "The Depression Era" vs "Post-War Boom" (Distinct)
- "2018" vs "2019" (Not distinct, same era)

## Output (JSON)
{
    "periods": [
        {"name": "Late 2010s", "representative_year": 2018},
        {"name": "Early 20th Century", "representative_year": 1910}
    ],
    "unique_years": [2018, 1910]
}
```

## Examples
```python
from tempoeval.metrics import TemporalDiversity

metric = TemporalDiversity(use_llm=True)
metric.llm = llm
score = metric.compute(retrieved_docs=["..."], k=5)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
