# Temporal Recall@K

<span class="metric-badge retrieval">Retrieval Metric</span>

Temporal Recall@K measures the fraction of all temporally relevant documents that appear in the top-K retrieved results. This tells you how many of the "correct" temporal documents your system successfully retrieved.

## Formula

$$
\text{TemporalRecall@K} = \frac{|\{d \in D_K : \text{rel}(d, q) = 1\}|}{|\{d \in C : \text{rel}(d, q) = 1\}|}
$$

Where:

- $D_K$ = set of top-K retrieved documents
- $C$ = full document collection
- $\text{rel}(d, q) = 1$ if $|QFT \cap DFT_d| > 0$, otherwise 0

**In simple terms**: What fraction of all temporally relevant documents did you retrieve?

## Inputs
- `retrieved_ids` and `gold_ids` (gold mode)
- `query` and `retrieved_docs` (LLM mode)
- `qft`, `dfts`, and `total_relevant` (Focus Time mode)
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
    "relevance_score": 0.0 to 1.0,
    "reasoning": "brief explanation"
}
```

## Examples
### Focus Time
```python
from tempoeval.metrics import TemporalRecall

metric = TemporalRecall(use_focus_time=True)
score = metric.compute(qft={2020}, dfts=[{2020}, {2019}], total_relevant=1, k=2)
```

### LLM
```python
from tempoeval.metrics import TemporalRecall

metric = TemporalRecall(use_llm=True)
metric.llm = llm
score = await metric.acompute(
    query="When did X happen?",
    retrieved_docs=["..."],
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
