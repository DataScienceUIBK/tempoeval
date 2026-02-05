# Temporal NDCG@K

<span class="metric-badge retrieval">Retrieval Metric</span>

Temporal NDCG@K (Normalized Discounted Cumulative Gain) measures ranking quality with graded temporal relevance. Unlike binary relevance, NDCG rewards documents with higher temporal overlap and penalizes poor ranking positions.

## Formula

$$
\text{TemporalNDCG@K} = \frac{DCG@K}{IDCG@K}
$$

Where:

$$
DCG@K = \sum_{i=1}^{K} \frac{\text{rel\_score}(d_i, q)}{\log_2(i + 1)}
$$

$$
\text{rel\_score}(d, q) = \frac{|QFT \cap DFT_d|}{|QFT \cup DFT_d|}
$$

And $IDCG@K$ is the DCG of the ideal ranking (documents sorted by relevance score).

**In Focus Time mode**: Relevance is Jaccard similarity scaled to [0, 4] for compatibility with standard NDCG grading.

**In simple terms**: How well are temporally relevant documents ranked? (1.0 = perfect ranking, 0.0 = worst ranking)

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
Rate the temporal relevance of this document to the query on a scale of 0-4.

## Query
{query}

## Document
{document}

## Grading Scale (G-EVAL style)
- 4: Perfect - Contains exact temporal information needed to fully answer the query
- 3: Highly Relevant - Contains most temporal information, minor gaps
- 2: Partially Relevant - Contains some temporal context but incomplete
- 1: Marginally Relevant - Mentions related time periods but doesn't directly answer
- 0: Not Relevant - No useful temporal information for this query

## Output (JSON)
{
    "relevance_score": 0 to 4,
    "reasoning": "brief explanation of temporal relevance"
}
```

## Examples
### Focus Time
```python
from tempoeval.metrics import TemporalNDCG

metric = TemporalNDCG(use_focus_time=True)
score = metric.compute(qft={2020, 2021}, dfts=[{2020, 2021}, {2019}], k=2)
```

### LLM
```python
from tempoeval.metrics import TemporalNDCG

metric = TemporalNDCG(use_llm=True)
metric.llm = llm
score = await metric.acompute(
    query="When did X happen?",
    retrieved_docs=["..."],
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
