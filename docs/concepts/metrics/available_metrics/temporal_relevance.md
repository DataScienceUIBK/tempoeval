# Temporal Relevance

Temporal Relevance measures the fraction of retrieved documents that are temporally relevant to the query. It treats all positions equally (no position weighting).

## Formula
$$
TemporalRelevance = (\\text{relevant docs}) / (\\text{retrieved docs})
$$

## Inputs
- `query` and `retrieved_docs`
- `temporal_focus`
- `k` (optional cutoff)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
Judge if a retrieved document helps answer the TEMPORAL aspects of a query.

Query: "{query}"
Temporal Focus: {temporal_focus}

Document:
{document}

Question: Does this document provide information that DIRECTLY helps answer the temporal aspects of the query?

Guidelines:
- Verdict = 1 if document contains temporal information (dates, durations, time periods, temporal sequences)
- Verdict = 0 if document lacks temporal information even if generally relevant
- For "when" queries: document must mention specific times/dates
- For "how long" queries: document must mention durations/time periods
- For "recent" queries: document must mention recency or recent dates
- Be STRICT: generic facts without temporal markers are NOT temporally relevant

Respond ONLY with valid JSON:
{
    "temporal_expressions_found": ["list of temporal expressions found"],
    "relevance_to_query": "high|medium|low|none",
    "verdict": 1 or 0,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}
```

## Examples
```python
from tempoeval.metrics import TemporalRelevance

metric = TemporalRelevance()
metric.llm = llm
score = metric.compute(
    query="When did X happen?",
    retrieved_docs=["..."],
    temporal_focus="specific_time",
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
