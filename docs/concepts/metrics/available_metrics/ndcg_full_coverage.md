# NDCG Full Coverage

NDCG Full Coverage returns NDCG only when both baseline and comparison periods are covered. Otherwise it returns 0.0.

## Formula
$$
NDCGFullCoverage = NDCG@K if coverage == 1.0 else 0.0
$$

## Inputs
- `query`, `retrieved_ids`, `gold_ids`, and `retrieved_docs`
- `baseline_anchor` and `comparison_anchor`
- `temporal_focus`
- `k` (cutoff)

## Output
- Range: [0, 1], higher is better when full coverage is achieved.

## Prompt (LLM mode)
```text
You are grading retrieved documents for a temporal trend/change query.

Query: "{query}"
Temporal Focus: {temporal_focus}

Baseline anchor period: {baseline_anchor}
Comparison/current anchor period: {comparison_anchor}

Document:
{document}

Decide:
1) covers_baseline: true if the document contains evidence about the BASELINE anchor period.
2) covers_comparison: true if the document contains evidence about the COMPARISON/current anchor period.

Return ONLY valid JSON:
{"covers_baseline": true/false, "covers_comparison": true/false}
```

## Examples
```python
from tempoeval.metrics import NDCGFullCoverage

metric = NDCGFullCoverage()
metric.llm = llm
score = metric.compute(
    query="How has X changed since 2015?",
    retrieved_ids=["doc1", "doc2"],
    gold_ids=["doc1"],
    retrieved_docs=["..."],
    baseline_anchor="2015",
    comparison_anchor="2024",
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
