# Temporal Coverage

Temporal Coverage measures whether retrieved documents cover the baseline and comparison periods required for trend or change queries.

## Formula
$$
TemporalCoverage@K = (baseline_covered + comparison_covered) / 2
$$

## Inputs
- `query` and `retrieved_docs`
- `baseline_anchor` and `comparison_anchor`
- `temporal_focus` (should be change over time)
- `k` (cutoff)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
You are grading retrieved documents for a temporal trend/change query that needs cross-period evidence.

Query: "{query}"
Temporal Focus: {temporal_focus}

Baseline anchor period: {baseline_anchor}
Comparison/current anchor period: {comparison_anchor}

Document:
{document}

Decide:
1) verdict: 1 if the document DIRECTLY helps answer the temporal aspects of the query (contains relevant temporal info), else 0.
2) covers_baseline: true if the document contains evidence about the BASELINE anchor period.
3) covers_comparison: true if the document contains evidence about the COMPARISON/current anchor period.

Strictness rules:
- A random date not related to the anchors does NOT count.
- "Currently/as of 2023/recent years" can count for comparison coverage if relevant.
- Baseline coverage should connect to the baseline anchor period (e.g., around 2017).

Return ONLY valid JSON:
{
  "verdict": 1 or 0,
  "reason": "brief explanation",
  "temporal_contribution": "what temporal information provided, or 'none'",
  "covers_baseline": true or false,
  "covers_comparison": true or false
}
```

## Examples
```python
from tempoeval.metrics import TemporalCoverage

metric = TemporalCoverage()
metric.llm = llm
score = await metric.acompute(
    query="How has X changed since 2017?",
    retrieved_docs=["..."],
    baseline_anchor="2017",
    comparison_anchor="2024",
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
