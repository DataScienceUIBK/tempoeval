# Cross Period Reasoning

Cross Period Reasoning evaluates how well the answer synthesizes information across time periods for trend or change queries.

## Inputs
- `query`
- `answer`
- `baseline_anchor` and `comparison_anchor`

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Evaluate how well this answer synthesizes information across time periods.

## Query
{query}

## Answer
{answer}

## Time Periods to Cover
- Baseline: {baseline_anchor}
- Comparison: {comparison_anchor}

## Evaluation Criteria

1. **Period Coverage** (0-1): Does the answer discuss both periods?
2. **Explicit Comparison** (0-1): Is change/difference explicitly stated?
3. **Quantification** (0-1): Are changes quantified (numbers, percentages)?
4. **Causal Reasoning** (0-1): Are reasons for changes explained?
5. **Temporal Clarity** (0-1): Are time references unambiguous?

## Output (JSON)
{
    "baseline_addressed": true or false,
    "comparison_addressed": true or false,
    "explicit_comparison_made": true or false,
    "quantified_changes": ["list of quantified changes"],
    "causal_explanations": ["list of causal explanations"],
    "criteria_scores": {
        "period_coverage": 0.0,
        "explicit_comparison": 0.0,
        "quantification": 0.0,
        "causal_reasoning": 0.0,
        "temporal_clarity": 0.0
    },
    "cross_period_score": 0.0
}
```

## Examples
```python
from tempoeval.metrics import CrossPeriodReasoning

metric = CrossPeriodReasoning()
metric.llm = llm
score = await metric.acompute(
    query="How has X changed since 2010?",
    answer="...",
    baseline_anchor="2010",
    comparison_anchor="2020"
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
