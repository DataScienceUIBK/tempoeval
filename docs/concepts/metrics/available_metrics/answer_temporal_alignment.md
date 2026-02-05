# Answer Temporal Alignment

Answer Temporal Alignment measures how well the answer's temporal scope matches the query's intent.

## Inputs
- `query`
- `answer`
- `temporal_focus`
- `target_anchors` or `key_time_anchors`

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Evaluate how well the answer's temporal focus aligns with the query's temporal intent.

## Query
{query}

## Answer
{answer}

## Expected Time Focus
Target anchors: {target_anchors}
Temporal focus type: {temporal_focus}

## Evaluation Criteria
1. Does the answer address the specific time period(s) asked about?
2. Are the time references in the answer related to the query's temporal scope?
3. Does the answer avoid discussing irrelevant time periods?

## Output (JSON)
{
    "query_time_focus": ["list of time periods the query asks about"],
    "answer_time_focus": ["list of time periods the answer discusses"],
    "overlap_analysis": {
        "matching_periods": ["periods that align"],
        "missing_periods": ["query periods not addressed"],
        "extra_periods": ["answer periods not in query scope"]
    },
    "alignment_score": 0.0,
    "reason": "explanation of alignment assessment"
}
```

## Examples
```python
from tempoeval.metrics import AnswerTemporalAlignment

metric = AnswerTemporalAlignment()
metric.llm = llm
score = await metric.acompute(
    query="What happened in 2020?",
    answer="...",
    temporal_focus="specific_time",
    target_anchors=["2020"]
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
