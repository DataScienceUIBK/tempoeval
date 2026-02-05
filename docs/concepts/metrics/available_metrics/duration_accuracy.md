# Duration Accuracy

Duration Accuracy evaluates whether duration or time span claims in the answer are correct. It supports LLM or regex-based verification.

## Inputs
- `answer`
- `gold_durations` (optional)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Evaluate the correctness of duration/time span claims in the answer.

## Answer
{answer}

## Gold Durations (if provided)
{gold_durations}

## Instructions
1. Extract all duration claims from the answer (e.g., "lasted 5 years", "for 3 months")
2. Compare with gold durations if provided
3. Check if durations are consistent with any dates mentioned

## Output (JSON)
{
    "duration_claims": [
        {
            "claim": "duration claim text",
            "extracted_duration": "5 years",
            "associated_event": "event this duration refers to",
            "gold_duration": "actual duration if known",
            "is_correct": true or false,
            "error_magnitude": 0
        }
    ],
    "total_claims": 0,
    "correct_claims": 0,
    "accuracy_score": 0.0
}
```

## Examples
```python
from tempoeval.metrics import DurationAccuracy

metric = DurationAccuracy(use_llm=True)
metric.llm = llm
score = metric.compute(answer="...", gold_durations={"Event": "5 years"})
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
