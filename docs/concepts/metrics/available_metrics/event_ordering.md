# Event Ordering

Event Ordering measures whether the temporal sequence of events in the answer matches the correct order.

## Inputs
- `answer`
- `gold_order` (optional)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Extract the temporal sequence of events from this answer and verify ordering.

## Answer
{answer}

## Gold Event Order (if provided)
{gold_order}

## Instructions
1. Extract all events mentioned in chronological order as stated in the answer
2. If gold order is provided, compare with it
3. If no gold order, assess internal consistency of the ordering

## Output (JSON)
{
    "extracted_events": [
        {"event": "event description", "stated_order": 1, "temporal_marker": "first/then/after/etc"}
    ],
    "extracted_order": ["Event A", "Event B", "Event C"],
    "gold_order": ["Event A", "Event B", "Event C"],
    "order_matches": true or false,
    "ordering_score": 0.0,
    "ordering_errors": [
        {"expected": "A before B", "actual": "B before A"}
    ]
}
```

## Examples
```python
from tempoeval.metrics import EventOrdering

metric = EventOrdering()
metric.llm = llm
score = await metric.acompute(
    answer="First A, then B.",
    gold_order=["A", "B"]
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
