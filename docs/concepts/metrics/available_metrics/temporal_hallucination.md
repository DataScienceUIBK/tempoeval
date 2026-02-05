# Temporal Hallucination

Temporal Hallucination measures the proportion of temporal claims that are fabricated or unsupported. Lower is better.

## Inputs
- `answer`
- `contexts` (optional)
- `gold_answer` (optional)

## Output
- Range: [0, 1], lower is better.

## Prompt (LLM mode)
```text
## Task
Identify temporal hallucinations in the generated answer.

A temporal hallucination is a temporal claim that is:
1. NOT supported by the provided context, AND
2. Likely factually incorrect or fabricated

## Context
{context}

## Generated Answer
{answer}

## Gold Answer (if available)
{gold_answer}

## Instructions
For each temporal claim in the answer:
1. Check if it appears in the context
2. If not in context, assess if it's plausibly correct or fabricated
3. Cross-reference with gold answer if available

## Output (JSON)
{
    "temporal_claims": [
        {
            "claim": "the temporal claim",
            "in_context": true or false,
            "likely_fabricated": true or false,
            "contradicts_gold": true or false,
            "is_hallucination": true or false,
            "reason": "explanation"
        }
    ],
    "total_claims": 0,
    "hallucinated_claims": ["list of hallucinated claims"],
    "hallucination_rate": 0.0
}
```

## Examples
```python
from tempoeval.metrics import TemporalHallucination

metric = TemporalHallucination()
metric.llm = llm
score = await metric.acompute(
    contexts=["..."],
    answer="...",
    gold_answer="..."
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
