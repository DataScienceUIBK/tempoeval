# Temporal Coherence

Temporal Coherence evaluates whether the answer is internally consistent in time (no contradictions or impossible timelines).

## Inputs
- `answer`

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Critically evaluate the temporal coherence of this answer.

## Answer
{answer}

## Step 1: Reconstruct Timeline
First, extract every event and its associated time. List them chronologically.
If the timeline is impossible to construct (jumps around, contradictory), note it.

## Step 2: Check for Issues
Look for:
1. **Contradictions**: Is the same event given two different dates?
2. **Causality Violations**: Does an effect happen before its cause?
3. **Anachronisms**: Are technologies/concepts mentioned before they existed?

## Scoring Guide
- 1.0: Perfect timeline, clear causal flow.
- 0.7: Minor ambiguities but generally consistent.
- 0.4: Confusing timeline or minor contradictions.
- 0.0: Impossible sequence (e.g., born after death) or major contradictions.

## Output (JSON)
{
    "timeline_reconstruction": [
        {"time": "1990", "event": "Event A"},
        {"time": "1995", "event": "Event B"}
    ],
    "coherence_issues": [
        {
            "issue": "Event B (1995) causes Event A (1990)",
            "severity": "critical"
        }
    ],
    "coherence_score": 0.0
}
```

## Examples
```python
from tempoeval.metrics import TemporalCoherence

metric = TemporalCoherence()
metric.llm = llm
score = await metric.acompute(answer="...")
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
