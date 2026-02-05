# Reasoning Metrics

Detailed reference for reasoning evaluation metrics (Layer 3).

## EventOrdering

Evaluates whether the chronological order of events described in the answer is correct relative to the known ground truth or logical sequence.

```python
from tempoeval.metrics import EventOrdering
metric = EventOrdering()
score = metric.compute(answer="First X happened, then Y...", gold_order=["X", "Y"])
```

---

## DurationAccuracy

Evaluates whether stated durations (e.g., "The war lasted 5 years") are factually correct.

```python
from tempoeval.metrics import DurationAccuracy
score = metric.compute(answer="It lasted 5 years", gold_duration="5 years")
```

---

## CrossPeriodReasoning

Evaluates the model's ability to synthesize information across different time periods. Tests for:
1.  **Trend Analysis**: "Prices increased..."
2.  **Comparison**: "X in 1990 vs Y in 2000"

```python
from tempoeval.metrics import CrossPeriodReasoning
score = metric.compute(query="Compare X in 1990 and 2000", answer="...")
```

---

## TemporalConsistency

Evaluates whether the answer is robust and consistent across multiple generations (Sampling).

```python
from tempoeval.metrics import TemporalConsistency
score = metric.compute(answers=["Answer 1...", "Answer 2..."])
```
