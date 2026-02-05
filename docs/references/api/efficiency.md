# Efficiency & Cost Tracking

Track the latency and cost of your evaluation pipeline.

## Usage

```python
from tempoeval.efficiency import EfficiencyTracker

tracker = EfficiencyTracker()

with tracker.track("retrieval"):
    # Run retrieval
    results = retriever.retrieve(...)

with tracker.track("evaluation"):
    # Run heavy LLM metric
    metric.compute(...)

print(tracker.report())
```

## Report Format

```json
{
  "retrieval": {
    "latency": 1.25,
    "cost": 0.0
  },
  "evaluation": {
    "latency": 45.0,
    "cost": 0.15,
    "tokens": 15000
  }
}
```
