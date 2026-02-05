# TempoScore (Composite Metric)

TempoScore is the unified composite metric for evaluating Temporal RAG systems.

## Overview

TempoScore provides **three different aggregation methods** to compute an overall quality score:

| Method | Key | Description | Best For |
|--------|-----|-------------|----------|
| **Weighted Average** | `tempo_weighted` | Simple weighted sum | Robust, no NaN |
| **Harmonic Mean** | `tempo_harmonic` | Penalizes low values | Strict balance |
| **Layered F-Score** | `tempo_f_score` | Hierarchical F1 | **Recommended** |

## Usage

```python
from tempoeval.metrics.composite import TempoScore

metric = TempoScore()
result = metric.compute(
    temporal_precision=0.8,
    temporal_recall=0.7,
    temporal_faithfulness=0.9,
    temporal_coherence=0.85
)

print(result)
# {
#     'tempo_weighted': 0.8125,
#     'tempo_harmonic': 0.7983,
#     'tempo_f_score': 0.8324,
#     'retrieval_f1': 0.7467,
#     'generation_f1': 0.8727
# }
```

## The Three Methods

### 1. Weighted Average (`tempo_weighted`)
Simple weighted sum. Most robust, never produces NaN.

```
tempo_weighted = Σ(weight_i × score_i) / Σ(weights)
```

### 2. Harmonic Mean (`tempo_harmonic`)
Penalizes when any component is low. Can return 0 if any component is 0.

```
tempo_harmonic = Σ(weights) / Σ(weight_i / score_i)
```

### 3. Layered F-Score (`tempo_f_score`) ⭐ Recommended
Hierarchical F1-based approach:

1. **Retrieval F1** = F1(Precision, Recall)
2. **Generation F1** = F1(Faithfulness, Coherence)
3. **TempoScore** = F1(Retrieval_F1, Generation_F1)

This ensures:
- Good retrieval AND good generation are required for a high score
- Interpretable breakdown: see exactly which layer failed
- Never produces NaN

## Customization

```python
# Retrieval-focused (for search systems)
metric = TempoScore.create_retrieval_focused()

# Generation-focused (for LLM-heavy systems)
metric = TempoScore.create_generation_focused()

# Balanced (default)
metric = TempoScore.create_balanced()
```

## Component Metrics

| Layer | Metrics | What it measures |
|-------|---------|-----------------|
| **Retrieval** | `temporal_precision`, `temporal_recall` | Did we find the right time periods? |
| **Generation** | `temporal_faithfulness`, `temporal_coherence` | Is the answer temporally accurate? |
