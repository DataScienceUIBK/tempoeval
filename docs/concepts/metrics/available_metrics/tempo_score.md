# TempoScore

TempoScore is a unified composite metric that combines retrieval and generation quality into a single score for evaluating Temporal RAG systems.

## Three Aggregation Methods

TempoScore provides **three different scoring methods**:

| Method | Key | Description |
|--------|-----|-------------|
| **Weighted Average** | `tempo_weighted` | Simple weighted sum. Robust, never NaN. |
| **Harmonic Mean** | `tempo_harmonic` | Penalizes low values. Strict balance. |
| **Layered F-Score** | `tempo_f_score` | Hierarchical F1. ⭐ **Recommended** |

## Output Structure

```python
{
    'tempo_weighted': 0.8125,   # Method 1: Weighted Average
    'tempo_harmonic': 0.7983,   # Method 2: Harmonic Mean
    'tempo_f_score': 0.8324,    # Method 3: Layered F-Score (recommended)
    'retrieval_f1': 0.7467,     # F1(Precision, Recall)
    'generation_f1': 0.8727     # F1(Faithfulness, Coherence)
}
```

## Components

| Layer | Metrics | What it measures |
|-------|---------|-----------------|
| **Retrieval** | `temporal_precision`, `temporal_recall` | Did we find the right time periods? |
| **Generation** | `temporal_faithfulness`, `temporal_coherence` | Is the answer temporally accurate? |

## Formulas

### 1. Weighted Average

$$
TempoScore_{weighted} = \frac{\sum w_i \cdot x_i}{\sum w_i}
$$

### 2. Harmonic Mean

$$
TempoScore_{harmonic} = \frac{\sum w_i}{\sum \frac{w_i}{x_i}}
$$

### 3. Layered F-Score (Recommended)

$$
Retrieval_{F1} = F_1(Precision, Recall)
$$

$$
Generation_{F1} = F_1(Faithfulness, Coherence)
$$

$$
TempoScore_{fscore} = F_1(Retrieval_{F1}, Generation_{F1})
$$

## Usage

```python
from tempoeval.metrics import TempoScore

metric = TempoScore()
result = metric.compute(
    temporal_precision=0.8,
    temporal_recall=0.7,
    temporal_faithfulness=0.9,
    temporal_coherence=0.85
)

# Access individual scores
print(f"Weighted: {result['tempo_weighted']}")
print(f"Harmonic: {result['tempo_harmonic']}")
print(f"F-Score:  {result['tempo_f_score']}")
```

## Factory Methods

```python
# Retrieval-focused (for search systems)
metric = TempoScore.create_retrieval_focused()

# Generation-focused (for LLM-heavy systems)
metric = TempoScore.create_generation_focused()

# Balanced (default)
metric = TempoScore.create_balanced()
```
