# Quick Start

This guide will walk you through your first evaluation with TempoEval.

## Prerequisites
Ensure you have installed the package:
```bash
pip install tempoeval
```

---

## 5-Minute "Hello World"

We will evaluate a simple scenario:
*   **Query**: "What happened in 2020?"
*   **Doc A**: "The COVID-19 pandemic began in 2020." (Relevant)
*   **Doc B**: "The 2008 financial crisis was severe." (Irrelevant)

### Step 1: Import Components

```python
from tempoeval.core import extract_qft, extract_dft
from tempoeval.metrics import TemporalPrecision
```

### Step 2: Extract Focus Time

First, we need to understand *what time* the texts are about. We use `extract_qft` (Query Focus Time) and `extract_dft` (Document Focus Time).

```python
query = "What happened in 2020?"
documents = [
    "The COVID-19 pandemic began in 2020.",
    "The 2008 financial crisis was severe."
]

# Extract
qft = extract_qft(query)
print(f"Query Time: {qft}") 
# Output: FocusTime(years={2020})

dfts = [extract_dft(doc) for doc in documents]
print(f"Doc Times: {dfts}") 
# Output: [FocusTime(years={2020}), FocusTime(years={2008})]
```

### Step 3: Compute Metric

Now we verify if the retrieved documents match the query's time.

```python
# Initialize Metric
metric = TemporalPrecision(use_focus_time=True)

# Compute
score = metric.compute(
    qft=qft, 
    dfts=dfts, 
    k=2  # Evaluated at top 2 documents
)

print(f"Precision@2: {score}")
# Output: 0.5 (Because 1 out of 2 docs was relevant)
```

---

## What just happened?

1.  **Extraction**: The library found "2020" in the query and the first document.
2.  **Intersection**: It compared `{2020}` vs `{2020}` (Match!) and `{2020}` vs `{2008}` (No Match).
3.  **Scoring**: It calculated Precision = (Matches / Total) = 1/2 = 0.5.

## Next Steps

Now that you understand the basics, try evaluating a full RAG pipeline.

[➡️ Evaluate Retrieval Tutorial](../tutorials/retrieval.md)
