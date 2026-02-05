# Computation Modes ⚙️

TempoEval allows you to compute metrics in different ways depending on your resources (time, compute, money) and reliability needs.

## 1. Focus Time Mode (Recommended)

Uses set operations on extracted time intervals.

-   **Logic**: $Relevance = QFT \cap DFT \neq \emptyset$
-   **Pros**:
    -   ⚡ **Fast**: Millions of docs in seconds.
    -   💸 **Free**: No API calls (using Regex or pure Python temporal taggers).
    -   🔬 **Interpretable**: You can see exactly which years matched.
    -   🔄 **Deterministic**: Same input always gives same score.
-   **Cons**:
    -   Requires accurate extraction.

```python
metric = TemporalPrecision(use_focus_time=True)
```

## 2. LLM-as-a-Judge Mode

Uses an LLM to judge each retrieved document individually.

-   **Logic**: Prompt LLM "Is this document temporally relevant to the query?"
-   **Pros**:
    -   🧠 **Flexible**: Captures nuance extraction might miss.
    -   🛠️ **Easy Setup**: No regex tuning needed.
-   **Cons**:
    -   🔴 **Slow**: Latency scales linearly with $N_{docs}$.
    -   $$$ **Expensive**: Significant API costs for large benchmarks.
    -   🎲 **Non-deterministic**: Scores can drift.

```python
metric = TemporalPrecision(use_focus_time=False, use_llm=True)
metric.llm = provider
```

## 3. Gold Labels Mode

Standard Information Retrieval evaluation.

-   **Logic**: Compare retrieved IDs with human-labeled relevant IDs.
-   **Pros**:
    -   Industry Standard ($NDCG$, $MAP$).
-   **Cons**:
    -   ❌ **Binary**: Ignores *why* (temporal vs topical).
    -   📉 **Sparse**: Unlabeled relevant docs are counted as errors.

```python
metric = TemporalPrecision()
metric.compute(retrieved_ids=[...], gold_ids=[...])
```
