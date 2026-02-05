# Welcome to TempoEval

<div class="hero-banner" markdown>

# Evaluate Temporal Reasoning in RAG Systems

**TempoEval** is a comprehensive framework for evaluating the temporal reasoning capabilities of RAG (Retrieval-Augmented Generation) systems with 20+ specialized metrics.

[Get Started](getstarted/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/tempo26/TempoEval){ .md-button }

</div>

---

## Why TempoEval?

Traditional RAG evaluation metrics (like precision/recall on text overlap) fail to capture **time**—a critical dimension for many real-world applications.

!!! example "The Temporal Gap"

    **Query**: *"Who was president in 1999?"*

    - **Retrieved**: "President Clinton" ✅
    - **Retrieved**: "President Bush" ❌

    Both are semantically similar (both presidents), but **temporally distinct**. Standard metrics can't tell the difference.

### The Problem with Current Metrics

<div class="grid cards" markdown>

-   :material-alert-circle:{ .lg .middle } **Wrong Time Period**

    ---

    Standard metrics score high if keywords match, even when retrieving documents from the wrong year.

-   :material-calendar-remove:{ .lg .middle } **Date Hallucinations**

    ---

    Traditional evaluation often misses fabricated dates in generated answers.

-   :material-swap-horizontal:{ .lg .middle } **Temporal Confusion**

    ---

    Confusing "before" and "after" or getting event ordering wrong goes undetected.

-   :material-currency-usd-off:{ .lg .middle } **Expensive LLM Calls**

    ---

    LLM-as-judge is slow, costly, and non-deterministic for temporal evaluation.

</div>

---

## Key Features

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Focus Time Extraction**

    ---

    Extract the temporal "focus" of queries, documents, and answers using Regex, HeidelTime, or LLMs.

    [:octicons-arrow-right-24: Learn about Focus Time](concepts/focus_time.md)

-   :material-layers-triple:{ .lg .middle } **Multi-Layer Metrics**

    ---

    Evaluate across three layers: Retrieval (12 metrics), Generation (5 metrics), and Reasoning (4 metrics).

    [:octicons-arrow-right-24: Explore Metrics](concepts/metrics/available_metrics/index.md)

-   :material-chart-line:{ .lg .middle } **TempoScore**

    ---

    A single composite score combining precision, recall, faithfulness, and coherence.

    [:octicons-arrow-right-24: TempoScore Details](concepts/metrics/available_metrics/tempo_score.md)

-   :material-database:{ .lg .middle } **Benchmark Datasets**

    ---

    Built-in support for TEMPO, TimeQA, TimeBench, and SituatedQA datasets.

    [:octicons-arrow-right-24: Work with Datasets](tutorials/datasets.md)

-   :material-code-braces:{ .lg .middle } **Easy Integration**

    ---

    Simple API that works with any retriever or LLM. Three modes: Focus Time, LLM-as-judge, or Gold labels.

    [:octicons-arrow-right-24: Quick Start](getstarted/quickstart.md)

-   :material-test-tube:{ .lg .middle } **Pre-built Experiments**

    ---

    Benchmark your system instantly with our experiment suite and reproduce published results.

    [:octicons-arrow-right-24: Run Experiments](experiments/index.md)

</div>

---

## Quick Example

```python
from tempoeval.core import extract_qft, extract_dft
from tempoeval.metrics import TemporalPrecision

# Define query and documents
query = "What happened during the 2008 financial crisis?"
documents = [
    "The collapse of Lehman Brothers in 2008 triggered...",  # ✅ Relevant (2008)
    "The COVID-19 pandemic started in 2019...",              # ❌ Irrelevant (2019)
]

# Extract Focus Times
qft = extract_qft(query)  # {2008}
dfts = [extract_dft(doc) for doc in documents]  # [{2008}, {2019}]

# Compute Metric
metric = TemporalPrecision(use_focus_time=True)
score = metric.compute(qft=qft, dfts=dfts, k=2)

print(f"Temporal Precision@2: {score}")  # 0.5 (1/2 relevant)
```

---

## Three Evaluation Modes

| Mode | Input | Pros | Cons |
|------|-------|------|------|
| **Focus Time** | QFT & DFT | ⚡ Fast, 💰 Free, 🎯 Interpretable | Requires extraction |
| **LLM-as-Judge** | Query & Docs | 🧠 Flexible, No gold labels needed | 🐌 Slow, 💸 Expensive |
| **Gold Labels** | IDs & Gold | 📊 Standard IR benchmark | Requires annotations |

---

## Community & Support

<div class="grid cards" markdown>

-   :fontawesome-brands-github:{ .lg .middle } **GitHub**

    ---

    Star us on GitHub and contribute to the project.

    [:octicons-arrow-right-24: tempo26/TempoEval](https://github.com/tempo26/TempoEval)

-   :fontawesome-brands-python:{ .lg .middle } **PyPI**

    ---

    Install via pip and start evaluating in minutes.

    [:octicons-arrow-right-24: PyPI Package](https://pypi.org/project/tempoeval/)

-   :fontawesome-solid-book:{ .lg .middle } **Documentation**

    ---

    Comprehensive guides, tutorials, and API reference.

    [:octicons-arrow-right-24: Get Started](getstarted/index.md)

-   :fontawesome-solid-paper-plane:{ .lg .middle } **Contact**

    ---

    Have questions? Reach out to the team.

    [:octicons-arrow-right-24: Contact Us](mailto:contact@tempoeval.org)

</div>

---

## What's Next?

1. **[Install TempoEval](getstarted/installation.md)** - Get up and running in 2 minutes
2. **[Quick Start](getstarted/quickstart.md)** - Your first evaluation in 5 minutes
3. **[Understand Focus Time](concepts/focus_time.md)** - Learn the core concept
4. **[Explore Metrics](concepts/metrics/available_metrics/index.md)** - Choose the right metrics for your use case
