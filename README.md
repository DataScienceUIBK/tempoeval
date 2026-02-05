<p align="center">
  <img src="https://img.shields.io/badge/🕐_TempoEval-Temporal_RAG_Evaluation-4B8BBE?style=for-the-badge&labelColor=306998" alt="TempoEval"/>
</p>

<h1 align="center">⏱️ TempoEval</h1>

<p align="center">
  <strong>A Comprehensive Framework for Evaluating Temporal Reasoning in RAG Systems</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/tempoeval/"><img src="https://img.shields.io/pypi/v/tempoeval?style=flat-square&logo=pypi&logoColor=white&color=blue" alt="PyPI Version"/></a>
  <a href="https://github.com/DataScienceUIBK/tempoeval"><img src="https://img.shields.io/github/stars/DataScienceUIBK/tempoeval?style=flat-square&logo=github&color=yellow" alt="GitHub Stars"/></a>
  <a href="https://github.com/DataScienceUIBK/tempoeval/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version"/></a>
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv"/></a>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-metrics">Metrics</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 🎯 Overview

**TempoEval** is a state-of-the-art evaluation framework designed specifically for assessing temporal reasoning capabilities in Retrieval-Augmented Generation (RAG) systems. Unlike traditional metrics that only measure relevance, TempoEval provides **16 specialized metrics** that evaluate how well your RAG system understands, retrieves, and generates temporally accurate content.

<p align="center">
  <img src="https://img.shields.io/badge/16-Temporal_Metrics-blue?style=for-the-badge" alt="16 Metrics"/>
  <img src="https://img.shields.io/badge/3-Evaluation_Layers-green?style=for-the-badge" alt="3 Layers"/>
  <img src="https://img.shields.io/badge/Focus_Time-Extraction-orange?style=for-the-badge" alt="Focus Time"/>
</p>

### 🤔 Why TempoEval?

Traditional RAG evaluation metrics fail to capture temporal nuances:

| Scenario | Traditional Metrics | TempoEval |
|----------|-------------------|-----------|
| Query: "What happened in 2020?" → Retrieved doc about 2019 | ✅ High similarity | ❌ Low temporal precision |
| Answer mentions dates not in context | ✅ Fluent text | ❌ Temporal hallucination detected |
| Cross-period query needs docs from multiple eras | ❌ Partial coverage | ✅ Full temporal coverage measured |

---

## ✨ Key Features

### 📊 Three-Layer Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: REASONING METRICS                                     │
│  └─ Event Ordering • Duration Accuracy • Cross-Period Reasoning │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: GENERATION METRICS                                    │
│  └─ Faithfulness • Hallucination • Coherence • Alignment        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: RETRIEVAL METRICS                                     │
│  └─ Precision • Recall • NDCG • Coverage • Diversity • MRR      │
└─────────────────────────────────────────────────────────────────┘
```

### 🔑 Core Capabilities

| Feature | Description |
|---------|-------------|
| 🎯 **Focus Time Extraction** | Automatically extract temporal focus from queries and documents |
| 📈 **16 Specialized Metrics** | Comprehensive temporal evaluation across retrieval, generation, and reasoning |
| 🤖 **LLM-as-Judge** | Use GPT-4, Claude, or other LLMs for nuanced temporal assessment |
| ⚡ **Dual-Mode Evaluation** | Rule-based (fast) or LLM-based (accurate) metric computation |
| 📊 **TempoScore** | Unified composite score combining all temporal dimensions |
| 💰 **Cost Tracking** | Built-in efficiency monitoring for latency and API costs |
| 📦 **TEMPO Benchmark** | Integrated support for the TEMPO temporal QA benchmark |

---

## 📦 Installation

### Via pip (Recommended)

```bash
pip install tempoeval
```

### From Source

```bash
git clone https://github.com/DataScienceUIBK/tempoeval.git
cd tempoeval
pip install -e .
```

### Optional Dependencies

```bash
# For LLM-based evaluation (recommended)
pip install openai anthropic

# For BM25 retrieval in examples
pip install gensim pyserini

# For TEMPO benchmark loading
pip install datasets huggingface_hub pyarrow
```

---

## 🚀 Quick Start

### Basic Retrieval Evaluation (No LLM Required)

```python
from tempoeval.metrics import TemporalRecall, TemporalNDCG, TemporalPrecision
from tempoeval.core import FocusTime

# Your retrieval results
retrieved_ids = ["doc_2020", "doc_2019", "doc_2021"]
gold_ids = ["doc_2020", "doc_2021"]

# Compute metrics
recall = TemporalRecall().compute(retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=5)
ndcg = TemporalNDCG().compute(retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=5)

print(f"Temporal Recall@5: {recall:.3f}")
print(f"Temporal NDCG@5: {ndcg:.3f}")
```

### Focus Time-Based Evaluation

```python
from tempoeval.core import FocusTime, extract_qft, extract_dft
from tempoeval.metrics import TemporalPrecision

# Extract Focus Time from query
query = "What happened to Bitcoin in 2017?"
qft = extract_qft(query)  # FocusTime(years={2017})

# Extract Focus Time from documents
documents = [
    "Bitcoin reached $20,000 in December 2017.",
    "Ethereum launched in 2015.",
    "The SegWit upgrade activated in August 2017.",
]
dfts = [extract_dft(doc) for doc in documents]

# Evaluate temporal precision
precision = TemporalPrecision(use_focus_time=True)
score = precision.compute(qft=qft, dfts=dfts, k=3)
print(f"Temporal Precision@3: {score:.3f}")
```

### LLM-Based Generation Evaluation

```python
import os
from tempoeval.llm import AzureOpenAIProvider
from tempoeval.metrics import (
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    TempoScore
)

# Configure LLM
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-endpoint.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"

llm = AzureOpenAIProvider()

# Your RAG output
query = "When was Bitcoin pruning introduced?"
contexts = ["Bitcoin Core 0.11.0 was released on July 12, 2015 with pruning support."]
answer = "Bitcoin pruning was introduced in version 0.11.0, released on July 12, 2015."

# Evaluate generation quality
faithfulness = TemporalFaithfulness(llm=llm)
hallucination = TemporalHallucination(llm=llm)
coherence = TemporalCoherence(llm=llm)

print(f"Faithfulness: {faithfulness.compute(answer=answer, contexts=contexts):.3f}")
print(f"Hallucination: {hallucination.compute(answer=answer, contexts=contexts):.3f}")
print(f"Coherence: {coherence.compute(answer=answer):.3f}")

# Compute unified TempoScore
tempo_scorer = TempoScore()
result = tempo_scorer.compute(
    temporal_precision=0.9,
    temporal_recall=0.85,
    temporal_faithfulness=1.0,
    temporal_coherence=1.0
)
print(f"\n🎯 TempoScore: {result['tempo_weighted']:.3f}")
```

---

## 📊 Metrics

### Layer 1: Retrieval Metrics

| Metric | Description | LLM Required |
|--------|-------------|--------------|
| `TemporalPrecision` | % of retrieved docs matching query's temporal focus | Optional |
| `TemporalRecall` | % of relevant temporal docs retrieved | Optional |
| `TemporalNDCG` | Ranking quality with temporal relevance grading | No |
| `TemporalMRR` | Reciprocal rank of first temporally relevant doc | No |
| `TemporalCoverage` | Coverage of required time periods (cross-period) | Yes |
| `TemporalDiversity` | Variety of time periods in retrieved docs | Optional |
| `AnchorCoverage` | Coverage of key temporal anchors | Optional |

### Layer 2: Generation Metrics

| Metric | Description | LLM Required |
|--------|-------------|--------------|
| `TemporalFaithfulness` | Are temporal claims supported by context? | Yes |
| `TemporalHallucination` | % of fabricated temporal information | Yes |
| `TemporalCoherence` | Internal consistency of temporal statements | Yes |
| `AnswerTemporalAlignment` | Does answer focus on the right time period? | Yes |

### Layer 3: Reasoning Metrics

| Metric | Description | LLM Required |
|--------|-------------|--------------|
| `EventOrdering` | Correctness of event sequence | Yes |
| `DurationAccuracy` | Accuracy of duration/interval claims | Yes |
| `CrossPeriodReasoning` | Quality of comparison across time periods | Yes |

### Composite Metrics

| Metric | Description |
|--------|-------------|
| `TempoScore` | Unified score combining all temporal dimensions |

---

## 📁 Project Structure

```
tempoeval/
├── 📦 core/                    # Core components
│   ├── focus_time.py          # Focus Time extraction
│   ├── evaluator.py           # Main evaluation orchestrator
│   ├── config.py              # Configuration management
│   └── result.py              # Result containers
├── 📊 metrics/                 # All 16 metrics
│   ├── retrieval/             # Layer 1 metrics
│   ├── generation/            # Layer 2 metrics
│   ├── reasoning/             # Layer 3 metrics
│   └── composite/             # TempoScore
├── 🤖 llm/                     # LLM provider integrations
│   ├── openai_provider.py
│   ├── azure_provider.py
│   └── anthropic_provider.py
├── 📈 datasets/                # Dataset loaders
│   ├── tempo.py               # TEMPO benchmark
│   └── timebench.py           # TimeBench
├── 🔧 guidance/                # Temporal guidance generation
├── ⚡ efficiency/              # Cost & latency tracking
└── 🛠️ utils/                   # Utility functions
```

---

## 📚 Examples

We provide comprehensive examples in the `examples/` directory:

| Example | Description | LLM Required |
|---------|-------------|--------------|
| [`01_retrieval_bm25.py`](examples/01_retrieval_bm25.py) | Basic retrieval evaluation | ❌ |
| [`02_rag_generation.py`](examples/02_rag_generation.py) | RAG generation evaluation | ✅ |
| [`03_full_pipeline.py`](examples/03_full_pipeline.py) | Complete RAG pipeline | ✅ |
| [`04_tempo_dataset.py`](examples/04_tempo_dataset.py) | Using TEMPO benchmark | ❌ |
| [`05_cross_period.py`](examples/05_cross_period.py) | Cross-period queries | ✅ |
| [`06_tempo_hsm_complete.py`](examples/06_tempo_hsm_complete.py) | Full HSM evaluation | ✅ |
| [`07_generate_guidance.py`](examples/07_generate_guidance.py) | Generate temporal guidance | ✅ |
| [`08_pipeline_with_generated_guidance.py`](examples/08_pipeline_with_generated_guidance.py) | End-to-end pipeline | ✅ |

### Running Examples

```bash
cd examples

# Copy and configure credentials (for LLM examples)
cp .env.example .env
# Edit .env with your API keys

# Run examples
python 01_retrieval_bm25.py      # No LLM needed
python 02_rag_generation.py      # Requires .env
```

---

## 🔧 Configuration

### Environment Variables (for LLM-based evaluation)

Create a `.env` file or set environment variables:

```bash
# Azure OpenAI (Recommended)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-05-01-preview

# Or OpenAI
OPENAI_API_KEY=your-openai-key

# Or Anthropic
ANTHROPIC_API_KEY=your-anthropic-key
```

### Programmatic Configuration

```python
from tempoeval.core import TempoEvalConfig

config = TempoEvalConfig(
    k_values=[5, 10, 20],           # Evaluation depths
    use_focus_time=True,            # Enable Focus Time extraction
    llm_provider="azure",           # LLM provider
    parallel_requests=10,           # Concurrent LLM calls
)
```

---

## 📈 TEMPO Benchmark

TempoEval includes built-in support for the **TEMPO** benchmark - a comprehensive temporal QA dataset:

```python
from tempoeval.datasets import load_tempo, load_tempo_documents

# Load queries with temporal annotations
queries = load_tempo(domain="bitcoin", max_samples=100)

# Load corpus documents
documents = load_tempo_documents(domain="bitcoin")

# Available domains: bitcoin, cardano, economics, hsm (History of Science & Medicine)
```

---

## ⚡ Efficiency Tracking

Track latency and costs for LLM-based evaluation:

```python
from tempoeval.efficiency import EfficiencyTracker

tracker = EfficiencyTracker(model_name="gpt-4o")

# ... run your evaluation ...

# Get summary
summary = tracker.summary()
print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
print(f"Avg Latency: {summary['avg_latency_ms']:.1f}ms")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=tempoeval --cov-report=html
```

---

## 📖 Documentation

Full documentation is available at: **[https://aa17626.github.io/tempoeval](https://aa17626.github.io/tempoeval)**

- [Getting Started Guide](https://aa17626.github.io/tempoeval/getting-started)
- [Metrics Reference](https://aa17626.github.io/tempoeval/metrics)
- [API Documentation](https://aa17626.github.io/tempoeval/api)
- [TEMPO Benchmark](https://aa17626.github.io/tempoeval/tempo)

---

## 📄 Citation

If you use TempoEval in your research, please cite our paper:

```bibtex
soon
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on top of the [TEMPO Benchmark](https://huggingface.co/datasets/tempo26/Tempo)
- LLM integrations via [OpenAI](https://openai.com), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), and [Anthropic](https://anthropic.com)

---

<p align="center">
  <strong>Made with ❤️ for the Temporal IR Community</strong>
</p>

<p align="center">
  <a href="https://github.com/DataScienceUIBK/tempoeval">
    <img src="https://img.shields.io/badge/⭐_Star_on_GitHub-000?style=for-the-badge&logo=github" alt="Star on GitHub"/>
  </a>
</p>
