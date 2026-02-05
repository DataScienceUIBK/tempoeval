# Installation

Get TempoEval up and running in minutes.

---

## Quick Install

The easiest way to install TempoEval is via pip:

=== "PyPI (Recommended)"

    ```bash
    pip install tempoeval
    ```

=== "With Optional Dependencies"

    ```bash
    # Install with LLM provider support
    pip install tempoeval[openai]
    pip install tempoeval[anthropic]
    pip install tempoeval[google]

    # Install with dataset support
    pip install tempoeval[datasets]

    # Install everything
    pip install tempoeval[all]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/tempo26/TempoEval.git
    cd TempoEval
    pip install -e .
    ```

!!! success "What's Included"
    The base installation includes:

    - ✅ **Regex Extraction** - Fast year extraction (default method)
    - ✅ **All 20+ Metrics** - Retrieval, generation, and reasoning evaluation
    - ✅ **Focus Time Core** - QFT/DFT/AFT extraction and comparison
    - ✅ **BM25 Retriever** - Basic retrieval functionality

---

## Advanced Setup

### Temporal Taggers

For more accurate temporal expression extraction beyond regex, TempoEval supports multiple external temporal taggers:

=== "Dateparser (Recommended)"

    The **recommended** choice for most users - pure Python, fast, and multi-language support.

    ```bash
    pip install dateparser
    ```

    ```python
    from tempoeval.core.focus_time import QueryFocusTime, TemporalTagger

    extractor = QueryFocusTime()
    qft = extractor.extract(
        "Events from last decade",
        use_temporal_tagger=True,
        tagger=TemporalTagger.DATEPARSER
    )
    ```

=== "Parsedatetime"

    Pure Python, focused on English natural language expressions.

    ```bash
    pip install parsedatetime
    ```

    ```python
    qft = extractor.extract(
        "Next month's events",
        use_temporal_tagger=True,
        tagger=TemporalTagger.PARSEDATETIME
    )
    ```

=== "Fast-Parse-Time"

    Ultra-fast for high-throughput scenarios (sub-millisecond).

    ```bash
    pip install fast-parse-time
    ```

    ```python
    qft = extractor.extract(
        query,
        use_temporal_tagger=True,
        tagger=TemporalTagger.FAST_PARSE_TIME
    )
    ```

=== "HeidelTime (Java)"

    Most accurate for complex expressions, but **very slow** (requires Java JVM startup).

    !!! warning "Requirements"
        HeidelTime requires Java Runtime Environment (JRE) 8 or higher.

    **1. Verify Java Installation:**

    ```bash
    java -version
    ```

    If Java is not installed:

    - **Ubuntu/Debian**: `sudo apt-get install default-jre`
    - **macOS**: `brew install openjdk@11`
    - **Windows**: Download from [Oracle Java](https://www.oracle.com/java/technologies/downloads/)

    **2. Install Python Wrapper:**

    ```bash
    pip install py_heideltime
    ```

    **3. Usage:**

    ```python
    qft = extractor.extract(
        "Events in the 1990s",
        use_temporal_tagger=True,
        tagger=TemporalTagger.HEIDELTIME
    )
    ```

---

### LLM Providers

Install LLM provider packages to use LLM-based metrics:

=== "OpenAI"

    ```bash
    pip install openai>=1.0.0

    # Set API key
    export OPENAI_API_KEY="your-api-key"
    ```

    ```python
    from tempoeval.llm import OpenAIProvider

    llm = OpenAIProvider(model="gpt-4o")
    ```

=== "Anthropic"

    ```bash
    pip install anthropic>=0.20.0

    # Set API key
    export ANTHROPIC_API_KEY="your-api-key"
    ```

    ```python
    from tempoeval.llm import AnthropicProvider

    llm = AnthropicProvider(model="claude-3-5-sonnet-20241022")
    ```

=== "Google Gemini"

    ```bash
    pip install google-generativeai>=0.3.0

    # Set API key
    export GOOGLE_API_KEY="your-api-key"
    ```

    ```python
    from tempoeval.llm import GeminiProvider

    llm = GeminiProvider(model="gemini-pro")
    ```

=== "Azure OpenAI"

    ```bash
    pip install openai>=1.0.0

    # Set environment variables
    export AZURE_OPENAI_ENDPOINT="your-endpoint"
    export AZURE_OPENAI_API_KEY="your-api-key"
    export AZURE_DEPLOYMENT_NAME="gpt-4o"
    ```

    ```python
    from tempoeval.llm import AzureOpenAIProvider

    llm = AzureOpenAIProvider()
    ```

---

### Dense Retrievers

For semantic search with dense embeddings:

```bash
pip install torch sentence-transformers
```

**Supported Models:**

- BGE (Beijing Academy of Artificial Intelligence)
- Contriever (Facebook AI)
- Custom sentence-transformer models

```python
from tempoeval.retrieval import DenseRetriever

retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5")
```

---

## Development Installation

For contributors or those who want to modify the source code:

### 1. Clone the Repository

```bash
git clone https://github.com/tempo26/TempoEval.git
cd TempoEval
```

### 2. Create Virtual Environment

=== "Linux/macOS"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

=== "Windows"

    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

### 3. Install in Editable Mode

```bash
# Install package in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Run Tests

```bash
pytest tests/
```

---

## Verify Installation

Test that everything is working:

```python
from tempoeval.core import extract_qft, extract_dft
from tempoeval.metrics import TemporalPrecision

# Extract focus times
qft = extract_qft("What happened in 2020?")
dft = extract_dft("The COVID-19 pandemic began in 2020.")

print(f"Query Focus Time: {qft.years}")      # {2020}
print(f"Document Focus Time: {dft.years}")   # {2020}

# Test metric
metric = TemporalPrecision(use_focus_time=True)
score = metric.compute(qft=qft, dfts=[dft], k=1)
print(f"Temporal Precision: {score}")        # 1.0
```

!!! success "Installation Complete"
    You're all set! Head to the [Quick Start](quickstart.md) guide to run your first evaluation.

---

## Troubleshooting

??? question "ImportError: No module named 'tempoeval'"

    Make sure you've installed the package and activated your virtual environment:

    ```bash
    pip list | grep tempoeval
    ```

??? question "HeidelTime not working"

    Ensure Java is installed and accessible:

    ```bash
    java -version
    which java  # Linux/macOS
    where java  # Windows
    ```

??? question "LLM API errors"

    Verify your API keys are set:

    ```bash
    echo $OPENAI_API_KEY
    echo $ANTHROPIC_API_KEY
    ```

??? question "GPU not detected for dense retrievers"

    Check PyTorch CUDA installation:

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

---

## What's Next?

- **[Quick Start Guide](quickstart.md)** - Your first evaluation in 5 minutes
- **[Core Concepts](../concepts/index.md)** - Understand Focus Time and evaluation modes
- **[Tutorials](../tutorials/index.md)** - Step-by-step guides for common tasks
