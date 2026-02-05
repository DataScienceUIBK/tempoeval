# Contributing to TempoEval

Thank you for your interest in contributing to TempoEval! 🎉

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tempoeval.git
   cd tempoeval
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tempoeval --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## 📝 How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Minimal code example if possible

### Suggesting Features

1. **Open a discussion** or issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered

### Submitting Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Follow the code style (we use Black for formatting)
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests locally**
   ```bash
   pytest tests/ -v
   ```

4. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation only
   - `test:` - Adding tests
   - `refactor:` - Code refactoring

5. **Push and create PR**
   ```bash
   git push origin feature/amazing-feature
   ```

## 🎨 Code Style

- **Formatting**: We use [Black](https://black.readthedocs.io/) with line length 120
- **Imports**: Sorted with [isort](https://pycqa.github.io/isort/)
- **Linting**: [Ruff](https://github.com/astral-sh/ruff) for fast linting
- **Type hints**: Encouraged for public APIs

```bash
# Format code
black tempoeval/ --line-length=120
isort tempoeval/

# Check linting
ruff check tempoeval/
```

## 📁 Project Structure

```
tempoeval/
├── core/           # Core classes (FocusTime, Evaluator, etc.)
├── metrics/        # All temporal metrics
│   ├── retrieval/  # Layer 1: Retrieval metrics
│   ├── generation/ # Layer 2: Generation metrics
│   ├── reasoning/  # Layer 3: Reasoning metrics
│   └── composite/  # TempoScore
├── llm/            # LLM provider integrations
├── datasets/       # Dataset loaders
├── guidance/       # Temporal guidance generation
├── efficiency/     # Cost & latency tracking
└── utils/          # Utility functions
```

## 🧪 Adding New Metrics

1. **Create metric file** in appropriate directory (`metrics/retrieval/`, etc.)
2. **Inherit from base class** (`BaseRetrievalMetric`, `BaseGenerationMetric`, etc.)
3. **Implement required methods**:
   - `compute()` - Synchronous computation
   - `acompute()` - Async computation (if LLM-based)
4. **Add to exports** in `__init__.py`
5. **Write tests** in `tests/`
6. **Add documentation**

Example:
```python
from tempoeval.core.base import BaseRetrievalMetric

class MyNewMetric(BaseRetrievalMetric):
    name = "my_new_metric"
    requires_llm = False
    
    def compute(self, **kwargs) -> float:
        # Implementation
        return score
```

## 📖 Documentation

- Documentation is built with [MkDocs](https://www.mkdocs.org/)
- API docs are auto-generated from docstrings
- Update `docs/` for new features

```bash
# Preview documentation locally
mkdocs serve
```

## 🙏 Thank You!

Every contribution, no matter how small, helps make TempoEval better for everyone. We appreciate your time and effort!

---

<p align="center">
  Questions? Open an issue or reach out to the maintainers.
</p>
