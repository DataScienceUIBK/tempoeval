# TempoEval Examples

This directory contains example scripts demonstrating how to use the TempoEval library.

## Setup

Before running examples that require LLM calls (examples 02, 03, 05, 06, 07, 08), you need to configure your API credentials:

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials:**
   ```bash
   # Azure OpenAI (recommended)
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_API_VERSION=2024-05-01-preview
   
   # Or OpenAI
   OPENAI_API_KEY=your-openai-api-key-here
   ```

3. **Install dependencies:**
   ```bash
   pip install python-dotenv
   ```

> ⚠️ **Never commit your `.env` file!** It's already in `.gitignore`.

## Examples Overview

| Example | Description | LLM Required? |
|---------|-------------|---------------|
| `01_retrieval_bm25.py` | Basic retrieval evaluation with BM25 | ❌ No |
| `02_rag_generation.py` | RAG generation evaluation with LLM-as-judge | ✅ Yes |
| `03_full_pipeline.py` | Complete retrieval + generation pipeline | ✅ Yes |
| `04_tempo_dataset.py` | Loading and using TEMPO benchmark data | ❌ No |
| `05_cross_period.py` | Cross-period query evaluation | ✅ Yes |
| `06_tempo_hsm_complete.py` | Full HSM evaluation with threading | ✅ Yes |
| `07_generate_guidance.py` | Generate temporal guidance for custom data | ✅ Yes |
| `08_pipeline_with_generated_guidance.py` | Complete pipeline with generated guidance | ✅ Yes |
| `09_bm25_tempo_hsm.py` | BM25 retrieval on TEMPO HSM | ❌ No |
| `reproduce_sigir_results.py` | SIGIR 2026 reproduction script | ❌ No |

## Running Examples

### No LLM Required
These examples work out of the box:

```bash
# Example 1: Basic retrieval evaluation
python 01_retrieval_bm25.py

# Example 4: Using TEMPO dataset
python 04_tempo_dataset.py

# Example 9: BM25 on TEMPO HSM
python 09_bm25_tempo_hsm.py

# SIGIR reproduction
python reproduce_sigir_results.py
```

### LLM Required
Make sure you've configured `.env` first:

```bash
# Example 2: RAG generation evaluation
python 02_rag_generation.py

# Example 3: Full pipeline
python 03_full_pipeline.py

# Example 5: Cross-period evaluation
python 05_cross_period.py

# Example 6: Complete HSM evaluation (threaded)
python 06_tempo_hsm_complete.py

# Example 7: Generate guidance
python 07_generate_guidance.py

# Example 8: Pipeline with generated guidance
python 08_pipeline_with_generated_guidance.py
```

## Additional Dependencies

Some examples require additional packages:

```bash
# For dataset loading
pip install datasets huggingface_hub pyarrow

# For BM25 retrieval
pip install gensim pyserini

# For LLM calls
pip install openai python-dotenv
```

## Output Files

Some examples generate output files in the current directory:
- `generated_guidance.json` - Temporal guidance output
- `timeqa_with_generated_guidance.json` - TimeQA evaluation results
- `tempo_hsm_results_threaded.json` - HSM evaluation results

## Troubleshooting

### Missing API Credentials
If you see:
```
ERROR: Azure OpenAI credentials not found!
```
Make sure you've copied `.env.example` to `.env` and filled in your credentials.

### Import Errors
If you see import errors, make sure TempoEval is installed:
```bash
pip install -e ..
```

### Missing Dependencies
Install any missing packages:
```bash
pip install datasets gensim openai python-dotenv
```
