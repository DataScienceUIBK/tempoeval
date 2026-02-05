# Tutorial 2: Evaluating a RAG Pipeline

In this tutorial, we will evaluate the full RAG pipeline: **Retrieval + Generation**.
We will use an LLM provider to generate an answer and measure its **Temporal Faithfulness**.

---

## 1. Configure LLM

TempoEval supports Azure OpenAI, OpenAI, Anthropic, and more.

```python
import os
from tempoeval.llm import AzureOpenAIProvider

# Set env vars or pass directly
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"

llm = AzureOpenAIProvider(
    deployment_name="gpt-4",
    api_version="2024-02-15-preview"
)
```

## 2. The RAG Scenario

Retrieval has already happened. We have a context and a generated answer.

**Query**: "What happened to Lehman Brothers?"
**Context**: "Lehman Brothers filed for bankruptcy in September 2008."
**Generated Answer**: "Lehman Brothers collapsed in 2008 due to the subprime mortgage crisis."

```python
query = "What happened to Lehman Brothers?"
context = "Lehman Brothers filed for bankruptcy in September 2008."
answer = "Lehman Brothers collapsed in 2008 due to the subprime mortgage crisis."
```

## 3. Evaluate Faithfulness

**Temporal Faithfulness** checks if the temporal claims in the answer (e.g., "in 2008") are supported by the context.

```python
from tempoeval.metrics import TemporalFaithfulness

faithfulness_metric = TemporalFaithfulness()
faithfulness_metric.llm = llm  # Assign the LLM provider

score = faithfulness_metric.compute(
    contexts=[context],
    answer=answer
)

print(f"Faithfulness Score: {score}")
# Expected: 1.0 (Claim "2008" is in context)
```

## 4. Evaluate Hallucination (Optional)

What if the model hallucinates a wrong date?

**Bad Answer**: "Lehman Brothers collapsed in **1999**."

```python
from tempoeval.metrics import TemporalHallucination

hallucination_metric = TemporalHallucination()
hallucination_metric.llm = llm

score = hallucination_metric.compute(
    contexts=[context],
    answer="Lehman Brothers collapsed in 1999."
)

print(f"Hallucination Score: {score}")
# Expected: High score (Hallucinated date not found in context)
```

## 5. Composite Score (TempoScore)

To get a single number for your system, use **TempoScore**.

```python
from tempoeval.metrics import TempoScore

# Combine scores from retrieval and generation
rag_score = TempoScore().compute(
    temporal_precision=1.0,  # From Tutorial 1
    temporal_faithfulness=1.0 # From step 3
)

print(f"TempoScore: {rag_score}")
```
