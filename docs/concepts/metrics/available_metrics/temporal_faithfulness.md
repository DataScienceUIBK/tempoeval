# Temporal Faithfulness

<span class="metric-badge generation">Generation Metric</span>

Temporal Faithfulness measures whether the temporal claims in the generated answer are grounded in the retrieved documents. This prevents temporal hallucinations—when the model fabricates dates or time periods not mentioned in the context.

## Formula (Focus Time Mode)

$$
\text{TemporalFaithfulness} = \frac{|AFT \cap \bigcup_{i=1}^{K} DFT_i|}{|AFT|}
$$

Where:

- $AFT$ = Answer Focus Time (years in the generated answer)
- $DFT_i$ = Document Focus Time for document $i$
- $K$ = number of retrieved documents

**In simple terms**: What fraction of years mentioned in the answer appear in the retrieved documents?

## Formula (LLM Mode)

$$
\text{Faithfulness} = \frac{\text{supported} + 0.5 \times \text{partially\_supported}}{\text{total\_claims}}
$$

Where the LLM judges each temporal claim as SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or CONTRADICTED based on the context.

## Example (Focus Time Mode)

Retrieved documents:
- Doc 1: *"In 2008, Lehman Brothers collapsed."* → DFT = `{2008}`
- Doc 2: *"The 2009 stimulus package..."* → DFT = `{2009}`

Answer: *"The crisis occurred in 2008 and continued into 2009."* → AFT = `{2008, 2009}`

Union of document years: `{2008, 2009}`

Temporal Faithfulness = 2/2 = 1.0 (all answer years are grounded)

## Example (Hallucination)

Retrieved documents: DFT = `{2008, 2009}`

Answer (hallucinated): *"The crisis started in 2007 and ended in 2010."* → AFT = `{2007, 2010}`

Temporal Faithfulness = 0/2 = 0.0 (no answer years are grounded in context)

## Inputs
- `answer` - Generated answer text
- `contexts` or `retrieved_docs` - List of retrieved document texts
- Optional: `aft` and `dfts` if already extracted

## Output
- Range: [0, 1], higher is better

## Code Example (Focus Time Mode)

```python
from tempoeval.core import extract_aft, extract_dft
from tempoeval.metrics import TemporalFaithfulness

# Extract Focus Times
answer = "The crisis occurred in 2008 and continued into 2009."
contexts = [
    "In 2008, Lehman Brothers collapsed.",
    "The 2009 stimulus package helped recovery."
]

aft = extract_aft(answer)
dfts = [extract_dft(ctx) for ctx in contexts]

# Compute metric
metric = TemporalFaithfulness()
score = metric.compute(aft=aft, dfts=dfts)
print(f"Temporal Faithfulness: {score}")  # 1.0
```

## Code Example (LLM Mode)

```python
from tempoeval.metrics import TemporalFaithfulness
from tempoeval.llm import OpenAIProvider

llm = OpenAIProvider(model="gpt-4o")
metric = TemporalFaithfulness()
metric.llm = llm

score = await metric.acompute(
    answer="The crisis occurred in 2008 and continued into 2009.",
    contexts=["In 2008, Lehman Brothers collapsed.", "The 2009 stimulus..."]
)
print(f"Temporal Faithfulness: {score}")
```

## LLM Prompt

The LLM mode uses a detailed prompt to:

1. **Extract temporal claims** from the answer (dates, durations, sequences)
2. **Verify each claim** against the retrieved context
3. **Classify** as SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or CONTRADICTED
4. **Calculate score** using the formula above

!!! danger "Temporal Hallucination"
    A Faithfulness score < 0.5 indicates serious hallucination—the answer mentions years or time periods not supported by the retrieved documents.

!!! tip "When to use Faithfulness"
    Use Temporal Faithfulness to:
    - Detect hallucinated dates in generated answers
    - Ensure RAG systems stay grounded in retrieved documents
    - Evaluate if the retrieval step provided sufficient temporal coverage

!!! note "Focus Time vs LLM Mode"
    - **Focus Time Mode**: Fast, checks if answer years appear in documents
    - **LLM Mode**: Slower, but verifies specific temporal claims and their evidence
