# Anchor Coverage

Anchor Coverage measures the F1 score between time anchors required by the query and anchors present in retrieved documents.

## Formula
$$
F1 = 2 * (precision * recall) / (precision + recall)
$$

## Inputs
- `required_anchors` or `key_time_anchors`
- `retrieved_docs` (for extraction)
- `k` (cutoff)

## Output
- Range: [0, 1], higher is better.

## Prompt (LLM mode)
```text
## Task
Extract the CENTRAL temporal focus of the documents.

## Documents
{documents}

## Instructions
Identify ONLY the time periods that are fundamentally important to the document's main topic.
Do NOT include:
- Publication dates, copyright years, or citation dates
- Incidental mentions (e.g., "born in 1950" if the article is about their work in 1990)
- Relative dates without context (e.g., "last year")

Focus on:
- The main era/period being discussed
- Key event dates
- Explicit time scopes (e.g., "study from 2010 to 2015")

## Output (JSON)
{
    "extracted_anchors": ["1999", "the 1990s", "Victorian era"],
    "anchor_count": 3
}
```

## Examples
```python
from tempoeval.metrics import AnchorCoverage

metric = AnchorCoverage(use_llm=True)
metric.llm = llm
score = metric.compute(
    retrieved_docs=["..."],
    required_anchors=["2015", "2020"],
    k=5
)
```

!!! note "Synchronous usage"
    Use `compute(...)` for sync calls and `acompute(...)` for async.
