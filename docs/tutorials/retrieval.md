# Tutorial 1: Evaluating Retrieval

In this tutorial, we will evaluate a Retrieval system's ability to find temporally relevant documents.

We will:
1.  Set up a simple document corpus.
2.  Run a retriever (BM25).
3.  Evaluate the results using **Focus Time**.

---

## 1. Setup Data

Let's imagine we are building a RAG system for **History Question Answering**. We have a query about the **2008 Financial Crisis**, but our database contains documents from various years.

```python
from tempoeval.core import extract_qft, extract_dft

# The User Query
query = "What caused the 2008 financial crisis?"

# The Corpus (Documents)
documents = [
    "The collapse of Lehman Brothers in 2008 marked the beginning...", # Relevant (2008)
    "The 1929 Wall Street Crash was the most devastating...",         # Irrelevant (1929)
    "In 2020, the COVID-19 pandemic caused a global recession...",    # Irrelevant (2020)
    "The Housing and Economic Recovery Act of 2008 was enacted..."    # Relevant (2008)
]
doc_ids = ["doc1", "doc2", "doc3", "doc4"]
```

## 2. Run Retrieval

We'll use TempoEval's built-in `BM25Retriever`.

```python
from tempoeval.retrieval import BM25Retriever

retriever = BM25Retriever()
results = retriever.retrieve(
    queries=[query], 
    query_ids=["q1"], 
    documents=documents, 
    doc_ids=doc_ids,
    k=2
)

# Output: {'q1': [{'doc_id': 'doc1', 'score': 1.5}, {'doc_id': 'doc4', 'score': 1.2}]}
retrieved_ids = [r['doc_id'] for r in results['q1']]
print(f"Retrieved: {retrieved_ids}")
```

## 3. Evaluate with Focus Time

Traditional evaluation needs "Gold Labels" (human-annotated relevance). But we can use **Focus Time** to automatically judge relevance based on the *years* mentioned.

```python
from tempoeval.metrics import TemporalPrecision

# Calculate Focus Times
# Simple Regex is fast and free!
qft = extract_qft(query)  # {2008}
dfts = [extract_dft(doc) for doc in documents] 

# Re-map stored DFTs to retrieved docs
# (In a real app, you'd cache these DFTs)
retrieved_dfts = [
    dfts[doc_ids.index(did)] 
    for did in retrieved_ids
]

# Compute Metric
metric = TemporalPrecision(use_focus_time=True)
score = metric.compute(
    qft=qft, 
    dfts=retrieved_dfts, 
    k=2
)

print(f"Temporal Precision@2: {score}")
```

### Result Analysis
If the retriever pulled `doc1` (2008) and `doc4` (2008), the intersection with the query `2008` is perfect.
**Score: 1.0**

If it pulled `doc2` (1929), the intersection is empty. The score would drop.

## Next Steps

Now that you've evaluated retrieval, let's look at generation.

[➡️ Tutorial 2: Evaluate RAG Pipeline](rag_pipeline.md)
