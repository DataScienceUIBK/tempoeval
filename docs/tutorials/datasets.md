# Tutorial 3: Benchmarking with TEMPO

Do you want to compare your system against state-of-the-art benchmarks? TempoEval integrates directly with the **TEMPO** dataset.

## 1. Load Dataset

We can load a specific domain from TEMPO (e.g., `history`, `covid`, `politics`).

```python
from tempoeval.datasets import load_tempo

# Load 100 samples from the 'history' split
dataset = load_tempo(domain="history", split="test", max_samples=100)

print(f"Loaded {len(dataset)} queries")
print(f"Query 1: {dataset[0].query}")
print(f"Gold ids: {dataset[0].gold_ids}")
```

## 2. Run Your Retrieval System

Iterate through the dataset and retrieve documents using your own system (or our built-in ones).

```python
from tempoeval.retrieval import BM25Retriever

# Extract queries and doc_ids
queries = [d.query for d in dataset]
query_ids = [d.id for d in dataset]

# Assuming you already have 'documents' list loaded
# ...

retriever = BM25Retriever()
results = retriever.retrieve(queries, query_ids, documents, doc_ids, k=10)
```

## 3. Standard vs Temporal Evaluation

You can compute standard IR metrics and Temporal metrics side-by-side.

```python
from tempoeval.metrics import TemporalNDCG, TemporalRecall

# Standard IR (using gold labels provided in dataset)
# ... code to compute pytrec_eval ...

# Temporal Metrics (using Focus Time)
temp_ndcg = TemporalNDCG(use_focus_time=True)

scores = []
for q in dataset:
    # Get retrieved docs for this query
    retrieved_docs_content = [...] 
    
    qft = extract_qft(q.query)
    dfts = [extract_dft(d) for d in retrieved_docs_content]
    
    s = temp_ndcg.compute(qft, dfts, k=10)
    scores.append(s)

print(f"Average Temporal NDCG: {sum(scores)/len(scores)}")
```

## Next Steps

Check out our [examples on GitHub](https://github.com/DataScienceUIBK/tempoeval/tree/main/examples) to see this implemented at scale.

