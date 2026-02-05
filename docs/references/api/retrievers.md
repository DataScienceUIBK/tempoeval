# Retrievers API

TempoEval includes a comprehensive `retrieval` module to help you build and benchmark temporal RAG systems.

## Sparse Retrievers

### BM25
The standard baseline. Uses `pyserini` or `rank_bm25`.

```python
from tempoeval.retrieval import BM25Retriever
retriever = BM25Retriever()
```

---

## Dense Retrievers

State-of-the-art embedding models.

### BGE (BAAI General Embedding)
High performance dense retrieval.

```python
from tempoeval.retrieval import BGERetriever
# Defaults to 'BAAI/bge-large-en-v1.5'
retriever = BGERetriever()
```

### Contriever
Contrastive training for dense retrieval.

```python
from tempoeval.retrieval import ContrieverRetriever
retriever = ContrieverRetriever()
```

### Sentence Transformers (Generic)
Load any model from Hugging Face.

```python
from tempoeval.retrieval import SentenceTransformerRetriever
retriever = SentenceTransformerRetriever("sentence-transformers/all-mpnet-base-v2")
```

### Nomic
Nomic Embed Text v1 (Long context).

```python
from tempoeval.retrieval import NomicRetriever
```

### Instructor
Instruction-finetuned embeddings.

```python
from tempoeval.retrieval import InstructorRetriever
retriever = InstructorRetriever(instruction="Represent the query for retrieval:")
```

---

## API Retrievers

### OpenAI / Azure OpenAI
Embedding models from OpenAI.

```python
from tempoeval.retrieval import AzureOpenAIRetriever
retriever = AzureOpenAIRetriever(
    deployment="text-embedding-3-small"
)
```

### Cohere
Cohere Command R / Embed v3.

```python
from tempoeval.retrieval import CohereRetriever
```

### Voyage
Voyage AI embeddings (voyage-large-2).

```python
from tempoeval.retrieval import VoyageRetriever
```

---

## LLM-Based Retrievers (Generative / SF)

### Qwen2 / E5 / SF
Using LLMs as retrievers (Score function or encoder).

```python
from tempoeval.retrieval import QwenE5SFRetriever
retriever = QwenE5SFRetriever(model_id="qwen2") # or "e5", "sf"
```

### GritLM
Generative Representational Instruction Tuning.

```python
from tempoeval.retrieval import GritLMRetriever
```

### Diver
Diverse LLM-based retriever.

```python
from tempoeval.retrieval import DiverRetriever
```
