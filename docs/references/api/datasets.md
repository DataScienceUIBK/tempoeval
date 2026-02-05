# Datasets API

Loaders for standard Temporal RAG benchmarks.

## TEMPO (Primary)
A dataset for temporal retrieval and reasoning across multiple domains (History, Politics, Finance, etc.).

```python
from tempoeval.datasets import load_tempo
ds = load_tempo(domain="history", split="test")
```

---

## TimeQA (Reasoning)
Question Answering dataset requiring explicit temporal constraint satisfaction.

```python
from tempoeval.datasets import load_timeqa
ds = load_timeqa(split="test")
```

---

## SituatedQA
Dataset where answers change depending on the temporal context (e.g., "Who represents District 5?").

```python
from tempoeval.datasets import load_situatedqa
ds = load_situatedqa()
```

---

## TimeBench
Complex temporal reasoning benchmark involving reasoning over timelines.

```python
from tempoeval.datasets import load_timebench
```

---

## Complex TempQA
Multi-hop queries requiring synthesis of multiple temporal facts.

```python
from tempoeval.datasets import load_complex_tempqa
```

---

## Dataset Structure

All loaders return a list of `TemporalQuery` objects:

```python
@dataclass
class TemporalQuery:
    id: str
    query: str
    gold_ids: List[str]
    year: Optional[int] = None
    domain: Optional[str] = None
    query_guidance: Optional[Dict] = None
```
