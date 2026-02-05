# Generation Metrics

Detailed reference for generation evaluation metrics (Layer 2).

## TemporalFaithfulness

Measures whether temporal claims in the generated answer are supported by the retrieved context.

```python
from tempoeval.metrics import TemporalFaithfulness
metric = TemporalFaithfulness()
score = metric.compute(contexts=[...], answer="...")
```

---

## TemporalHallucination

Measures whether temporal claims are hallucinated (fabricated) or contradict ground truth.

```python
from tempoeval.metrics import TemporalHallucination
metric = TemporalHallucination()
score = metric.compute(contexts=[...], answer="...", gold_answer="...")
```

---

## TemporalCoherence

Measures the internal chronological consistency of the answer. Checks for self-contradictions (e.g., "born in 1990... died in 1980").

```python
from tempoeval.metrics import TemporalCoherence
score = metric.compute(answer="...")
```

---

## AnswerTemporalAlignment

Measures whether the ANSWER's temporal scope aligns with the QUERY's focus time. Does the answer talk about the *requested* time?

```python
from tempoeval.metrics import AnswerTemporalAlignment
score = metric.compute(query="...", answer="...")
```

---

## AnswerTemporalRecall

Measures how much of the Query Focus Time is covered by the answer.

$$Recall = \frac{|AFT \cap QFT|}{|QFT|}$$

```python
from tempoeval.metrics import AnswerTemporalRecall
metric = AnswerTemporalRecall(use_focus_time=True)
```
