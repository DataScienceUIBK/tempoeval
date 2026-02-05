# Retrieval Metrics

Detailed reference for retrieval evaluation metrics (Layer 1).

## TemporalPrecision

Computes the proportion of retrieved documents that are temporally relevant to the query.

```python
from tempoeval.metrics import TemporalPrecision
```

### Formula
$$P@K = \frac{|\{d \in R_K | relevance(d, q) > 0\}|}{K}$$
where relevance is determined by Focus Time overlap: $QFT \cap DFT \neq \emptyset$.

---

## TemporalRecall

Computes the proportion of relevant documents retrieved.

```python
from tempoeval.metrics import TemporalRecall
```

---

## TemporalNDCG

Normalized Discounted Cumulative Gain adapted for temporal relevance. Considers the *degree* of overlap or relevance rank.

```python
from tempoeval.metrics import TemporalNDCG
```

---

## TemporalMRR

Mean Reciprocal Rank. The reciprocal rank of the *first* temporally relevant document.

```python
from tempoeval.metrics import TemporalMRR
```

---

## TemporalMAP

Mean Average Precision. Calculates precision at every relevant document retrieved and averages it.

```python
from tempoeval.metrics import TemporalMAP
```

---

## YearPrecision

Precision of unique *years* retrieved versus requested. Evaluates temporal coverage scope.

$$YearP@K = \frac{|(\cup DFTs) \cap QFT|}{|\cup DFTs|}$$

```python
from tempoeval.metrics import TemporalYearPrecision
```

---

## YearRecall

Recall of unique *years* retrieved versus requested.

$$YearR@K = \frac{|(\cup DFTs) \cap QFT|}{|QFT|}$$

```python
from tempoeval.metrics import TemporalYearRecall
```

---

## TemporalDiversity

Measures the diversity of time periods covered by the retrieved documents. High diversity uses Standard Deviation of centroids of time intervals.

```python
from tempoeval.metrics import TemporalDiversity
```

---

## TemporalCoverage

Measures the total coverage of the Query Focus Time by the retrieved documents.

$$Coverage = \frac{|\cup (DFT \cap QFT)|}{|QFT|}$$

```python
from tempoeval.metrics import TemporalCoverage
```

---

## AnchorCoverage

Measures whether specific "key anchors" (important years mentioned in the query) are covered.

```python
from tempoeval.metrics import AnchorCoverage
```

---

## TemporalPersistence

Evaluates if relevant documents appear consistently across different retrieval depths or methods.

```python
from tempoeval.metrics import TemporalPersistence
```
