# Year Recall@K

<span class="metric-badge retrieval">Retrieval Metric</span>

Year Recall@K measures the fraction of query-relevant years that are covered by the retrieved documents. This evaluates temporal coverage at the year level.

## Formula

$$
\text{YearRecall@K} = \frac{\left|\bigcup_{i=1}^{K} DFT_i \cap QFT\right|}{|QFT|}
$$

Where:

- $\bigcup_{i=1}^{K} DFT_i$ = union of all years from the top-K documents
- $QFT$ = Query Focus Time (set of query-relevant years)

**In simple terms**: Of all the years the query asks about, what fraction are mentioned in the retrieved documents?

## Example

Query: *"Events from 2019 to 2021"* → QFT = `{2019, 2020, 2021}`

Retrieved documents:
- Doc 1: DFT = `{2020, 2021}`
- Doc 2: DFT = `{2022}`

Union of all years: `{2020, 2021, 2022}`

Covered query years: `{2020, 2021}`

**Year Recall** = 2/3 = 0.667 (missing 2019)

## Inputs
- `qft` - Query Focus Time (set of years)
- `dfts` - List of Document Focus Times
- `k` - Cutoff for top-K documents

## Output
- Range: [0, 1], higher is better

## Code Example

```python
from tempoeval.metrics import TemporalYearRecall

metric = TemporalYearRecall()
score = metric.compute(
    qft={2010, 2015, 2020},
    dfts=[{2010}, {2015}],
    k=2
)
print(f"Year Recall@2: {score}")  # 0.667 (2 out of 3 query years covered)
```

!!! tip "When to use Year Recall"
    Use Year Recall when you need comprehensive temporal coverage—ensuring all query-relevant years are represented in the results.

!!! note "Comparison with Temporal Recall"
    - **Temporal Recall**: Measures document-level coverage (what fraction of relevant docs were retrieved?)
    - **Year Recall**: Measures year-level coverage (what fraction of query years are covered?)
