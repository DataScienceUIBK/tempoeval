# Year Precision@K

<span class="metric-badge retrieval">Retrieval Metric</span>

Year Precision@K measures the fraction of all years mentioned in the retrieved documents that are relevant to the query. This metric evaluates temporal precision at the year level rather than the document level.

## Formula

$$
\text{YearPrecision@K} = \frac{\left|\bigcup_{i=1}^{K} DFT_i \cap QFT\right|}{\left|\bigcup_{i=1}^{K} DFT_i\right|}
$$

Where:

- $\bigcup_{i=1}^{K} DFT_i$ = union of all years from the top-K documents
- $QFT$ = Query Focus Time (set of query-relevant years)

**In simple terms**: Of all the years mentioned in retrieved documents, what fraction are actually relevant to the query?

## Example

Query: *"Events in 2020"* → QFT = `{2020}`

Retrieved documents:
- Doc 1: DFT = `{2020, 2021}`
- Doc 2: DFT = `{2019}`

Union of all years: `{2019, 2020, 2021}`

Relevant years: `{2020}`

**Year Precision** = 1/3 = 0.333

## Inputs
- `qft` - Query Focus Time (set of years)
- `dfts` - List of Document Focus Times
- `k` - Cutoff for top-K documents

## Output
- Range: [0, 1], higher is better

## Code Example

```python
from tempoeval.metrics import TemporalYearPrecision

metric = TemporalYearPrecision()
score = metric.compute(
    qft={2020, 2021},
    dfts=[{2020, 2022}, {2019}],
    k=2
)
print(f"Year Precision@2: {score}")  # 0.333 (1 relevant out of 3 total years)
```

!!! tip "When to use Year Precision"
    Use Year Precision when you care about temporal noise—retrieving documents that mention too many irrelevant years dilutes focus.

!!! note "Comparison with Temporal Precision"
    - **Temporal Precision**: Measures document-level relevance (what fraction of docs are relevant?)
    - **Year Precision**: Measures year-level relevance (what fraction of years are relevant?)
