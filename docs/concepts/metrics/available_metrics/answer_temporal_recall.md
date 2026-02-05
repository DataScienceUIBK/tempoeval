# Answer Temporal Recall

<span class="metric-badge generation">Generation Metric</span>

Answer Temporal Recall measures whether the generated answer addresses all query-relevant time periods. A low score indicates the answer missed important temporal aspects of the question.

## Formula

$$
\text{AnswerTemporalRecall} = \frac{|AFT \cap QFT|}{|QFT|}
$$

Where:

- $AFT$ = Answer Focus Time (years mentioned in the generated answer)
- $QFT$ = Query Focus Time (years relevant to the query)

**In simple terms**: Of all the years the query asks about, what fraction are mentioned in the answer?

## Example

Query: *"What happened from 2019 to 2021?"* → QFT = `{2019, 2020, 2021}`

Answer: *"In 2020, the pandemic began. By 2021, vaccines were available."* → AFT = `{2020, 2021}`

Answer Temporal Recall = 2/3 = 0.667 (missing 2019)

## Inputs
- `qft` - Query Focus Time (set of years)
- `aft` - Answer Focus Time (set of years)
- Or: `query` and `answer` (strings, will extract Focus Times)

## Output
- Range: [0, 1], higher is better

## Code Example

```python
from tempoeval.core import extract_qft, extract_aft
from tempoeval.metrics import AnswerTemporalRecall

# Extract Focus Times
qft = extract_qft("What happened from 2019 to 2021?")
answer = "In 2020, the pandemic began. By 2021, vaccines were available."
aft = extract_aft(answer)

# Compute metric
metric = AnswerTemporalRecall()
score = metric.compute(qft=qft, aft=aft)
print(f"Answer Temporal Recall: {score}")  # 0.667
```

!!! warning "Common Issue"
    Low Answer Temporal Recall can indicate:
    - The retrieval system didn't provide documents from all relevant years
    - The generation model focused on only part of the query's temporal scope
    - The answer is incomplete

!!! tip "When to use this metric"
    Use Answer Temporal Recall when:
    - Queries ask about multiple years or time periods
    - Comprehensive temporal coverage is important
    - You want to ensure the answer doesn't ignore part of the query's time scope
