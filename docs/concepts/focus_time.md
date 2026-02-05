# Focus Time

**Focus Time** is the core innovation behind TempoEval. It represents the specific time period(s) that a piece of text is *semantically about*—not when it was written, but what time it discusses.

---

## The Problem with Traditional Metrics

Traditional text evaluation relies on surface-level text matching:

!!! failure "Text Overlap (ROUGE, BLEU)"
    ```
    Query: "What happened in 2020?"
    Doc A: "Events in 2020"  ← Relevance: High (keyword match)
    Doc B: "Events in 2021"  ← Relevance: High (similar text)
    ```
    **Problem**: "2020" and "2021" are only one character apart, but **temporally distinct**.

!!! warning "LLM-as-a-Judge"
    - 💸 **Expensive**: $0.01-$0.10 per evaluation
    - 🐌 **Slow**: 1-5 seconds per query
    - 🎲 **Non-deterministic**: Different results on reruns
    - 🔍 **Opaque**: Hard to debug why something scored low

---

## The Solution: Focus Time

Focus Time converts text into **sets of years** (e.g., `{2020, 2021}`), enabling:

✅ **Fast evaluation** with set operations (intersection, union)
✅ **Cheap computation** (no API calls)
✅ **Deterministic results** (same input = same output)
✅ **Interpretable scores** (you can see which years match/mismatch)

<div class="formula-box">
**Core Principle**: Temporal relevance = Set overlap between Query Focus Time (QFT) and Document Focus Time (DFT)

$$
\text{Relevant} = |QFT \cap DFT| > 0
$$

</div>

---

## Three Types of Focus Time

<div class="timeline">

<div class="timeline-item">

### 1. Query Focus Time (QFT)

The time period the **user is asking about**.

**Examples**:

- *"What happened in **2020**?"* → QFT = `{2020}`
- *"Events in **the 1990s**"* → QFT = `{1990, 1991, ..., 1999}`
- *"During **World War II**"* → QFT = `{1939, 1940, ..., 1945}`
- *"**Recent** climate policy"* → QFT = `{2023, 2024, 2025, 2026}` (context-dependent)

```python
from tempoeval.core import extract_qft

qft = extract_qft("What happened during the 2008 financial crisis?")
print(qft.years)  # {2008}
```

</div>

<div class="timeline-item">

### 2. Document Focus Time (DFT)

The time period the **document discusses**.

!!! warning "Not the Publication Date"
    A document published in 2024 **about** the 2008 crisis has DFT = `{2008}`, not `{2024}`.

**Examples**:

- *"The collapse of Lehman Brothers in **2008** triggered..."* → DFT = `{2008}`
- *"From **1990 to 1995**, the economy grew..."* → DFT = `{1990, 1991, 1992, 1993, 1994, 1995}`
- *"The **Victorian era** saw major reforms."* → DFT = `{1837, 1838, ..., 1901}`

```python
from tempoeval.core import extract_dft

doc = "The COVID-19 pandemic began in 2019 and peaked in 2020."
dft = extract_dft(doc)
print(dft.years)  # {2019, 2020}
```

</div>

<div class="timeline-item">

### 3. Answer Focus Time (AFT)

The time period mentioned in the **generated answer**.

**Use case**: Verify the answer discusses the right time period.

**Example**:

- Query: *"When did the Cold War end?"*
- Answer: *"The Cold War ended in **1991** with the dissolution of the USSR."*
- AFT = `{1991}` ✅

```python
from tempoeval.core import extract_aft

answer = "The event occurred in 2015 and had lasting effects through 2018."
aft = extract_aft(answer)
print(aft.years)  # {2015, 2018}
```

</div>

</div>

---

## Extraction Methods

TempoEval supports **three extraction methods** with different trade-offs:

| Method | Speed | Cost | Accuracy | Best For |
|--------|-------|------|----------|----------|
| <span class="metric-badge retrieval">REGEX</span> | ⚡ **Instant** | 💰 **Free** | ✓ Good | Explicit years (*"in 2020"*) |
| <span class="metric-badge generation">TEMPORAL TAGGER</span> | 🔵 **Fast** | 💰 **Free** | ✓✓ Better | Complex expressions (*"last decade"*) |
| <span class="metric-badge reasoning">LLM</span> | 🔴 **Slow** | 💸 **Paid** | ✓✓✓ Best | Implicit references (*"during COVID"*) |

### Method 1: REGEX (Default)

Fast pattern matching for explicit years + rule-based mappings.

=== "Explicit Years"

    ```python
    from tempoeval.core.focus_time import QueryFocusTime

    extractor = QueryFocusTime()
    qft = extractor.extract("Events in 2020", use_regex=True)
    print(qft.years)  # {2020}
    ```

=== "Rule-Based Mappings"

    ```python
    # Rule-based mappings handle common implicit references
    qft = extractor.extract("During World War II", use_regex=True)
    print(qft.years)  # {1939, 1940, 1941, 1942, 1943, 1944, 1945}
    ```

=== "Year Ranges"

    ```python
    qft = extractor.extract("From 2015 to 2018", use_regex=True)
    print(qft.years)  # {2015, 2016, 2017, 2018}
    ```

### Method 2: TEMPORAL TAGGER

External temporal tagging library. **Multiple tagger options available**:

| Tagger | Speed | Install | Description |
|--------|-------|---------|-------------|
| **DATEPARSER** ⭐ | Fast | `pip install dateparser` | **RECOMMENDED** - Pure Python, multi-language |
| **PARSEDATETIME** | Fast | `pip install parsedatetime` | English natural language |
| **FAST_PARSE_TIME** | Ultra-fast | `pip install fast-parse-time` | Sub-millisecond |
| **HEIDELTIME** | Very slow | `pip install py_heideltime` | Java-based (requires Java JDK) |

```python
from tempoeval.core.focus_time import QueryFocusTime, TemporalTagger

extractor = QueryFocusTime()

# Use dateparser (default, recommended - fastest pure Python)
qft = extractor.extract(
    "What happened last decade?",
    use_regex=False,
    use_temporal_tagger=True,
    tagger=TemporalTagger.DATEPARSER
)
print(qft.years)  # Extracted years from dateparser

# Use HeidelTime (slower, requires Java)
qft = extractor.extract(
    "What happened last decade?",
    use_temporal_tagger=True,
    tagger=TemporalTagger.HEIDELTIME
)
```

**Handles**:

- Relative expressions: *"last year", "next month"*
- Fuzzy periods: *"early 2000s", "mid-1990s"*
- Complex syntax: *"between January 2020 and March 2021"*

### Method 3: LLM

Uses language models to infer implicit temporal references.

```python
from tempoeval.core.focus_time import QueryFocusTime
from tempoeval.llm import OpenAIProvider

llm = OpenAIProvider(model="gpt-4o")
extractor = QueryFocusTime(llm=llm)

qft = extractor.extract(
    "During the Great Depression",
    use_regex=True,
    use_llm=True
)
print(qft.years)  # {1929, 1930, ..., 1939}
```

**Handles**:

- Historical events: *"during WWII", "Victorian era"*
- Cultural references: *"the dot-com bubble", "the pandemic"*
- Contextual inference: *"when Obama was president"*

### Combining Methods

You can enable multiple methods - years will be **merged into a unique set**:

```python
# Combine REGEX + temporal tagger + LLM for maximum coverage
qft = extractor.extract(
    query,
    use_regex=True,           # Fast explicit extraction
    use_temporal_tagger=True, # Parse complex expressions
    use_llm=True              # Handle implicit references
)
```

---

## Set Operations on Focus Time

Once extracted, Focus Times support standard set operations:

### Intersection (Overlap)

```python
qft = FocusTime.from_years({2020, 2021})
dft = FocusTime.from_years({2021, 2022})

overlap = qft & dft  # {2021}
print(len(overlap))  # 1 year overlap
```

### Jaccard Similarity

```python
similarity = qft.jaccard(dft)
# similarity = |intersection| / |union|
# = 1 / 3 = 0.333
```

### Coverage Check

```python
is_relevant = qft.covers(dft)  # True if any overlap
```

---

## Practical Example

Let's evaluate a retrieval system:

```python
from tempoeval.core import extract_qft, extract_dft
from tempoeval.metrics import TemporalPrecision

# Query
query = "What caused the 2008 financial crisis?"
qft = extract_qft(query)  # {2008}

# Retrieved Documents
docs = [
    "The collapse of Lehman Brothers in 2008...",  # Relevant ✅
    "The 1929 Wall Street Crash was...",           # Irrelevant ❌
    "The 2020 pandemic caused...",                  # Irrelevant ❌
]
dfts = [extract_dft(doc) for doc in docs]
# dfts = [{2008}, {1929}, {2020}]

# Evaluate
metric = TemporalPrecision(use_focus_time=True)
score = metric.compute(qft=qft, dfts=dfts, k=3)

print(f"Temporal Precision@3: {score}")  # 0.333 (1/3 relevant)
```

**Analysis**:

- Only Doc 1 has `DFT ∩ QFT = {2008} ∩ {2008} = {2008}` ✅
- Doc 2: `{1929} ∩ {2008} = ∅` ❌
- Doc 3: `{2020} ∩ {2008} = ∅` ❌

---

## Advantages of Focus Time

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg } **10,000x Faster**

    ---

    Set operations vs. LLM API calls: milliseconds vs. seconds

-   :material-cash-remove:{ .lg } **100% Free**

    ---

    No API costs for REGEX and Temporal Tagger methods

-   :material-check-circle:{ .lg } **Deterministic**

    ---

    Same input always produces same output

-   :material-magnify:{ .lg } **Interpretable**

    ---

    See exactly which years match or mismatch

-   :material-scale-balance:{ .lg } **Complements LLMs**

    ---

    Use Focus Time for speed, LLM for complex edge cases

</div>

---

## When to Use Which Method?

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Production at scale (1000s of queries) | **REGEX** | Instant, free, good enough |
| Benchmarking on datasets | **REGEX** or **TEMPORAL TAGGER** | Reproducible, no API keys |
| Complex relative expressions | **TEMPORAL TAGGER (dateparser)** | Pure Python, fast, handles "last decade" |
| Java environment available | **TEMPORAL TAGGER (heideltime)** | Most accurate for complex expressions |
| Implicit historical references | **LLM** | Only method that handles *"during COVID"* |
| Maximum coverage | **REGEX + TAGGER + LLM** | All methods combined, years merged |

---

## What's Next?

- **[Computation Modes](modes.md)** - How to use Focus Time, LLM-judge, or Gold labels
- **[Metrics Overview](metrics/overview/index.md)** - See how metrics use Focus Time
- **[Temporal Precision](metrics/available_metrics/temporal_precision.md)** - Example metric using Focus Time
