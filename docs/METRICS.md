# TempoEval Metrics Guide

A comprehensive guide to all 16 temporal metrics with system flow, detailed explanations, examples, and expected outputs.

---

## System Flow Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Query: "How has renewable energy adoption changed since 2015?"         │ │
│  │  Retrieved Docs: [doc1, doc2, doc3, ...]                                │ │
│  │  Generated Answer: "In 2015, renewables were 10%..."                    │ │
│  │  Gold IDs (optional): [doc1, doc5]                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL QUERY TYPE CLASSIFICATION                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  temporal_intent: "change_over_time"                                    │ │
│  │  temporal_focus: "trends_changes_and_cross_period"                      │ │
│  │  key_time_anchors: ["2015", "present"]                                  │ │
│  │  is_cross_period: True                                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   LAYER 1            │ │   LAYER 2            │ │   LAYER 3            │
│   RETRIEVAL (9)      │ │   GENERATION (4)     │ │   REASONING (4)      │
│   ══════════         │ │   ══════════         │ │   ═════════          │
│                      │ │                      │ │                      │
│ • Temporal Recall    │ │ • Temporal           │ │ • Event Ordering     │
│ • Temporal Precision │ │   Faithfulness       │ │ • Duration Accuracy  │
│ • Temporal NDCG      │ │ • Temporal           │ │ • Cross-Period       │
│ • Temporal MRR       │ │   Hallucination      │ │   Reasoning          │
│ • Temporal Coverage  │ │ • Temporal           │ │ • Temporal           │
│ • Anchor Coverage    │ │   Coherence          │ │   Consistency        │
│ • Temporal Diversity │ │ • Answer Temporal    │ │                      │
│ • Temporal Relevance │ │   Alignment          │ │                      │
│ • NDCG Full Coverage │ │                      │ │                      │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TEMPOSCORE                                         │
│    Weighted combination of metrics → Final Score: 0.82                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION RESULT                                   │
│  {                                                                            │
│    "temporal_recall@10": 0.75,                                                │
│    "temporal_precision@10": 0.83,                                             │
│    "temporal_coverage": 1.0,                                                  │
│    "temporal_faithfulness": 0.91,                                             │
│    "temporal_coherence": 0.88,                                                │
│    "cross_period_reasoning": 0.85,                                            │
│    "tempo_score": 0.82                                                        │
│  }                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Temporal Query Type Classification

```
                           ┌──────────────────────┐
                           │   TEMPORAL QUERY     │
                           └──────────┬───────────┘
                                      │
       ┌──────────────┬───────────────┼───────────────┬──────────────┐
       ▼              ▼               ▼               ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  SPECIFIC   │ │  DURATION   │ │  RECENCY    │ │ CROSS-PERIOD│ │  ORDERING   │
│    TIME     │ │             │ │             │ │             │ │             │
├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
│ "When did   │ │ "How long   │ │ "What are   │ │ "How has X  │ │ "What       │
│  X happen?" │ │  did X      │ │  recent     │ │  changed    │ │  happened   │
│             │ │  last?"     │ │  advances?" │ │  since Y?"  │ │  first?"    │
├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
│ Keywords:   │ │ Keywords:   │ │ Keywords:   │ │ Keywords:   │ │ Keywords:   │
│ • when      │ │ • how long  │ │ • recent    │ │ • changed   │ │ • first     │
│ • what year │ │ • duration  │ │ • latest    │ │ • since     │ │ • then      │
│ • what date │ │ • lasted    │ │ • new       │ │ • evolution │ │ • before    │
│             │ │ • for X yrs │ │ • nowadays  │ │ • over time │ │ • after     │
├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
│ Key Metric: │ │ Key Metric: │ │ Key Metric: │ │ Key Metric: │ │ Key Metric: │
│ Temporal    │ │ Duration    │ │ Temporal    │ │ Temporal    │ │ Event       │
│ Precision   │ │ Accuracy    │ │ Diversity   │ │ Coverage    │ │ Ordering    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

**Implemented temporal_intent values:**
- `when`, `specific_time`
- `duration`
- `recency`, `ongoing_status`
- `change_over_time`, `trends_changes_and_cross_period`
- `order`, `before_after`, `timeline`
- `period_definition`
- `none`

---

## Layer 1: Retrieval Metrics (7 Metrics)

---

### TemporalRecall@K

**What it measures:** Out of all the documents that *should* have been retrieved (gold documents), how many did your system actually find in the top-K results?

**Example:**
```
Query: "What happened during the French Revolution?"

Gold documents: [doc_1789, doc_1792, doc_1793, doc_1799]  (4 documents we NEED)
Retrieved Top-5: [doc_1789, doc_1792, doc_unrelated, doc_1793, doc_modern]

                    Overlap
                   ┌───┴───┐
Gold:    [ A ][ B ][ C ][ D ]
              │         │
Retrieved: [ A ]   [ C ]
              │         │
              └────┬────┘
                   │
         Found 3 out of 4

Temporal Recall@5 = 3/4 = 0.75
```

**Expected Output:**
```python
score = temporal_recall.compute(
    retrieved_ids=["doc_1789", "doc_1792", "doc_unrelated", "doc_1793", "doc_modern"],
    gold_ids=["doc_1789", "doc_1792", "doc_1793", "doc_1799"],
    k=5
)
>>> 0.75
```

**When it's useful:** Use this when you want to ensure your system doesn't *miss* important historical documents.

**Implementation:** `tempoeval/metrics/retrieval/temporal_recall.py`

---

### TemporalPrecision@K

**What it measures:** Out of the K documents your system retrieved, how many are actually temporally relevant to the query?

**Example:**
```
Query: "When did Bitcoin introduce pruning?"

Retrieved Docs:
┌─────┬────────────────────────────────────────┬──────────────┐
│ Pos │ Document Content                       │ LLM Verdict  │
├─────┼────────────────────────────────────────┼──────────────┤
│  1  │ "Pruning was added in v0.11 (2015)..." │ ✅ Relevant  │
│  2  │ "Bitcoin uses blockchain technology"   │ ❌ No dates  │
│  3  │ "In January 2015, the feature..."      │ ✅ Relevant  │
│  4  │ "Pruning reduces storage needs"        │ ❌ No dates  │
│  5  │ "The 2017 update improved..."          │ ❌ Wrong era │
└─────┴────────────────────────────────────────┴──────────────┘

Position-weighted calculation:
• Position 1 relevant: P@1 = 1/1 = 1.0
• Position 3 relevant: P@3 = 2/3 = 0.67

Temporal Precision = (1.0 + 0.67) / 2 = 0.83
```

**Expected Output:**
```python
score = temporal_precision.compute(
    query="When did Bitcoin introduce pruning?",
    retrieved_docs=[doc1, doc2, doc3, doc4, doc5],
    temporal_focus="specific_time",
    k=5
)
>>> 0.83
```

**When it's useful:** Use this when you want to minimize irrelevant documents cluttering your results.

**Implementation:** `tempoeval/metrics/retrieval/temporal_precision.py`

---

### TemporalNDCG@K

**What it measures:** Not just *whether* you found the right documents, but *where* you ranked them. Documents appearing higher in the list contribute more to the score.

**Example:**
```
Query: "When did the Berlin Wall fall?"
Gold doc: doc_1989_wall_fall

Scenario A: Gold doc at position 1
┌─────┬──────────────────────┐
│ Pos │ Document             │
├─────┼──────────────────────┤
│  1  │ doc_1989_wall_fall ★ │ ← GOLD DOC HERE
│  2  │ other_doc            │
│  3  │ another_doc          │
└─────┴──────────────────────┘
NDCG@10 = 1.0 ✅ (Perfect!)

Scenario B: Gold doc at position 10
┌─────┬──────────────────────┐
│ Pos │ Document             │
├─────┼──────────────────────┤
│  1  │ irrelevant_doc       │
│ ... │ ...                  │
│ 10  │ doc_1989_wall_fall ★ │ ← GOLD DOC BURIED
└─────┴──────────────────────┘
NDCG@10 = 0.29 ❌ (Poor ranking)
```

**Expected Output:**
```python
# Scenario A
score = temporal_ndcg.compute(
    retrieved_ids=["doc_1989_wall_fall", "other", "another"],
    gold_ids=["doc_1989_wall_fall"],
    k=10
)
>>> 1.0

# Scenario B
score = temporal_ndcg.compute(
    retrieved_ids=["irrelevant"] * 9 + ["doc_1989_wall_fall"],
    gold_ids=["doc_1989_wall_fall"],
    k=10
)
>>> 0.29
```

**When it's useful:** Use this when ranking quality matters—when users typically only look at the first few results.

**Implementation:** `tempoeval/metrics/retrieval/temporal_ndcg.py`

---

### TemporalMRR

**What it measures:** How quickly does your system find the *first* relevant document? The score is the inverse of the position.

**Example:**
```
Query: "When was the Magna Carta signed?"

Case 1: Gold doc at position 1
├── MRR = 1/1 = 1.0 ✅

Case 2: Gold doc at position 3
├── MRR = 1/3 = 0.33 ⚠️

Case 3: Gold doc at position 10
├── MRR = 1/10 = 0.1 ❌

Case 4: Gold doc not found in top-K
├── MRR = 0.0 ❌
```

**Expected Output:**
```python
# Gold doc at position 3
score = temporal_mrr.compute(
    retrieved_ids=["irrelevant1", "irrelevant2", "magna_carta_doc", "other"],
    gold_ids=["magna_carta_doc"],
    k=10
)
>>> 0.33
```

**When it's useful:** Use this for navigational queries where the user wants *one* correct answer quickly.

**Implementation:** `tempoeval/metrics/retrieval/temporal_mrr.py`

---

### TemporalCoverage

**What it measures:** For change/trend queries, do we have documents covering BOTH the old period AND the new period?

**Example:**
```
Query: "How has electric car adoption changed since 2015?"

Required Periods:
├── Baseline (2015): Need docs about EV market in 2015
└── Comparison (2024): Need docs about current EV market

Retrieved Docs Analysis:
┌─────┬─────────────────────────────────┬──────────┬────────────┐
│ Doc │ Content                         │ Baseline │ Comparison │
├─────┼─────────────────────────────────┼──────────┼────────────┤
│  1  │ "In 2015, EVs were 1% of..."    │ ✅       │            │
│  2  │ "Tesla's 2023 sales..."         │          │ ✅         │
│  3  │ "The history of EVs..."         │          │            │
│  4  │ "Current EV market shows..."    │          │ ✅         │
└─────┴─────────────────────────────────┴──────────┴────────────┘

Baseline covered: ✅ (Doc 1)
Comparison covered: ✅ (Docs 2, 4)

Temporal Coverage = (1 + 1) / 2 = 1.0 (Perfect!)

If we only had Docs 2, 3, 4:
Temporal Coverage = (0 + 1) / 2 = 0.5 (Missing baseline!)
```

**Expected Output:**
```python
score = temporal_coverage.compute(
    query="How has electric car adoption changed since 2015?",
    retrieved_docs=[doc1, doc2, doc3, doc4],
    baseline_anchor="2015",
    comparison_anchor="2024",
    temporal_focus="change_over_time"
)
>>> 1.0
```

**When it's useful:** Use this for queries spanning multiple time periods where comprehensive coverage matters.

**Implementation:** `tempoeval/metrics/retrieval/temporal_coverage.py`

---

### AnchorCoverage

**What it measures:** Does your retrieval find documents mentioning the specific dates/events (anchors) that the query needs?

**Example:**
```
Query: "Events between the stock market crash and Pearl Harbor"

Required anchors: ["1929", "1941", "Great Depression", "Pearl Harbor"]
Retrieved anchors: ["1929", "1933 New Deal", "1941", "Pearl Harbor"]

Precision = |Retrieved ∩ Required| / |Retrieved|
         = 3 / 4 = 0.75

Recall = |Retrieved ∩ Required| / |Required|
       = 3 / 4 = 0.75

Anchor Coverage (F1) = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
```

**Expected Output:**
```python
score = anchor_coverage.compute(
    retrieved_anchors=["1929", "1933 New Deal", "1941", "Pearl Harbor"],
    required_anchors=["1929", "1941", "Great Depression", "Pearl Harbor"]
)
>>> 0.75
```

**When it's useful:** Use this when specific dates or events are explicitly mentioned in the query.

**Implementation:** `tempoeval/metrics/retrieval/anchor_coverage.py`

---

### TemporalDiversity

**What it measures:** Do your retrieved documents represent a variety of time periods, or are they all from the same era?

**Example:**
```
Query: "History of computing"

Low Diversity (Score: 0.2):
┌──────────────────────────────────────────────────┐
│ All documents about the 1990s internet era      │
│ • "The 1995 internet boom..."                   │
│ • "In 1996, websites grew..."                   │
│ • "The 1997 dot-com era..."                     │
│ → All clustered in one decade ❌                │
└──────────────────────────────────────────────────┘

High Diversity (Score: 0.95):
┌──────────────────────────────────────────────────┐
│ Documents span multiple eras                     │
│ • "ENIAC was built in 1945..."        (1940s)   │
│ • "The Intel 4004 in 1971..."         (1970s)   │
│ • "The IBM PC launched in 1981..."    (1980s)   │
│ • "The World Wide Web began in 1991..." (1990s) │
│ • "The iPhone was introduced in 2007..." (2000s)│
│ → Covers 5 distinct decades ✅                  │
└──────────────────────────────────────────────────┘
```

**Expected Output:**
```python
score = temporal_diversity.compute(
    retrieved_docs=[
        "ENIAC was built in 1945...",
        "The Intel 4004 was released in 1971...",
        "The IBM PC launched in 1981...",
        "The World Wide Web began in 1991...",
        "The iPhone was introduced in 2007..."
    ],
    k=5
)
>>> 0.95
```

**When it's useful:** Use this for broad historical queries where you want comprehensive temporal coverage.

**Implementation:** `tempoeval/metrics/retrieval/temporal_diversity.py`

---

### TemporalRelevance

**What it measures:** Simple ratio of temporally relevant documents, without position weighting. Unlike TemporalPrecision which weights by position, this treats all positions equally.

**Example:**
```
Query: "When did WWI begin?"

Retrieved Docs:
┌─────┬────────────────────────────────────────┬──────────────┐
│ Pos │ Document Content                       │ LLM Verdict  │
├─────┼────────────────────────────────────────┼──────────────┤
│  1  │ "WWI started July 28, 1914..."         │ ✅ Relevant  │
│  2  │ "The war had many causes"              │ ❌ No dates  │
│  3  │ "The Armistice was signed in 1918..."  │ ✅ Relevant  │
│  4  │ "General overview of warfare"          │ ❌ No dates  │
│  5  │ "The assassination of Franz Ferdinand" │ ❌ No dates  │
└─────┴────────────────────────────────────────┴──────────────┘

Simple ratio (no position weighting):
Relevant docs: 2 out of 5

TemporalRelevance = 2/5 = 0.4
```

**Expected Output:**
```python
from tempoeval.metrics import TemporalRelevance

relevance = TemporalRelevance()
relevance.llm = llm  # Set LLM provider

score = relevance.compute(
    query="When did WWI begin?",
    retrieved_docs=[doc1, doc2, doc3, doc4, doc5],
    temporal_focus="specific_time",
    k=5
)
>>> 0.4
```

**When it's useful:** Use this when you want a simple percentage of temporally relevant docs, treating all positions equally.

**Implementation:** `tempoeval/metrics/retrieval/temporal_relevance.py`

---

### NDCGFullCoverage

**What it measures:** NDCG score ONLY when temporal coverage is 100%. Returns NaN if the retrieval set doesn't cover both baseline AND comparison periods.

**Example:**
```
Query: "How has renewable energy adoption changed since 2015?"

Scenario A: Full coverage (baseline + comparison)
┌─────┬─────────────────────────────────┬──────────┬────────────┐
│ Doc │ Content                         │ Baseline │ Comparison │
├─────┼─────────────────────────────────┼──────────┼────────────┤
│  1  │ "In 2015, renewables were 10%"  │ ✅       │            │
│  2  │ "By 2024, renewables are 30%"   │          │ ✅         │
└─────┴─────────────────────────────────┴──────────┴────────────┘
Coverage = 1.0 (both periods covered)
NDCG = 0.85
NDCGFullCoverage = 0.85 ✅

Scenario B: Partial coverage (missing baseline)
┌─────┬─────────────────────────────────┬──────────┬────────────┐
│ Doc │ Content                         │ Baseline │ Comparison │
├─────┼─────────────────────────────────┼──────────┼────────────┤
│  1  │ "General renewable energy info" │          │            │
│  2  │ "Current 2024 trends..."        │          │ ✅         │
└─────┴─────────────────────────────────┴──────────┴────────────┘
Coverage = 0.5 (missing baseline!)
NDCG = 0.85
NDCGFullCoverage = NaN ❌ (penalized for incomplete coverage)
```

**Expected Output:**
```python
from tempoeval.metrics import NDCGFullCoverage

ndcg_fc = NDCGFullCoverage()
ndcg_fc.llm = llm  # For coverage check

# Full coverage case
score = ndcg_fc.compute(
    query="How has renewable energy adoption changed since 2015?",
    retrieved_ids=["doc1", "doc2"],
    gold_ids=["doc1", "doc2"],
    retrieved_docs=["In 2015, renewables were 10%", "By 2024, renewables are 30%"],
    baseline_anchor="2015",
    comparison_anchor="2024",
    k=10
)
>>> 0.85  # Returns NDCG since coverage=1.0

# Partial coverage case
score = ndcg_fc.compute(
    query="...",
    retrieved_docs=["Current 2024 trends..."],  # Missing baseline!
    ...
)
>>> NaN  # Returns NaN since coverage<1.0
```

**When it's useful:** Use this for cross-period queries where you want good ranking ONLY if the retrieval set is complete.

**Implementation:** `tempoeval/metrics/retrieval/ndcg_full_coverage.py`

---

## Layer 2: Generation Metrics (4 Metrics)

---

### TemporalFaithfulness

**What it measures:** Are all the dates and temporal claims in the generated answer actually supported by the retrieved documents?

**Example:**
```
Retrieved Context:
┌──────────────────────────────────────────────────────────────┐
│ "Bitcoin Core 0.11.0, released in July 2015, introduced     │
│  block pruning, allowing nodes to delete old block data."   │
└──────────────────────────────────────────────────────────────┘

GOOD Generated Answer: "Bitcoin introduced pruning in 2015."
├── Claim "2015" → Found in context ✅
└── Temporal Faithfulness = 1.0

BAD Generated Answer: "Bitcoin introduced pruning in version 0.14 in 2017."
┌─────────────────────────┬───────────────┬────────────────────┐
│ Temporal Claim          │ In Context?   │ Verdict            │
├─────────────────────────┼───────────────┼────────────────────┤
│ "version 0.14"          │ ❌ No         │ NOT SUPPORTED      │
│ "in 2017"               │ ❌ No         │ CONTRADICTED       │
│                         │ (says 2015)   │                    │
└─────────────────────────┴───────────────┴────────────────────┘
└── Temporal Faithfulness = 0.0 (Completely unfaithful!)
```

**Expected Output:**
```python
# Good answer
score = temporal_faithfulness.compute(
    answer="Bitcoin introduced pruning in 2015.",
    contexts=["Bitcoin Core 0.11.0, released in July 2015, introduced block pruning..."]
)
>>> 1.0

# Bad answer
score = temporal_faithfulness.compute(
    answer="Bitcoin introduced pruning in version 0.14 in 2017.",
    contexts=["Bitcoin Core 0.11.0, released in July 2015, introduced block pruning..."]
)
>>> 0.0
```

**When it's useful:** Use this to ensure your RAG system doesn't add temporal details not found in sources.

**Implementation:** `tempoeval/metrics/generation/temporal_faithfulness.py`

---

### TemporalHallucination

**What it measures:** Did the model invent dates or temporal facts that aren't in the context? This is the inverse of faithfulness.

**Example:**
```
Context: "Einstein published his theory of relativity."
         (Note: No dates mentioned!)

Generated Answer: "Einstein published his theory of relativity in 1905."

Analysis:
┌───────────────────────────────────────────────────────────────┐
│ Temporal Claim: "in 1905"                                     │
│                                                               │
│ Is "1905" in the context? ❌ NO                               │
│                                                               │
│ The model added a date that wasn't in the source!            │
│ This is a TEMPORAL HALLUCINATION                              │
└───────────────────────────────────────────────────────────────┘

Hallucination Score = 1.0 (High - hallucination detected!)
```

**Expected Output:**
```python
score = temporal_hallucination.compute(
    answer="Einstein published his theory of relativity in 1905.",
    contexts=["Einstein published his theory of relativity."]
)
>>> 1.0  # High score = hallucination detected
```

**When it's useful:** Use this to catch systems that confidently state dates without evidence.

**Implementation:** `tempoeval/metrics/generation/temporal_hallucination.py`

---

### TemporalCoherence

**What it measures:** Is the answer internally consistent with itself regarding time? Do the temporal statements make logical sense together?

**Example:**
```
COHERENT Answer:
"The war began in 1914 and ended in 1918, lasting four years."

Timeline Analysis:
    1914         1918
      │            │
      ▼            ▼
   STARTED       ENDED
      │            │
      └────4 yrs───┘

✅ 1918 - 1914 = 4 years (MATCHES stated duration)
Temporal Coherence = 1.0

─────────────────────────────────────────────────────

INCOHERENT Answer:
"The project started in 2020 and was completed in 2018 after 5 years."

Timeline Analysis:
    2018         2020
      │            │
      ▼            ▼
  COMPLETED     STARTED
      │            │
      └─── ??? ────┘

❌ ERROR 1: End date (2018) before start (2020) - IMPOSSIBLE!
❌ ERROR 2: Stated "5 years" but 2020 - 2018 = 2 years

Issues Found: 2
Temporal Coherence = 0.33
```

**Expected Output:**
```python
# Coherent answer
score = temporal_coherence.compute(
    answer="The war began in 1914 and ended in 1918, lasting four years."
)
>>> 1.0

# Incoherent answer
score = temporal_coherence.compute(
    answer="The project started in 2020 and was completed in 2018 after 5 years."
)
>>> 0.33
```

**When it's useful:** Use this to catch logical errors in temporal reasoning.

**Implementation:** `tempoeval/metrics/generation/temporal_coherence.py`

---

### AnswerTemporalAlignment

**What it measures:** Does the answer address the correct time period that the query asked about?

**Example:**
```
Query: "What was happening in Germany in 1923?"

ALIGNED Answer:
"In 1923, Germany experienced hyperinflation. The German mark 
became virtually worthless, requiring wheelbarrows of cash to 
buy basic goods."

Query asks about: 1923
Answer discusses: 1923 ✅
Alignment = 1.0

─────────────────────────────────────────────────────

MISALIGNED Answer:
"Germany was divided after WWII in 1945. The Berlin Wall 
was built in 1961 and fell in 1989."

Query asks about: 1923
Answer discusses: 1945, 1961, 1989 ❌

Alignment = 0.0 (Wrong time period entirely!)
```

**Expected Output:**
```python
# Aligned answer
score = answer_temporal_alignment.compute(
    query="What was happening in Germany in 1923?",
    answer="In 1923, Germany experienced hyperinflation...",
    temporal_focus="specific_time",
    target_anchors=["1923"]
)
>>> 1.0

# Misaligned answer
score = answer_temporal_alignment.compute(
    query="What was happening in Germany in 1923?",
    answer="Germany was divided after WWII in 1945...",
    temporal_focus="specific_time",
    target_anchors=["1923"]
)
>>> 0.0
```

**When it's useful:** Use this to ensure the model answers the *actual* temporal question asked.

**Implementation:** `tempoeval/metrics/generation/answer_temporal_alignment.py`

---

## Layer 3: Reasoning Metrics (4 Metrics)

---

### EventOrdering

**What it measures:** When the answer lists multiple events, are they in the correct chronological order?

**Example:**
```
Query: "Describe the key events leading to American independence"

Correct Order:
1. Stamp Act (1765)
2. Boston Tea Party (1773)
3. Declaration (1776)
4. Treaty of Paris (1783)

CORRECT Answer Order:
1. Stamp Act ✅
2. Boston Tea Party ✅
3. Declaration ✅
4. Treaty of Paris ✅
Event Ordering = 1.0

─────────────────────────────────────────────────────

INCORRECT Answer: "After the Declaration of Independence, the 
Boston Tea Party occurred, leading to the Stamp Act."

┌─────────────────┬────────────┬──────────────┐
│ Pair            │ Correct?   │ Inversion?   │
├─────────────────┼────────────┼──────────────┤
│ Decl→Tea Party  │ ❌         │ YES          │
│ Tea Party→Stamp │ ❌         │ YES          │
└─────────────────┴────────────┴──────────────┘

Inversions: 2
Max possible: 3
Event Ordering = 1 - (2/3) = 0.33
```

**Expected Output:**
```python
score = event_ordering.compute(
    answer="The Stamp Act (1765) led to protests, followed by the Boston Tea Party (1773), then the Declaration of Independence (1776).",
    gold_order=["Stamp Act", "Boston Tea Party", "Declaration of Independence"]
)
>>> 1.0
```

**When it's useful:** Use this for timeline questions or narrative historical accounts.

**Implementation:** `tempoeval/metrics/reasoning/event_ordering.py`

---

### DurationAccuracy

**What it measures:** When the answer mentions how long something lasted, is the duration correct?

**Example:**
```
Query: "How long did the Hundred Years' War last?"

Context: "The war lasted from 1337 to 1453"
Actual Duration: 1453 - 1337 = 116 years

ACCURATE Answer: "The war lasted 116 years"
├── Claimed: 116 years
├── Actual: 116 years
└── Duration Accuracy = 1.0 ✅

INACCURATE Answer: "The war lasted exactly 100 years"
├── Claimed: 100 years
├── Actual: 116 years
├── Error: 16 years off
└── Duration Accuracy = 0.14 ❌

INACCURATE Answer: "The war lasted about 50 years"
├── Claimed: 50 years
├── Actual: 116 years
├── Error: 66 years off
└── Duration Accuracy = 0.0 ❌
```

**Expected Output:**
```python
score = duration_accuracy.compute(
    answer="The war lasted 116 years, from 1337 to 1453.",
    context="The Hundred Years' War was fought from 1337 to 1453."
)
>>> 1.0
```

**When it's useful:** Use this for questions about time spans, ages, or periods.

**Implementation:** `tempoeval/metrics/reasoning/duration_accuracy.py`

---

### CrossPeriodReasoning

**What it measures:** Can the model synthesize information from multiple time periods to form a coherent answer?

**Example:**
```
Query: "How did communication technology evolve from 1800 to 2000?"

EXCELLENT Cross-Period Answer:
┌──────────────────────────────────────────────────────────────────┐
│ "Communication evolved dramatically over two centuries:          │
│                                                                  │
│  1840s: Telegraph enabled instant long-distance messaging       │
│     ↓                                                           │
│  1870s: Telephone added voice communication                     │
│     ↓                                                           │
│  1890s: Radio broadcast to mass audiences                       │
│     ↓                                                           │
│  1920s: Television combined audio and visual                    │
│     ↓                                                           │
│  1990s: Internet connected the world digitally                  │
│                                                                  │
│  Each technology built upon its predecessors..."                │
└──────────────────────────────────────────────────────────────────┘

Evaluation:
├── Period Coverage: ✅ 5 distinct eras
├── Explicit Comparison: ✅ Shows progression
├── Causal Reasoning: ✅ "built upon predecessors"
├── Temporal Clarity: ✅ Clear date references
└── Cross-Period Reasoning = 0.95

─────────────────────────────────────────────────────

POOR Cross-Period Answer:
"The internet revolutionized communication in the 1990s."

├── Period Coverage: ❌ Only 1 era
├── Explicit Comparison: ❌ No comparison made
├── Causal Reasoning: ❌ No historical context
└── Cross-Period Reasoning = 0.2
```

**Expected Output:**
```python
score = cross_period_reasoning.compute(
    query="How did communication technology evolve from 1800 to 2000?",
    answer="Communication evolved from the telegraph in the 1840s, to telephone in 1876, radio in the 1890s, television in the 1920s, and finally the internet in the 1990s.",
    contexts=[...]
)
>>> 0.95
```

**When it's useful:** Use this for comparative or evolutionary historical questions.

**Implementation:** `tempoeval/metrics/reasoning/cross_period_reasoning.py`

---

### TemporalConsistency

**What it measures:** If you ask the same question multiple times, does the model give temporally consistent answers?

**Example:**
```
Query: "When was the first iPhone released?"

Run 1: "The iPhone was announced on January 9, 2007."
Run 2: "Apple released the iPhone on June 29, 2007."
Run 3: "The iPhone came out in 2007."
Run 4: "The first iPhone was released in 2008."

Consistency Analysis:
┌────────────────────────────────────────────────────────────┐
│ Run │ Date Mentioned    │ Consistent with others?          │
├─────┼───────────────────┼──────────────────────────────────┤
│  1  │ January 2007      │ ✅ (2007)                        │
│  2  │ June 2007         │ ✅ (2007)                        │
│  3  │ 2007              │ ✅ (2007)                        │
│  4  │ 2008              │ ❌ (WRONG YEAR!)                 │
└─────┴───────────────────┴──────────────────────────────────┘

3 out of 4 answers agree on 2007
Temporal Consistency = 0.75
```

**Expected Output:**
```python
score = temporal_consistency.compute(
    responses=[
        "The iPhone was announced on January 9, 2007.",
        "Apple released the iPhone on June 29, 2007.",
        "The iPhone came out in 2007.",
        "The first iPhone was released in 2008."
    ]
)
>>> 0.75
```

**When it's useful:** Use this to test reliability and reduce temporal variance in outputs.

**Implementation:** `tempoeval/metrics/reasoning/temporal_consistency.py`

---

## Composite Metric: TempoScore

**What it measures:** A single overall score combining multiple metrics, similar to how RAGAS combines metrics for general RAG evaluation.

**How it works:**
- You configure weights for different metrics
- The final score is a weighted/harmonic mean
- Different presets for different use cases

**Available Presets:**
- `TempoScore.create_rag_balanced()` - Balanced weights for RAG systems
- `TempoScore.create_retrieval_focused()` - Emphasizes retrieval quality
- `TempoScore.create_generation_focused()` - Emphasizes answer quality

**Example:**
```python
from tempoeval.metrics import TempoScore

# Use preset configuration
temposcore = TempoScore.create_rag_balanced()

score = temposcore.compute(
    temporal_recall=0.8,
    temporal_precision=0.7,
    temporal_faithfulness=0.9,
    temporal_coverage=0.6,
    cross_period_reasoning=0.75
)
>>> 0.76  # Weighted average

# Custom weights
temposcore = TempoScore(weights={
    "temporal_recall": 0.25,
    "temporal_precision": 0.25,
    "temporal_faithfulness": 0.35,
    "temporal_coverage": 0.15,
})
```

**Implementation:** `tempoeval/metrics/composite/tempo_score.py`

---

## Quick Reference Table

| Metric | Layer | LLM | Gold IDs | Best For |
|--------|-------|-----|----------|----------|
| TemporalRecall@K | Retrieval | ❌ | ✅ | Finding all relevant docs |
| TemporalPrecision@K | Retrieval | ✅ | ❌ | Avoiding irrelevant docs |
| TemporalNDCG@K | Retrieval | ❌ | ✅ | Ranking quality |
| TemporalMRR | Retrieval | ❌ | ✅ | First result quality |
| TemporalCoverage | Retrieval | ✅ | ❌ | Multi-period queries |
| AnchorCoverage | Retrieval | ❌ | ✅ | Specific date queries |
| TemporalDiversity | Retrieval | ❌ | ❌ | Broad historical queries |
| **TemporalRelevance** | Retrieval | ✅ | ❌ | Simple relevance ratio |
| **NDCGFullCoverage** | Retrieval | ✅ | ✅ | Cross-period ranking + coverage |
| TemporalFaithfulness | Generation | ✅ | ❌ | Source verification |
| TemporalHallucination | Generation | ✅ | ❌ | Detecting made-up dates |
| TemporalCoherence | Generation | ✅ | ❌ | Logical consistency |
| AnswerTemporalAlignment | Generation | ✅ | ❌ | Right time period |
| EventOrdering | Reasoning | ✅ | ⚠️ | Timeline accuracy |
| DurationAccuracy | Reasoning | ✅ | ✅ | Time span claims |
| CrossPeriodReasoning | Reasoning | ✅ | ❌ | Historical evolution |
| TemporalConsistency | Reasoning | ✅ | ❌ | Response reliability |
| TempoScore | Composite | Varies | Partial | Overall evaluation |

---

## When to Use Each Metric

| Use Case | Recommended Metrics |
|----------|---------------------|
| Evaluating a retriever only | Temporal Recall, NDCG, MRR, Coverage |
| Evaluating a RAG system | All Layer 1 + Temporal Faithfulness + TempoScore |
| Testing LLM temporal reasoning | Temporal Coherence, Event Ordering, Duration, Cross-Period |
| Benchmark comparison | TempoScore |
| Debugging temporal errors | All metrics |
| Recency queries | Temporal Diversity |
| Change/trend queries | Temporal Coverage, Cross-Period Reasoning |
| Specific date queries | Temporal Precision, Answer Alignment |

---

## See Also

- [Home](index.md) - Main documentation
- [Quick Start](getstarted/quickstart.md) - Get started guide
- [Tutorials](tutorials/index.md) - Practical guides

---

## Metric Quality Assurance (Meta-Evaluation)

TempoEval includes a **Meta-Evaluation Suite** to validate the correctness and robustness of its metrics. This suite ensures that the metrics accurately penalize incorrect or low-quality outputs.

### How it Works

The meta-evaluator takes a "good" input (high quality) and systematically perturbs it to create a "failure mode" (low quality). It then asserts that the metric score for the perturbed input is significantly worse than the original.

| Failure Mode | Perturbation | Target Metric | Expectation |
|--------------|--------------|---------------|-------------|
| **Date Scrambling** | Randomly changing years (e.g., 1999 -> 2025) | EventOrdering / Duration | Lower Score |
| **Event Shuffling** | Reordering sentences/events | EventOrdering | Lower Score |
| **Hallucination** | Injecting fake temporal claims | TemporalHallucination | Higher Score |

### Running Meta-Evaluation

```python
from tempoeval.meta_eval import MetricValidator, DateScrambler
from tempoeval.metrics import EventOrdering

# 1. Setup Validator
validator = MetricValidator()
scrambler = DateScrambler()

# 2. Define Test Case
test_case = TestCase(
    original_input={"answer": "Event A (1990) -> Event B (1995)"},
    perturbed_input={"answer": "Event A (2020) -> Event B (1800)"}, # Scrambled!
    perturbation_type="date_scramble",
    expected_result="lower"
)

# 3. Validate
report = validator.validate(EventOrdering(), [test_case])
print(report["passed"]) # Should be 1
```

---

## Efficiency & Cost Tracking

For SIGIR Resource Track compliance and "Green AI" reporting, TempoEval includes a built-in **Efficiency Tracker**.

### Features

- **Latency Tracking**: Measures execution time per metric (ms).
- **Cost Estimation**: Estimates USD cost based on character/token counts (approx. $0.005/1k input, $0.015/1k output for GPT-4o).
- **Breakdown**: Aggregates stats by metric type (Retrieval vs. Generation).

### Usage

```python
from tempoeval.core import evaluate
from tempoeval.efficiency import EfficiencyTracker

# Enable tracking
tracker = EfficiencyTracker(model_name="gpt-4o")

results = evaluate(
    ...,
    efficiency_tracker=tracker
)

# View summary
print(results.metadata["efficiency"])
# {
#   "total_cost_usd": 0.002,
#   "avg_latency_ms": 450.5,
#   "metrics_breakdown": { ... }
# }
```
