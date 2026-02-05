# Core API

The foundational classes and functions for TempoEval.

## FocusTime

The extracted temporal representation of a text.

```python
from tempoeval.core import FocusTime

@dataclass
class FocusTime:
    years: Set[int]
    # Optionally store the method used
    method: ExtractionMethod
```

---

## FocusTime Extractors

### QueryFocusTime
**Extract Query Focus Time**. Determines what time the user is asking about.

```python
from tempoeval.core.focus_time import QueryFocusTime, TemporalTagger

extractor = QueryFocusTime()

# Extract with REGEX only (default, fast)
qft = extractor.extract("What happened in 2020?", use_regex=True)

# Extract with Temporal Tagger (requires external library)
qft = extractor.extract(
    "What happened last decade?",
    use_temporal_tagger=True,
    tagger=TemporalTagger.DATEPARSER  # or PARSEDATETIME, FAST_PARSE_TIME, HEIDELTIME
)

# Combine multiple methods (years are merged)
qft = extractor.extract(
    query,
    use_regex=True,
    use_temporal_tagger=True,
    use_llm=True
)
```

### DocumentFocusTime
**Extract Document Focus Time**. Determines what time the document is about.

```python
from tempoeval.core.focus_time import DocumentFocusTime

extractor = DocumentFocusTime()
dft = extractor.extract("The 2008 financial crisis affected...")
```

### AnswerFocusTime
**Extract Answer Focus Time**. Determines what time the answer discusses.

```python
from tempoeval.core.focus_time import AnswerFocusTime

extractor = AnswerFocusTime()
aft = extractor.extract("The event occurred in 2015.")
```

---

## Extraction Methods

Enum defining available extraction strategies.

```python
from tempoeval.core import ExtractionMethod

class ExtractionMethod(str, Enum):
    REGEX = "regex"                    # Default, fast
    TEMPORAL_TAGGER = "temporal_tagger" # External temporal tagger
    LLM = "llm"                        # Requires LLM provider
```

---

## Temporal Taggers

Enum defining available temporal tagger implementations for the `use_temporal_tagger` option.

```python
from tempoeval.core import TemporalTagger

class TemporalTagger(str, Enum):
    DATEPARSER = "dateparser"         # ⭐ RECOMMENDED - Pure Python, fast
    PARSEDATETIME = "parsedatetime"   # Pure Python, English focused
    FAST_PARSE_TIME = "fast-parse-time" # Ultra-fast, sub-millisecond
    HEIDELTIME = "heideltime"         # Java-based, very slow
```

### Tagger Comparison

| Tagger | Speed | Install | Description |
|--------|-------|---------|-------------|
| **DATEPARSER** ⭐ | Fast | `pip install dateparser` | **RECOMMENDED** - Multi-language support |
| **PARSEDATETIME** | Fast | `pip install parsedatetime` | English natural language |
| **FAST_PARSE_TIME** | Ultra-fast | `pip install fast-parse-time` | Sub-millisecond |
| **HEIDELTIME** | Very slow | `pip install py_heideltime` + Java JDK | Most accurate |
