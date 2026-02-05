"""Focus Time extraction module for temporal evaluation.

This module implements the Focus Time concept where:
- Query Focus Time (QFT): Years relevant to the query
- Document Focus Time (DFT): Years a document is about (not publication date)
- Answer Focus Time (AFT): Years the generated answer discusses

Extraction Methods:
- REGEX: Fast regex-based extraction (default)
- HEIDELTIME: HeidelTime temporal tagger (more accurate, requires Java)
- LLM: LLM-based extraction (most flexible, requires API)

Based on the methodology from:
"Generic method for detecting focus time of documents" (IPM 2015)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from enum import Enum


class ExtractionMethod(Enum):
    """Temporal expression extraction method."""
    REGEX = "regex"                     # Fast regex + dictionary (default)
    TEMPORAL_TAGGER = "temporal_tagger" # External temporal tagger (see TemporalTagger)
    LLM = "llm"                         # LLM-based extraction


class TemporalTagger(Enum):
    """Temporal tagger library for use_temporal_tagger mode.
    
    Options:
    - DATEPARSER: Pure Python, fast, multi-language support (RECOMMENDED)
    - PARSEDATETIME: Pure Python, English natural language dates
    - FAST_PARSE_TIME: Ultra-fast, sub-millisecond performance
    - HEIDELTIME: Java-based HeidelTime (slow but accurate)
    """
    DATEPARSER = "dateparser"           # pip install dateparser (RECOMMENDED)
    PARSEDATETIME = "parsedatetime"     # pip install parsedatetime
    FAST_PARSE_TIME = "fast_parse_time" # pip install fast-parse-time
    HEIDELTIME = "heideltime"           # pip install py_heideltime (requires Java)


# Prompt for implicit year resolution
IMPLICIT_YEAR_RESOLUTION_PROMPT = """## Task
Extract the specific years this text is referring to.

## Text
{text}

## Instructions
1. Extract EXPLICIT years mentioned directly (e.g., "in 1994" → 1994)
2. Resolve IMPLICIT temporal references to specific years:
   - Historical events: "during WWII" → [1939, 1940, 1941, 1942, 1943, 1944, 1945]
   - Named periods: "Victorian era" → [1837-1901]
   - Relative terms: "recently" → last 3 years from current year
   - Decades: "the 1990s" → [1990, 1991, ..., 1999]

3. Only include years that are CENTRAL to the query/text meaning

## Output (JSON only)
{{
    "explicit_years": [1994, 2001],
    "implicit_years": [1939, 1940, 1941],
    "all_years": [1939, 1940, 1941, 1994, 2001],
    "reasoning": "brief explanation"
}}
"""


# Prompt for document focus time
DOCUMENT_FOCUS_TIME_PROMPT = """## Task
Determine the Focus Time of this document - the time period(s) the document is ABOUT.

## Document
{document}

## Publication Date (if known)
{publication_date}

## Instructions
Focus Time is DIFFERENT from publication date. A document published in 2024 about 
the 2008 financial crisis has Focus Time around 2008, not 2024.

Extract:
1. The main time period(s) the document discusses
2. Key event years mentioned
3. The temporal scope of the document's subject matter

## Output (JSON only)
{{
    "focus_years": [2008, 2009],
    "publication_year": 2024,
    "temporal_scope": "2008-2009",
    "reasoning": "Document discusses the 2008 financial crisis and its immediate aftermath"
}}
"""


@dataclass
class FocusTime:
    """Represents a Focus Time as a set of years with optional weights."""
    
    years: Set[int] = field(default_factory=set)
    weights: Optional[Dict[int, float]] = None  # For weighted/fuzzy cases
    
    def __and__(self, other: "FocusTime") -> Set[int]:
        """Intersection of two FocusTimes."""
        return self.years & other.years
    
    def __or__(self, other: "FocusTime") -> Set[int]:
        """Union of two FocusTimes."""
        return self.years | other.years
    
    def __len__(self) -> int:
        return len(self.years)
    
    def jaccard(self, other: "FocusTime") -> float:
        """Jaccard similarity between two FocusTimes."""
        intersection = len(self.years & other.years)
        union = len(self.years | other.years)
        return intersection / union if union > 0 else 0.0
    
    def overlap(self, other: "FocusTime") -> int:
        """Number of overlapping years."""
        return len(self.years & other.years)
    
    def covers(self, other: "FocusTime", threshold: float = 0.0) -> bool:
        """Check if this FocusTime covers the other (with optional threshold)."""
        if threshold == 0.0:
            return len(self.years & other.years) > 0
        return self.jaccard(other) >= threshold
    
    @classmethod
    def from_years(cls, years: Union[List[int], Set[int]]) -> "FocusTime":
        """Create FocusTime from a list or set of years."""
        return cls(years=set(years) if isinstance(years, list) else years)
    
    @classmethod
    def from_range(cls, start: int, end: int) -> "FocusTime":
        """Create FocusTime from a year range (inclusive)."""
        return cls(years=set(range(start, end + 1)))


@dataclass
class FocusTimeExtractor:
    """Base class for extracting Focus Time from text."""
    
    llm: Optional[object] = None
    current_year: int = field(default_factory=lambda: datetime.now().year)
    extraction_method: ExtractionMethod = ExtractionMethod.REGEX
    
    # Regex pattern for explicit years (1800-2099)
    YEAR_PATTERN = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')
    
    # Decade pattern
    DECADE_PATTERN = re.compile(r'\b(?:the\s+)?(1[89]\d0|20\d0)s\b', re.IGNORECASE)
    
    # HeidelTime instance (lazy loaded)
    _heideltime = None
    
    def extract_explicit_years(self, text: str) -> Set[int]:
        """Extract explicit years using regex."""
        years = set()
        
        # Extract individual years
        year_matches = self.YEAR_PATTERN.findall(text)
        years.update(int(y) for y in year_matches)
        
        # Extract decades and expand to all years
        decade_matches = self.DECADE_PATTERN.findall(text)
        for decade in decade_matches:
            decade_start = int(decade)
            years.update(range(decade_start, decade_start + 10))
        
        return years
    
    def _cleanup_heideltime_temp(self):
        """Clean up HeidelTime temporary files."""
        import shutil
        import glob
        import gc
        
        # Force garbage collection to close file handles
        gc.collect()
        
        try:
            import py_heideltime
            pkg_dir = os.path.dirname(py_heideltime.__file__)
            # Remove temp files created by py_heideltime
            for tmp_file in glob.glob(os.path.join(pkg_dir, 'tmp*')):
                try:
                    if os.path.isdir(tmp_file):
                        shutil.rmtree(tmp_file, ignore_errors=True)
                    else:
                        os.remove(tmp_file)
                except:
                    pass
        except:
            pass
    
    def extract_heideltime_years(self, text: str) -> Set[int]:
        """Extract years using HeidelTime temporal tagger.
        
        Requires: pip install py_heideltime
        Also requires: Java JDK installed
        """
        import gc
        
        # Clean up before starting (in case previous calls left files)
        self._cleanup_heideltime_temp()
        
        try:
            if self._heideltime is None:
                from py_heideltime import heideltime
                self._heideltime = heideltime
            
            # Get temporal annotations - returns a LIST of dicts
            result = self._heideltime(text, language='English', document_type='news')
            
            # Force garbage collection immediately after call
            gc.collect()
            
            # Parse result - py_heideltime returns list like:
            # [{'text': '1923', 'tid': 't1', 'type': 'DATE', 'value': '1923', 'span': [3, 7]}]
            years = set()
            
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and 'value' in item:
                        value = str(item['value'])
                        # Extract 4-digit year from value
                        year_match = re.match(r'^(\d{4})', value)
                        if year_match:
                            year = int(year_match.group(1))
                            if 1800 <= year <= 2099:
                                years.add(year)
            else:
                # Fallback: try to parse as string/XML (old format)
                year_pattern = re.compile(r'value="(\d{4})')
                matches = year_pattern.findall(str(result))
                for match in matches:
                    year = int(match)
                    if 1800 <= year <= 2099:
                        years.add(year)
            
            return years
        except ImportError:
            print("HeidelTime not installed. Install with: pip install py_heideltime")
            return set()
        except Exception as e:
            print(f"HeidelTime error: {e}")
            return set()
        finally:
            # Clean up after each call
            self._cleanup_heideltime_temp()
    
    def extract_temporal_tagger_years(
        self, 
        text: str, 
        tagger: TemporalTagger = TemporalTagger.DATEPARSER
    ) -> Set[int]:
        """Extract years using a temporal tagger library.
        
        Args:
            text: Text to extract years from
            tagger: Which tagger to use (default: DATEPARSER - fastest)
        
        Supported taggers:
            - DATEPARSER: Pure Python, fast, recommended (pip install dateparser)
            - PARSEDATETIME: Pure Python, English NL (pip install parsedatetime)
            - FAST_PARSE_TIME: Ultra-fast sub-ms (pip install fast-parse-time)
            - HEIDELTIME: Java-based, slow but accurate (pip install py_heideltime)
        """
        if tagger == TemporalTagger.DATEPARSER:
            return self._extract_dateparser(text)
        elif tagger == TemporalTagger.PARSEDATETIME:
            return self._extract_parsedatetime(text)
        elif tagger == TemporalTagger.FAST_PARSE_TIME:
            return self._extract_fast_parse_time(text)
        elif tagger == TemporalTagger.HEIDELTIME:
            return self.extract_heideltime_years(text)
        else:
            # Default to dateparser
            return self._extract_dateparser(text)
    
    def _extract_dateparser(self, text: str) -> Set[int]:
        """Extract years using dateparser library (FAST, RECOMMENDED)."""
        try:
            from dateparser.search import search_dates
            
            years = set()
            results = search_dates(text, settings={'STRICT_PARSING': False})
            
            if results:
                for text_match, dt in results:
                    if dt and hasattr(dt, 'year'):
                        year = dt.year
                        if 1800 <= year <= 2099:
                            years.add(year)
            
            return years
        except ImportError:
            print("dateparser not installed. Install with: pip install dateparser")
            return set()
        except Exception as e:
            print(f"dateparser error: {e}")
            return set()
    
    def _extract_parsedatetime(self, text: str) -> Set[int]:
        """Extract years using parsedatetime library."""
        try:
            import parsedatetime
            from datetime import datetime
            
            cal = parsedatetime.Calendar()
            years = set()
            
            # Parse the text
            time_struct, parse_status = cal.parse(text)
            if parse_status > 0:
                dt = datetime(*time_struct[:6])
                if 1800 <= dt.year <= 2099:
                    years.add(dt.year)
            
            # Also try to find all year mentions
            year_pattern = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')
            for match in year_pattern.findall(text):
                year = int(match)
                if 1800 <= year <= 2099:
                    years.add(year)
            
            return years
        except ImportError:
            print("parsedatetime not installed. Install with: pip install parsedatetime")
            return set()
        except Exception as e:
            print(f"parsedatetime error: {e}")
            return set()
    
    def _extract_fast_parse_time(self, text: str) -> Set[int]:
        """Extract years using fast-parse-time library (ULTRA-FAST)."""
        try:
            from fast_parse_time import parse_time
            
            years = set()
            result = parse_time(text)
            
            if result and hasattr(result, 'year'):
                year = result.year
                if 1800 <= year <= 2099:
                    years.add(year)
            
            # Also extract explicit years with regex
            year_pattern = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')
            for match in year_pattern.findall(text):
                year = int(match)
                if 1800 <= year <= 2099:
                    years.add(year)
            
            return years
        except ImportError:
            print("fast-parse-time not installed. Install with: pip install fast-parse-time")
            return set()
        except Exception as e:
            print(f"fast-parse-time error: {e}")
            return set()
    def extract_implicit_years(self, text: str) -> Set[int]:
        """Extract implicit years using LLM."""
        if self.llm is None:
            return set()
        
        try:
            result = self.llm.generate_json(
                IMPLICIT_YEAR_RESOLUTION_PROMPT.format(text=text[:2000])
            )
            all_years = result.get("all_years", [])
            return set(int(y) for y in all_years if isinstance(y, (int, float)))
        except Exception:
            return set()
    
    def extract(
        self, 
        text: str, 
        method: Optional[ExtractionMethod] = None,
        use_llm: bool = True
    ) -> FocusTime:
        """Extract Focus Time from text.
        
        Args:
            text: Text to extract years from
            method: Extraction method (REGEX, HEIDELTIME, or LLM). 
                    If None, uses self.extraction_method
            use_llm: Deprecated. Use method=ExtractionMethod.LLM instead
        """
        method = method or self.extraction_method
        
        if method == ExtractionMethod.HEIDELTIME:
            years = self.extract_heideltime_years(text)
        elif method == ExtractionMethod.LLM and self.llm:
            explicit = self.extract_explicit_years(text)
            implicit = self.extract_implicit_years(text)
            years = explicit | implicit
        else:
            # Default: REGEX
            years = self.extract_explicit_years(text)
        
        return FocusTime(years=years)


@dataclass
class QueryFocusTime(FocusTimeExtractor):
    """Extract Focus Time from a query."""
    
    # Common implicit references
    IMPLICIT_MAPPINGS = {
        "wwi": list(range(1914, 1919)),
        "world war i": list(range(1914, 1919)),
        "world war 1": list(range(1914, 1919)),
        "wwii": list(range(1939, 1946)),
        "world war ii": list(range(1939, 1946)),
        "world war 2": list(range(1939, 1946)),
        "cold war": list(range(1947, 1992)),
        "great depression": list(range(1929, 1940)),
        "victorian era": list(range(1837, 1902)),
        "covid": [2020, 2021, 2022, 2023],
        "pandemic": [2020, 2021, 2022],
    }
    
    # Relative time patterns
    RELATIVE_PATTERNS = {
        r'\brecently\b': lambda y: list(range(y - 2, y + 1)),
        r'\bcurrently\b': lambda y: [y],
        r'\btoday\b': lambda y: [y],
        r'\blast\s+year\b': lambda y: [y - 1],
        r'\bthis\s+year\b': lambda y: [y],
        r'\blast\s+decade\b': lambda y: list(range(y - 10, y)),
        r'\bpast\s+(\d+)\s+years?\b': lambda y, n: list(range(y - int(n), y + 1)),
    }
    
    def extract_with_rules(self, query: str) -> Set[int]:
        """Extract years using rule-based mappings."""
        years = set()
        query_lower = query.lower()
        
        # Check implicit mappings with word boundary matching
        # Sort by length descending to match longer patterns first (e.g., "world war ii" before "world war i")
        sorted_keys = sorted(self.IMPLICIT_MAPPINGS.keys(), key=len, reverse=True)
        matched_keys = set()
        
        for key in sorted_keys:
            # Use regex with word boundaries to avoid substring issues
            # e.g., "world war i" should NOT match "world war ii"
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, query_lower):
                # Check if this key is a substring of an already matched longer key
                already_covered = False
                for matched in matched_keys:
                    if key in matched:
                        already_covered = True
                        break
                
                if not already_covered:
                    years.update(self.IMPLICIT_MAPPINGS[key])
                    matched_keys.add(key)
        
        # Check relative patterns
        for pattern, func in self.RELATIVE_PATTERNS.items():
            match = re.search(pattern, query_lower)
            if match:
                if match.groups():
                    years.update(func(self.current_year, *match.groups()))
                else:
                    years.update(func(self.current_year))
        
        return years
    
    def extract(
        self, 
        query: str, 
        use_regex: bool = True,
        use_temporal_tagger: bool = False,
        use_llm: bool = False,
        tagger: TemporalTagger = TemporalTagger.DATEPARSER
    ) -> FocusTime:
        """Extract QFT from query.
        
        Args:
            query: The query text to extract years from
            use_regex: Use regex + rule-based extraction (default True)
            use_temporal_tagger: Use external temporal tagger (see tagger param)
            use_llm: Use LLM for implicit year extraction (requires llm client)
            tagger: Which temporal tagger to use (default: DATEPARSER - fastest)
        
        Multiple methods can be True - years will be merged into a unique set.
        
        Example:
            >>> extractor = QueryFocusTime(llm=llm)
            >>> # Regex only (default, fastest)
            >>> qft = extractor.extract(query)
            >>> # Dateparser temporal tagger
            >>> qft = extractor.extract(query, use_regex=False, use_temporal_tagger=True)
            >>> # HeidelTime (slow)
            >>> qft = extractor.extract(query, use_temporal_tagger=True, tagger=TemporalTagger.HEIDELTIME)
            >>> # All methods combined
            >>> qft = extractor.extract(query, use_regex=True, use_temporal_tagger=True, use_llm=True)
        """
        all_years = set()
        
        # REGEX: explicit years + rule-based mappings
        if use_regex:
            explicit = self.extract_explicit_years(query)
            rule_based = self.extract_with_rules(query)
            all_years |= explicit | rule_based
        
        # TEMPORAL TAGGER: dateparser, parsedatetime, fast-parse-time, or heideltime
        if use_temporal_tagger:
            tagger_years = self.extract_temporal_tagger_years(query, tagger=tagger)
            all_years |= tagger_years
        
        # LLM: implicit year extraction using GPT-4
        if use_llm and self.llm:
            llm_years = self.extract_implicit_years(query)
            all_years |= llm_years
        
        return FocusTime(years=all_years)


@dataclass
class DocumentFocusTime(FocusTimeExtractor):
    """Extract Focus Time from a document."""
    
    def extract(
        self, 
        content: str, 
        publication_date: Optional[str] = None,
        use_regex: bool = True,
        use_temporal_tagger: bool = False,
        use_llm: bool = False,
        tagger: TemporalTagger = TemporalTagger.DATEPARSER
    ) -> FocusTime:
        """
        Extract DFT from document content.
        
        Args:
            content: Document text
            publication_date: Optional publication date (YYYY or YYYY-MM-DD)
            use_regex: Use regex-based extraction (default True)
            use_temporal_tagger: Use external temporal tagger (see tagger param)
            use_llm: Use LLM for focus time detection
            tagger: Which temporal tagger to use (default: DATEPARSER - fastest)
        
        Multiple methods can be True - years will be merged into a unique set.
        
        Example:
            >>> extractor = DocumentFocusTime(llm=llm)
            >>> # Regex only (default, fastest)
            >>> dft = extractor.extract(document)
            >>> # Dateparser temporal tagger
            >>> dft = extractor.extract(document, use_regex=False, use_temporal_tagger=True)
            >>> # HeidelTime (slow)
            >>> dft = extractor.extract(document, use_temporal_tagger=True, tagger=TemporalTagger.HEIDELTIME)
        """
        all_years = set()
        
        # REGEX: explicit years
        if use_regex:
            years = self.extract_explicit_years(content)
            all_years |= years
        
        # TEMPORAL TAGGER: dateparser, parsedatetime, fast-parse-time, or heideltime
        if use_temporal_tagger:
            tagger_years = self.extract_temporal_tagger_years(content, tagger=tagger)
            all_years |= tagger_years
        
        # LLM: extract focus years using document analysis prompt
        if use_llm and self.llm:
            llm_years = self._extract_llm_years(content, publication_date)
            all_years |= llm_years
        
        return FocusTime(years=all_years)
    
    def _extract_llm_years(self, content: str, publication_date: Optional[str]) -> Set[int]:
        """LLM-based focus time detection - returns year set."""
        try:
            result = self.llm.generate_json(
                DOCUMENT_FOCUS_TIME_PROMPT.format(
                    document=content[:3000],
                    publication_date=publication_date or "Unknown"
                )
            )
            focus_years = result.get("focus_years", [])
            return set(int(y) for y in focus_years)
        except Exception:
            return set()


@dataclass
class AnswerFocusTime(FocusTimeExtractor):
    """Extract Focus Time from a generated answer."""
    
    def extract(
        self, 
        answer: str, 
        use_regex: bool = True,
        use_temporal_tagger: bool = False,
        use_llm: bool = False,
        tagger: TemporalTagger = TemporalTagger.DATEPARSER
    ) -> FocusTime:
        """Extract AFT from answer text.
        
        Args:
            answer: The generated answer text
            use_regex: Use regex-based extraction (default True)
            use_temporal_tagger: Use external temporal tagger (see tagger param)
            use_llm: Use LLM for implicit year extraction
            tagger: Which temporal tagger to use (default: DATEPARSER - fastest)
        
        Multiple methods can be True - years will be merged into a unique set.
        """
        all_years = set()
        
        # REGEX: explicit years
        if use_regex:
            explicit = self.extract_explicit_years(answer)
            all_years |= explicit
        
        # TEMPORAL TAGGER: dateparser, parsedatetime, fast-parse-time, or heideltime
        if use_temporal_tagger:
            tagger_years = self.extract_temporal_tagger_years(answer, tagger=tagger)
            all_years |= tagger_years
        
        # LLM: implicit year extraction
        if use_llm and self.llm:
            implicit = self.extract_implicit_years(answer)
            all_years |= implicit
        
        return FocusTime(years=all_years)


@dataclass
class WeightedFocusTime:
    """
    Fuzzy Focus Time with year weights for vague temporal expressions.
    
    Example: "recently" -> {2026: 1.0, 2025: 0.9, 2024: 0.8, 2023: 0.7, ...}
    """
    
    year_weights: Dict[int, float] = field(default_factory=dict)
    
    @property
    def years(self) -> Set[int]:
        """Get years with non-zero weights."""
        return {y for y, w in self.year_weights.items() if w > 0}
    
    def __len__(self) -> int:
        return len(self.years)
    
    @classmethod
    def from_recently(cls, base_year: int = 2026, decay: float = 0.1) -> "WeightedFocusTime":
        """Create weighted distribution for 'recently'."""
        weights = {}
        for i in range(10):
            year = base_year - i
            weights[year] = max(0, 1.0 - (i * decay))
        return cls(year_weights=weights)
    
    @classmethod
    def from_before(cls, year: int, decay: float = 0.1) -> "WeightedFocusTime":
        """Create weighted distribution for 'before X'."""
        weights = {}
        for i in range(20):
            y = year - 1 - i
            weights[y] = max(0, 1.0 - (i * decay))
        return cls(year_weights=weights)
    
    @classmethod
    def from_after(cls, year: int, base_year: int = 2026, decay: float = 0.1) -> "WeightedFocusTime":
        """Create weighted distribution for 'after X'."""
        weights = {}
        for i, y in enumerate(range(year + 1, base_year + 1)):
            weights[y] = max(0, 1.0 - (i * decay))
        return cls(year_weights=weights)
    
    @classmethod
    def from_around(cls, year: int, spread: int = 2, decay: float = 0.3) -> "WeightedFocusTime":
        """Create weighted distribution for 'around X'."""
        weights = {year: 1.0}
        for i in range(1, spread + 1):
            w = max(0, 1.0 - (i * decay))
            weights[year - i] = w
            weights[year + i] = w
        return cls(year_weights=weights)
    
    def weighted_overlap(self, other: "FocusTime") -> float:
        """Compute weighted overlap with another FocusTime."""
        total_weight = 0.0
        for year in other.years:
            total_weight += self.year_weights.get(year, 0.0)
        return total_weight


# Convenience functions
def extract_qft(query: str, llm: Optional[object] = None, use_llm: bool = True) -> FocusTime:
    """Extract Query Focus Time."""
    extractor = QueryFocusTime(llm=llm)
    return extractor.extract(query, use_llm=use_llm)


def extract_dft(
    document: str, 
    publication_date: Optional[str] = None,
    llm: Optional[object] = None, 
    use_llm: bool = True
) -> FocusTime:
    """Extract Document Focus Time."""
    extractor = DocumentFocusTime(llm=llm)
    return extractor.extract(document, publication_date, use_llm=use_llm)


def extract_aft(answer: str, llm: Optional[object] = None, use_llm: bool = True) -> FocusTime:
    """Extract Answer Focus Time."""
    extractor = AnswerFocusTime(llm=llm)
    return extractor.extract(answer, use_llm=use_llm)


def compute_temporal_relevance(
    qft: FocusTime, 
    dft: FocusTime, 
    mode: str = "any"
) -> Tuple[bool, float]:
    """
    Compute temporal relevance between QFT and DFT.
    
    Args:
        qft: Query Focus Time
        dft: Document Focus Time
        mode: Relevance mode
            - "any": At least one year overlap
            - "jaccard": Jaccard similarity
            - "all": All QFT years must be in DFT
    
    Returns:
        Tuple of (is_relevant: bool, score: float)
    """
    if not qft.years or not dft.years:
        return False, 0.0
    
    overlap = len(qft.years & dft.years)
    
    if mode == "any":
        return overlap > 0, float(overlap > 0)
    elif mode == "jaccard":
        score = qft.jaccard(dft)
        return score > 0, score
    elif mode == "all":
        is_all = qft.years <= dft.years
        return is_all, float(is_all)
    else:
        return overlap > 0, float(overlap > 0)
