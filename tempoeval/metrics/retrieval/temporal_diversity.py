"""Temporal Diversity metric with Rule-based/LLM/Focus Time modes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


PERIOD_EXTRACTION_PROMPT = """## Task
Identify distinct HIGHLIGHTS of the timeline in these documents.

## Documents
{documents}

## Instructions
Extract distinct historical eras or time clusters.
Cluster close dates together (e.g., "2018, 2019" -> "Late 2010s").
Do not list every single year. We want to see DIVERSITY of coverage.

Examples of distinct periods:
- "The Depression Era" vs "Post-War Boom" (Distinct)
- "2018" vs "2019" (Not distinct, same era)

## Output (JSON)
{{
    "periods": [
        {{"name": "Late 2010s", "representative_year": 2018}},
        {{"name": "Early 20th Century", "representative_year": 1910}}
    ],
    "unique_years": [2018, 1910]
}}
"""


@dataclass
class TemporalDiversity(BaseRetrievalMetric):
    """
    Temporal Diversity measures coverage of different time periods.
    
    Supports three modes:
    - Rule-based (default): Uses regex to extract years
    - LLM-based: Uses LLM to extract semantic periods
    - Focus Time: Counts unique years from DFTs
    
    Focus Time Formula:
        Diversity@K = |⋃(DFT_i for i=1..K)| / |T|
        where T is the total temporal scope
    
    Example (Focus Time mode):
        >>> metric = TemporalDiversity(use_focus_time=True)
        >>> dfts = [{2020}, {2019}, {2018, 2020}]
        >>> score = metric.compute(dfts=dfts, total_scope=10, k=3)
        >>> print(score)  # 3/10 = 0.3
    """
    
    name: str = field(init=False, default="temporal_diversity")
    requires_gold: bool = field(init=False, default=False)
    use_llm: bool = False
    use_focus_time: bool = False
    
    @property
    def requires_llm(self) -> bool:
        return self.use_llm and not self.use_focus_time
    
    def compute(
        self,
        retrieved_docs: Optional[List[str]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        total_scope: Optional[int] = None,
        temporal_scope_start: Optional[str] = None,
        temporal_scope_end: Optional[str] = None,
        expected_periods: Optional[int] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal Diversity.
        
        Args:
            retrieved_docs: Document texts (Rule/LLM mode)
            dfts: Document Focus Times (Focus Time mode)
            total_scope: Total years in collection (Focus Time mode)
            temporal_scope_start: Start year (Rule/LLM mode)
            temporal_scope_end: End year (Rule/LLM mode)
            expected_periods: Expected periods (Rule/LLM mode)
            k: Cutoff for top-K
        """
        if self.use_focus_time:
            return self._compute_focus_time(dfts, total_scope, k)
        
        if not retrieved_docs:
            return 0.0
        
        if self.use_llm:
            years_found = self._extract_years_llm(retrieved_docs[:k])
        else:
            years_found = self._extract_years_regex(retrieved_docs[:k])
        
        if not years_found:
            return 0.0
        
        # Determine expected periods
        if expected_periods is not None:
            n_expected = expected_periods
        elif temporal_scope_start and temporal_scope_end:
            try:
                start = int(temporal_scope_start)
                end = int(temporal_scope_end)
                n_expected = max(1, end - start + 1)
            except ValueError:
                n_expected = 10
        else:
            n_expected = 10
        
        n_unique = len(years_found)
        diversity = min(1.0, n_unique / n_expected)
        
        return round(diversity, 5)
    
    def _compute_focus_time(
        self,
        dfts: Optional[List[Union[Set[int], FocusTime]]],
        total_scope: Optional[int],
        k: int
    ) -> float:
        """Focus Time-based diversity: unique years covered / total scope."""
        if dfts is None or len(dfts) == 0:
            return 0.0
        
        union_years = set()
        for dft in dfts[:k]:
            if isinstance(dft, FocusTime):
                union_years.update(dft.years)
            else:
                union_years.update(dft)
        
        if not union_years:
            return 0.0
        
        if total_scope is None or total_scope <= 0:
            total_scope = len(union_years)  # Use actual coverage as baseline
            if total_scope == 0:
                return 0.0
            return 1.0  # Perfect coverage of what's available
        
        diversity = len(union_years) / total_scope
        return round(min(1.0, diversity), 5)

    def _extract_years_regex(self, docs: List[str]) -> Set[int]:
        """Extract decade clusters using regex."""
        clusters: Set[int] = set()
        decade_pattern = re.compile(r'\b(19|20)(\d)\d\b')
        
        for doc in docs:
            matches = decade_pattern.findall(doc)
            for century, decade in matches:
                try:
                    cluster_year = int(century + decade + "0")
                    clusters.add(cluster_year)
                except ValueError:
                    continue
        
        return clusters

    def _extract_years_llm(self, docs: List[str]) -> Set[int]:
        """Extract periods using LLM."""
        if self.llm is None:
            raise ValueError("LLM mode requires an LLM provider")
        
        docs_text = "\n\n---\n\n".join([f"[Doc {i+1}]: {d[:2000]}" for i, d in enumerate(docs)])
        
        try:
            result = self.llm.generate_json(
                PERIOD_EXTRACTION_PROMPT.format(documents=docs_text)
            )
            unique_years = result.get("unique_years", [])
            buckets = set()
            for y in unique_years:
                if isinstance(y, (int, float)):
                    buckets.add(int(y) // 5 * 5)
            return buckets
        except Exception:
            return self._extract_years_regex(docs)
    
    async def acompute(
        self,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        total_scope: Optional[int] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async compute."""
        if self.use_focus_time:
            return self._compute_focus_time(dfts, total_scope, k)
        return self.compute(**kwargs, k=k)

