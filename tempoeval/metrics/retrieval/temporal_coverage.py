"""Temporal Coverage metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import BaseRetrievalMetric


# LLM prompt for temporal coverage judgment
TEMPORAL_EVIDENCE_PROMPT = """You are grading retrieved documents for a temporal trend/change query that needs cross-period evidence.

Query: "{query}"
Temporal Focus: {temporal_focus}

Baseline anchor period: {baseline_anchor}
Comparison/current anchor period: {comparison_anchor}

Document:
{document}

Decide:
1) verdict: 1 if the document DIRECTLY helps answer the temporal aspects of the query (contains relevant temporal info), else 0.
2) covers_baseline: true if the document contains evidence about the BASELINE anchor period.
3) covers_comparison: true if the document contains evidence about the COMPARISON/current anchor period.

Strictness rules:
- A random date not related to the anchors does NOT count.
- "Currently/as of 2023/recent years" can count for comparison coverage if relevant.
- Baseline coverage should connect to the baseline anchor period (e.g., around 2017).

Return ONLY valid JSON:
{{
  "verdict": 1 or 0,
  "reason": "brief explanation",
  "temporal_contribution": "what temporal information provided, or 'none'",
  "covers_baseline": true or false,
  "covers_comparison": true or false
}}
"""


@dataclass
class TemporalCoverage(BaseRetrievalMetric):
    """
    Temporal Coverage measures whether retrieved documents cover
    the required time periods for cross-period/trend queries.
    
    For a query like "How has X changed since 2017?", we need documents
    covering both 2017 (baseline) and present (comparison).
    
    Properties:
        - Range: [0, 1]
        - 0 = neither period covered
        - 0.5 = one period covered
        - 1.0 = both periods covered
        - Only applicable to cross-period queries
        - Requires LLM
    
    Formula:
        Temporal_Coverage@K = (baseline_covered + comparison_covered) / 2
    
    Example:
        >>> metric = TemporalCoverage()
        >>> score = await metric.acompute(
        ...     query="How has Bitcoin's energy consumption changed since 2017?",
        ...     retrieved_docs=docs,
        ...     temporal_focus="change_over_time",
        ...     baseline_anchor="2017",
        ...     comparison_anchor="2024"
        ... )
    """
    
    name: str = field(init=False, default="temporal_coverage")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        query: Optional[str] = None,
        retrieved_docs: Optional[List[str]] = None,
        temporal_focus: str = "change_over_time",
        baseline_anchor: str = "",
        comparison_anchor: str = "",
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal Coverage@K (synchronous).
        
        Args:
            query: The query text
            retrieved_docs: List of retrieved document texts
            temporal_focus: Type of temporal query
            baseline_anchor: The baseline time period
            comparison_anchor: The comparison time period
            k: Cutoff for top-K
            
        Returns:
            Coverage score in [0, 1]
        """
        # Skip if not a cross-period query
        if temporal_focus not in ["change_over_time", "trends_changes", "trends_changes_and_cross_period"]:
            return 0.0
        
        if not query or not retrieved_docs:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalCoverage requires an LLM provider")
        
        baseline_covered = False
        comparison_covered = False
        
        for doc in retrieved_docs[:k]:
            try:
                judgment = self.llm.generate_json(
                    TEMPORAL_EVIDENCE_PROMPT.format(
                        query=query,
                        temporal_focus=temporal_focus,
                        baseline_anchor=baseline_anchor,
                        comparison_anchor=comparison_anchor,
                        document=doc[:2000]
                    )
                )
                
                if judgment.get("covers_baseline"):
                    baseline_covered = True
                if judgment.get("covers_comparison"):
                    comparison_covered = True
                
                # Early exit if both covered
                if baseline_covered and comparison_covered:
                    break
                    
            except Exception:
                continue
        
        coverage = (int(baseline_covered) + int(comparison_covered)) / 2
        return round(coverage, 5)
    
    async def acompute(
        self,
        query: Optional[str] = None,
        retrieved_docs: Optional[List[str]] = None,
        temporal_focus: str = "change_over_time",
        baseline_anchor: str = "",
        comparison_anchor: str = "",
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal Coverage@K (asynchronous).
        """
        import asyncio
        
        if temporal_focus not in ["change_over_time", "trends_changes", "trends_changes_and_cross_period"]:
            return 0.0
        
        if not query or not retrieved_docs:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalCoverage requires an LLM provider")
        
        async def judge_doc(doc: str):
            try:
                return await self.llm.agenerate_json(
                    TEMPORAL_EVIDENCE_PROMPT.format(
                        query=query,
                        temporal_focus=temporal_focus,
                        baseline_anchor=baseline_anchor,
                        comparison_anchor=comparison_anchor,
                        document=doc[:2000]
                    )
                )
            except Exception:
                return {}
        
        judgments = await asyncio.gather(*[judge_doc(doc) for doc in retrieved_docs[:k]])
        
        baseline_covered = any(j.get("covers_baseline") for j in judgments)
        comparison_covered = any(j.get("covers_comparison") for j in judgments)
        
        coverage = (int(baseline_covered) + int(comparison_covered)) / 2
        return round(coverage, 5)
