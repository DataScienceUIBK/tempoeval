"""NDCG with Full Coverage - NDCG score conditioned on temporal coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class NDCGFullCoverage:
    """
    NDCG conditioned on full temporal coverage.
    
    This metric returns the NDCG score ONLY if temporal coverage is 1.0
    (both baseline and comparison periods are covered). Otherwise, it
    returns NaN.
    
    This is useful for cross-period queries where you want to ensure
    that good ranking (NDCG) is achieved ONLY when the retrieval set
    has complete temporal coverage. It penalizes systems that rank well
    but miss important time periods.
    
    Example:
        >>> from tempoeval.metrics import NDCGFullCoverage
        >>> metric = NDCGFullCoverage()
        >>> metric.llm = llm  # Set LLM for coverage check
        >>> score = metric.compute(
        ...     query="How has X changed since 2015?",
        ...     retrieved_ids=["doc1", "doc2", "doc3"],
        ...     gold_ids=["doc1", "doc3"],
        ...     retrieved_docs=[doc1_text, doc2_text, doc3_text],
        ...     baseline_anchor="2015",
        ...     comparison_anchor="2024",
        ...     k=10
        ... )
        >>> print(score)  # NDCG value if coverage=1, else NaN
    
    Attributes:
        name: Metric name for reporting
        llm: LLM provider for judging temporal coverage
    """
    
    name: str = field(init=False, default="ndcg_full_coverage")
    llm: Optional[object] = field(default=None)
    
    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        gold_ids: List[str],
        k: int
    ) -> float:
        """Calculate standard NDCG@k."""
        if k <= 0:
            return 0.0
        
        gold_set = set(str(x) for x in gold_ids)
        if not gold_set:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            rel = 1.0 if str(doc_id) in gold_set else 0.0
            dcg += rel / math.log2(i + 1)
        
        # Calculate IDCG
        ideal_hits = min(k, len(gold_set))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        
        return round(dcg / idcg, 5) if idcg > 0 else 0.0
    
    def _check_coverage(
        self,
        query: str,
        retrieved_docs: List[str],
        baseline_anchor: str,
        comparison_anchor: str,
        temporal_focus: str = "change_over_time"
    ) -> float:
        """Check temporal coverage using LLM."""
        if self.llm is None:
            return 0.0
        
        COVERAGE_PROMPT = """You are grading retrieved documents for a temporal trend/change query.

Query: "{query}"
Temporal Focus: {temporal_focus}

Baseline anchor period: {baseline_anchor}
Comparison/current anchor period: {comparison_anchor}

Document:
{document}

Decide:
1) covers_baseline: true if the document contains evidence about the BASELINE anchor period.
2) covers_comparison: true if the document contains evidence about the COMPARISON/current anchor period.

Return ONLY valid JSON:
{{"covers_baseline": true/false, "covers_comparison": true/false}}
"""
        
        baseline_covered = False
        comparison_covered = False
        
        for doc in retrieved_docs:
            doc_text = doc[:2000] if len(doc) > 2000 else doc
            try:
                prompt = COVERAGE_PROMPT.format(
                    query=query,
                    temporal_focus=temporal_focus,
                    baseline_anchor=baseline_anchor,
                    comparison_anchor=comparison_anchor,
                    document=doc_text
                )
                result = self.llm.generate_json(prompt)
                if result.get("covers_baseline"):
                    baseline_covered = True
                if result.get("covers_comparison"):
                    comparison_covered = True
                
                # Early exit if both covered
                if baseline_covered and comparison_covered:
                    break
            except Exception:
                continue
        
        return (int(baseline_covered) + int(comparison_covered)) / 2.0
    
    def compute(
        self,
        query: str,
        retrieved_ids: List[str],
        gold_ids: List[str],
        retrieved_docs: List[str],
        baseline_anchor: str,
        comparison_anchor: str,
        temporal_focus: str = "change_over_time",
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute NDCG conditioned on full coverage.
        
        Args:
            query: The search query
            retrieved_ids: List of retrieved document IDs
            gold_ids: List of gold document IDs
            retrieved_docs: List of retrieved document texts
            baseline_anchor: The baseline time period (e.g., "2015")
            comparison_anchor: The comparison time period (e.g., "2024")
            temporal_focus: Type of temporal query
            k: Number of top documents to consider
            
        Returns:
            NDCG score if coverage=1.0, else NaN
        """
        # Calculate coverage
        coverage = self._check_coverage(
            query=query,
            retrieved_docs=retrieved_docs[:k],
            baseline_anchor=baseline_anchor,
            comparison_anchor=comparison_anchor,
            temporal_focus=temporal_focus
        )
        
        # Only return NDCG if full coverage
        if coverage < 1.0:
            return 0.0
        
        return self._calculate_ndcg(retrieved_ids, gold_ids, k)
    
    async def acompute(
        self,
        query: str,
        retrieved_ids: List[str],
        gold_ids: List[str],
        retrieved_docs: List[str],
        baseline_anchor: str,
        comparison_anchor: str,
        temporal_focus: str = "change_over_time",
        k: int = 10,
        **kwargs
    ) -> float:
        """Async version of compute."""
        # For now, just call sync version
        # TODO: Implement proper async coverage check
        return self.compute(
            query=query,
            retrieved_ids=retrieved_ids,
            gold_ids=gold_ids,
            retrieved_docs=retrieved_docs,
            baseline_anchor=baseline_anchor,
            comparison_anchor=comparison_anchor,
            temporal_focus=temporal_focus,
            k=k,
            **kwargs
        )
