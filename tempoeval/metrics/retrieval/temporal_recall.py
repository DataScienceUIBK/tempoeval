"""Temporal Recall metric with dual-mode (Gold/LLM/Focus Time)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


LLM_RELEVANCE_PROMPT = """## Task
Judge if this document is temporally relevant to answering the query.

## Query
{query}

## Document
{document}

## Criteria
A document is temporally relevant if it:
1. Contains temporal information (dates, periods, durations) needed to answer the query
2. Discusses events/facts from the time period the query asks about
3. Provides temporal context that helps answer the query

## Output (JSON)
{{
    "is_relevant": true or false,
    "relevance_score": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}
"""


@dataclass
class TemporalRecall(BaseRetrievalMetric):
    """
    Temporal Recall@K measures retrieval of temporally-relevant documents.
    
    Supports three modes:
    - Gold-based (default): Uses gold document IDs
    - LLM-as-Judge: LLM evaluates temporal relevance
    - Focus Time: Uses QFT/DFT year-set overlap
    
    Focus Time Formula:
        Recall@K = (# temporally relevant docs in top-K) / (total temporally relevant docs)
        where a doc is relevant if |QFT ∩ DFT| > 0
    
    Example (Focus Time mode):
        >>> metric = TemporalRecall(use_focus_time=True)
        >>> qft = {2020, 2021}
        >>> dfts = [{2020}, {2019}, {2021}, {2018}]  # Doc 0,2 relevant
        >>> all_relevant = 3  # Total relevant docs in collection
        >>> score = metric.compute(qft=qft, dfts=dfts, total_relevant=all_relevant, k=4)
    """
    
    name: str = field(init=False, default="temporal_recall")
    requires_gold: bool = field(init=False, default=False)
    use_llm: bool = False
    use_focus_time: bool = False
    relevance_threshold: float = 0.0
    
    @property
    def requires_llm(self) -> bool:
        """Dynamic LLM requirement."""
        return self.use_llm and not self.use_focus_time
    
    def compute(
        self,
        retrieved_ids: Optional[List[str]] = None,
        gold_ids: Optional[List[str]] = None,
        retrieved_docs: Optional[List[str]] = None,
        query: Optional[str] = None,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        total_relevant: Optional[int] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal Recall@K.
        
        Args:
            retrieved_ids: Retrieved document IDs (gold mode)
            gold_ids: Gold document IDs (gold mode)
            retrieved_docs: Document texts (LLM mode)
            query: Query text (LLM mode)
            qft: Query Focus Time (Focus Time mode)
            dfts: Document Focus Times (Focus Time mode)
            total_relevant: Total relevant docs in collection (Focus Time mode)
            k: Cutoff for top-K
        """
        if self.use_focus_time:
            return self._compute_focus_time(qft, dfts, total_relevant, k)
        elif self.use_llm:
            return self._compute_llm(query, retrieved_docs, k)
        else:
            return self._compute_gold(retrieved_ids, gold_ids, k)
    
    def _compute_focus_time(
        self,
        qft: Optional[Union[Set[int], FocusTime]],
        dfts: Optional[List[Union[Set[int], FocusTime]]],
        total_relevant: Optional[int],
        k: int
    ) -> float:
        """Focus Time-based recall."""
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years or dfts is None or len(dfts) == 0:
            return 0.0
        
        # Count relevant docs in top-K
        relevant_in_topk = 0
        for dft in dfts[:k]:
            dft_years = dft.years if isinstance(dft, FocusTime) else set(dft)
            if len(qft_years & dft_years) > self.relevance_threshold:
                relevant_in_topk += 1
        
        # If total_relevant not provided, use count from dfts
        if total_relevant is None:
            total_relevant = sum(
                1 for dft in dfts 
                if len(qft_years & (dft.years if isinstance(dft, FocusTime) else set(dft))) > 0
            )
        
        if total_relevant == 0:
            return 0.0
        
        return round(relevant_in_topk / total_relevant, 5)
    
    def _compute_gold(self, retrieved_ids, gold_ids, k):
        """Traditional gold-based recall."""
        if gold_ids is None or len(gold_ids) == 0:
            return 0.0
        
        if retrieved_ids is None or len(retrieved_ids) == 0:
            return 0.0
        
        retrieved_set = set(str(d) for d in retrieved_ids[:k])
        gold_set = set(str(d) for d in gold_ids)
        
        intersection = len(retrieved_set & gold_set)
        return round(intersection / len(gold_set), 5)
    
    def _compute_llm(self, query, retrieved_docs, k):
        """LLM-as-Judge recall."""
        if self.llm is None:
            raise ValueError("LLM mode requires an LLM provider")
        
        if not query or not retrieved_docs:
            return 0.0
        
        docs_to_eval = retrieved_docs[:k]
        relevant_count = 0.0
        
        for doc in docs_to_eval:
            try:
                result = self.llm.generate_json(
                    LLM_RELEVANCE_PROMPT.format(query=query, document=doc[:2000])
                )
                if result.get("is_relevant", False):
                    relevant_count += result.get("relevance_score", 1.0)
            except Exception:
                continue
        
        return round(relevant_count / len(docs_to_eval), 5)
    
    async def acompute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        total_relevant: Optional[int] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async compute."""
        if self.use_focus_time:
            return self._compute_focus_time(qft, dfts, total_relevant, k)
        return self.compute(**kwargs, k=k)

