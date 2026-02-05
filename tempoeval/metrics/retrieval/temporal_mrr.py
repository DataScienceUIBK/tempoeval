"""Temporal MRR metric with Gold/LLM/Focus Time modes."""

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
    "confidence": 0.0 to 1.0
}}
"""


@dataclass
class TemporalMRR(BaseRetrievalMetric):
    """
    Temporal MRR measures how quickly we find the first temporally-relevant document.
    
    Supports three modes:
    - Gold-based (default): Uses gold document IDs
    - LLM-as-Judge: LLM evaluates relevance
    - Focus Time: Relevance if |QFT ∩ DFT| > 0
    
    Focus Time Formula:
        MRR = 1 / rank_of_first_doc_where |QFT ∩ DFT| > 0
    
    Example (Focus Time mode):
        >>> metric = TemporalMRR(use_focus_time=True)
        >>> qft = {2020, 2021}
        >>> dfts = [{2019}, {2020}, {2018}]  # Doc 1 is relevant
        >>> score = metric.compute(qft=qft, dfts=dfts, k=3)
        >>> print(score)  # 1/2 = 0.5
    """
    
    name: str = field(init=False, default="temporal_mrr")
    requires_gold: bool = field(init=False, default=False)
    use_llm: bool = False
    use_focus_time: bool = False
    relevance_threshold: float = 0.0
    
    @property
    def requires_llm(self) -> bool:
        return self.use_llm and not self.use_focus_time
    
    def compute(
        self,
        retrieved_ids: Optional[List[str]] = None,
        gold_ids: Optional[List[str]] = None,
        retrieved_docs: Optional[List[str]] = None,
        query: Optional[str] = None,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Compute Temporal MRR."""
        if self.use_focus_time:
            return self._compute_focus_time(qft, dfts, k)
        elif self.use_llm:
            return self._compute_llm(query, retrieved_docs, k)
        else:
            return self._compute_gold(retrieved_ids, gold_ids, k)
    
    def _compute_focus_time(
        self,
        qft: Optional[Union[Set[int], FocusTime]],
        dfts: Optional[List[Union[Set[int], FocusTime]]],
        k: int
    ) -> float:
        """Focus Time-based MRR."""
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years or dfts is None or len(dfts) == 0:
            return 0.0
        
        for i, dft in enumerate(dfts[:k], start=1):
            dft_years = dft.years if isinstance(dft, FocusTime) else set(dft)
            if len(qft_years & dft_years) > self.relevance_threshold:
                return round(1.0 / i, 5)
        
        return 0.0
    
    def _compute_gold(self, retrieved_ids, gold_ids, k):
        """Traditional gold-based MRR."""
        if gold_ids is None or len(gold_ids) == 0:
            return 0.0
        
        if retrieved_ids is None or len(retrieved_ids) == 0:
            return 0.0
        
        gold_set = set(str(x) for x in gold_ids)
        
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            if str(doc_id) in gold_set:
                return round(1.0 / i, 5)
        
        return 0.0
    
    def _compute_llm(self, query, retrieved_docs, k):
        """LLM-as-Judge MRR."""
        if self.llm is None:
            raise ValueError("LLM mode requires an LLM provider")
        
        if not query or not retrieved_docs:
            return 0.0
        
        for i, doc in enumerate(retrieved_docs[:k], start=1):
            try:
                result = self.llm.generate_json(
                    LLM_RELEVANCE_PROMPT.format(query=query, document=doc[:2000])
                )
                if result.get("is_relevant", False):
                    return round(1.0 / i, 5)
            except Exception:
                continue
        
        return 0.0
    
    async def acompute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async compute."""
        if self.use_focus_time:
            return self._compute_focus_time(qft, dfts, k)
        return self.compute(**kwargs, k=k)

