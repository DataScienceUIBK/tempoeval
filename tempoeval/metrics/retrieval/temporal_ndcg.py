"""Temporal NDCG metric with Gold/LLM/Focus Time modes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Set

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


LLM_GRADED_RELEVANCE_PROMPT = """## Task
Rate the temporal relevance of this document to the query on a scale of 0-4.

## Query
{query}

## Document
{document}

## Grading Scale (G-EVAL style)
- 4: Perfect - Contains exact temporal information needed to fully answer the query
- 3: Highly Relevant - Contains most temporal information, minor gaps
- 2: Partially Relevant - Contains some temporal context but incomplete
- 1: Marginally Relevant - Mentions related time periods but doesn't directly answer
- 0: Not Relevant - No useful temporal information for this query

## Output (JSON)
{{
    "relevance_score": 0 to 4,
    "reasoning": "brief explanation of temporal relevance"
}}
"""


@dataclass
class TemporalNDCG(BaseRetrievalMetric):
    """
    Temporal NDCG@K measures ranking quality with graded relevance.
    
    Supports three modes:
    - Gold-based (default): Uses gold document IDs/scores
    - LLM-as-Judge: LLM assigns graded relevance (0-4)
    - Focus Time: Jaccard(QFT, DFT) for graded relevance
    
    Focus Time Formula:
        relevance = Jaccard(QFT, DFT) × 4  (scaled to 0-4)
        NDCG = DCG / IDCG
    
    Example (Focus Time mode):
        >>> metric = TemporalNDCG(use_focus_time=True)
        >>> qft = {2020, 2021}
        >>> dfts = [{2020, 2021}, {2019}, {2020}]
        >>> score = metric.compute(qft=qft, dfts=dfts, k=3)
    """
    
    name: str = field(init=False, default="temporal_ndcg")
    requires_gold: bool = field(init=False, default=False)
    use_llm: bool = False
    use_focus_time: bool = False
    
    @property
    def requires_llm(self) -> bool:
        return self.use_llm and not self.use_focus_time
    
    @property
    def supports_graded_relevance(self) -> bool:
        return True

    def compute(
        self,
        retrieved_ids: Optional[List[str]] = None,
        gold_ids: Optional[Union[List[str], Dict[str, float]]] = None,
        retrieved_docs: Optional[List[str]] = None,
        query: Optional[str] = None,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Compute Temporal NDCG@K."""
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
        """Focus Time-based NDCG using Jaccard similarity."""
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years or dfts is None or len(dfts) == 0:
            return 0.0
        
        # Calculate Jaccard-based relevance scores (scaled to 0-4)
        relevance_scores = []
        for dft in dfts[:k]:
            dft_years = dft.years if isinstance(dft, FocusTime) else set(dft)
            if not dft_years:
                relevance_scores.append(0.0)
            else:
                intersection = len(qft_years & dft_years)
                union = len(qft_years | dft_years)
                jaccard = intersection / union if union > 0 else 0.0
                relevance_scores.append(jaccard * 4)  # Scale to 0-4
        
        return self._calculate_ndcg(relevance_scores)
    
    def _compute_gold(self, retrieved_ids, gold_ids, k):
        """Traditional gold-based NDCG."""
        if k <= 0 or retrieved_ids is None or len(retrieved_ids) == 0:
            return 0.0
            
        if gold_ids is None:
            return 0.0
            
        if isinstance(gold_ids, list):
            gold_scores = {str(x): 1.0 for x in gold_ids}
        elif isinstance(gold_ids, dict):
            gold_scores = {str(k): float(v) for k, v in gold_ids.items()}
        else:
            return 0.0
            
        if len(gold_scores) == 0:
            return 0.0
        
        relevance_scores = [gold_scores.get(str(doc_id), 0.0) for doc_id in retrieved_ids[:k]]
        
        dcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(relevance_scores, start=1))
        
        ideal_relevances = sorted(gold_scores.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_relevances, start=1))
        
        if idcg == 0:
            return 0.0
        
        return round(dcg / idcg, 5)
    
    def _compute_llm(self, query, retrieved_docs, k):
        """LLM-as-Judge NDCG (G-EVAL style)."""
        if self.llm is None:
            raise ValueError("LLM mode requires an LLM provider")
        
        if not query or not retrieved_docs:
            return 0.0
        
        relevance_scores = []
        for doc in retrieved_docs[:k]:
            try:
                result = self.llm.generate_json(
                    LLM_GRADED_RELEVANCE_PROMPT.format(query=query, document=doc[:2000])
                )
                score = min(4, max(0, result.get("relevance_score", 0)))
                relevance_scores.append(float(score))
            except Exception:
                relevance_scores.append(0.0)
        
        return self._calculate_ndcg(relevance_scores)
    
    def _calculate_ndcg(self, relevance_scores: List[float]) -> float:
        """Calculate NDCG from relevance scores."""
        if not relevance_scores:
            return 0.0
        
        dcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(relevance_scores, start=1))
        
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_scores, start=1))
        
        if idcg == 0:
            return 0.0
        
        return round(dcg / idcg, 5)
    
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

