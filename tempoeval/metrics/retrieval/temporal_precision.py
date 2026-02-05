"""Temporal Precision metric with dual-mode (LLM or Focus Time)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


# LLM prompt for temporal relevance judgment
TEMPORAL_RELEVANCE_PROMPT = """Judge if a retrieved document helps answer the TEMPORAL aspects of a query.

Query: "{query}"
Temporal Focus: {temporal_focus}

Document:
{document}

Question: Does this document provide information that DIRECTLY helps answer the temporal aspects of the query?

Guidelines:
- Verdict = 1 if document contains temporal information (dates, durations, time periods, temporal sequences)
- Verdict = 0 if document lacks temporal information even if generally relevant
- For "when" queries: document must mention specific times/dates
- For "how long" queries: document must mention durations/time periods
- For "recent" queries: document must mention recency or recent dates
- Be STRICT: generic facts without temporal markers are NOT temporally relevant

Respond ONLY with valid JSON:
{{
    "temporal_expressions_found": ["list of temporal expressions found"],
    "relevance_to_query": "high|medium|low|none",
    "verdict": 1 or 0,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}
"""


@dataclass
class TemporalPrecision(BaseRetrievalMetric):
    """
    Temporal Precision@K measures the precision of temporally relevant documents.
    
    Supports two modes:
    - LLM mode (default): Uses LLM-as-judge for relevance (position-weighted)
    - Focus Time mode: Uses QFT/DFT year-set overlap for relevance
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Position-weighted in LLM mode
    
    Focus Time Formula:
        Precision@K = (# docs where |QFT ∩ DFT| > 0) / K
    
    Example (Focus Time mode):
        >>> metric = TemporalPrecision(use_focus_time=True)
        >>> qft = {2020, 2021}
        >>> dfts = [{2020}, {2019}, {2021}]  # Doc 0,2 relevant
        >>> score = metric.compute(qft=qft, dfts=dfts, k=3)
        >>> print(score)  # 2/3 = 0.667
    """
    
    name: str = field(init=False, default="temporal_precision")
    requires_gold: bool = field(init=False, default=False)
    
    # Mode selection
    use_focus_time: bool = False  # If True, use Focus Time mode
    relevance_threshold: float = 0.0  # Jaccard threshold (0 = any overlap)
    
    @property
    def requires_llm(self) -> bool:
        """LLM required only in LLM mode."""
        return not self.use_focus_time
    
    def compute(
        self,
        query: Optional[str] = None,
        retrieved_docs: Optional[List[str]] = None,
        temporal_focus: str = "specific_time",
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal Precision@K.
        
        Args:
            query: The query text (LLM mode)
            retrieved_docs: List of retrieved document texts (LLM mode)
            temporal_focus: Type of temporal query (LLM mode)
            qft: Query Focus Time (Focus Time mode)
            dfts: List of Document Focus Times (Focus Time mode)
            k: Cutoff for top-K
            
        Returns:
            Precision score in [0, 1]
        """
        if self.use_focus_time:
            return self._compute_focus_time(qft, dfts, k)
        else:
            return self._compute_llm(query, retrieved_docs, temporal_focus, k)
    
    def _compute_focus_time(
        self,
        qft: Optional[Union[Set[int], FocusTime]],
        dfts: Optional[List[Union[Set[int], FocusTime]]],
        k: int
    ) -> float:
        """Focus Time-based precision: # relevant docs / K."""
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years or dfts is None or len(dfts) == 0:
            return 0.0
        
        relevant_count = 0
        for dft in dfts[:k]:
            dft_years = dft.years if isinstance(dft, FocusTime) else set(dft)
            
            if self.relevance_threshold > 0:
                # Jaccard-based relevance
                intersection = len(qft_years & dft_years)
                union = len(qft_years | dft_years)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard > self.relevance_threshold:
                    relevant_count += 1
            else:
                # Any overlap counts
                if len(qft_years & dft_years) > 0:
                    relevant_count += 1
        
        precision = relevant_count / min(k, len(dfts))
        return round(precision, 5)
    
    def _compute_llm(
        self,
        query: Optional[str],
        retrieved_docs: Optional[List[str]],
        temporal_focus: str,
        k: int
    ) -> float:
        """LLM-based precision with position weighting."""
        if not query or not retrieved_docs:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalPrecision requires an LLM provider in LLM mode")
        
        verdicts = []
        for doc in retrieved_docs[:k]:
            try:
                judgment = self.llm.generate_json(
                    TEMPORAL_RELEVANCE_PROMPT.format(
                        query=query,
                        temporal_focus=temporal_focus,
                        document=doc[:2000]
                    )
                )
                verdicts.append(judgment.get("verdict", 0))
            except Exception:
                verdicts.append(0)
        
        return self._position_weighted_precision(verdicts)
    
    async def acompute(
        self,
        query: Optional[str] = None,
        retrieved_docs: Optional[List[str]] = None,
        temporal_focus: str = "specific_time",
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async compute."""
        if self.use_focus_time:
            return self._compute_focus_time(qft, dfts, k)
        
        import asyncio
        
        if not query or not retrieved_docs:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalPrecision requires an LLM provider")
        
        async def judge_doc(doc: str) -> int:
            try:
                judgment = await self.llm.agenerate_json(
                    TEMPORAL_RELEVANCE_PROMPT.format(
                        query=query,
                        temporal_focus=temporal_focus,
                        document=doc[:2000]
                    )
                )
                return judgment.get("verdict", 0)
            except Exception:
                return 0
        
        verdicts = await asyncio.gather(*[judge_doc(doc) for doc in retrieved_docs[:k]])
        return self._position_weighted_precision(list(verdicts))
    
    def _position_weighted_precision(self, verdicts: List[int]) -> float:
        """Compute position-weighted precision from verdicts."""
        relevant_positions = [i + 1 for i, v in enumerate(verdicts) if v == 1]
        
        if not relevant_positions:
            return 0.0
        
        weighted_sum = 0.0
        for rank in relevant_positions:
            relevant_up_to = sum(1 for r in relevant_positions if r <= rank)
            weighted_sum += (relevant_up_to / rank)
        
        return round(weighted_sum / len(relevant_positions), 5)

