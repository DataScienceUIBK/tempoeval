"""Temporal MAP (Mean Average Precision) using Focus Time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


@dataclass
class TemporalMAP(BaseRetrievalMetric):
    """
    Temporal Mean Average Precision using Focus Time-based relevance.
    
    A document is considered temporally relevant if its DFT overlaps with QFT.
    The overlap can be configured via threshold parameter.
    
    Formula:
        MAP = (1/R) × Σ(P@r × rel(r)) for r = 1..K
        where rel(r) = 1 if |QFT ∩ DFT_r| > 0, else 0
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Does NOT require LLM (uses Focus Time sets)
        - Rewards good ranking of temporally relevant documents
    
    Example:
        >>> from tempoeval.metrics import TemporalMAP
        >>> metric = TemporalMAP()
        >>> qft = {2020, 2021}
        >>> dfts = [{2020}, {2019}, {2021}, {2018}]  # Docs 0,2 are relevant
        >>> score = metric.compute(qft=qft, dfts=dfts, k=4)
        >>> # At pos 1: P=1/1=1, At pos 3: P=2/3=0.67
        >>> # MAP = (1 + 0.67) / 2 = 0.833
    """
    
    name: str = field(init=False, default="temporal_map")
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    
    # Relevance threshold (0 = any overlap, higher = stricter)
    relevance_threshold: float = 0.0
    use_jaccard: bool = False  # If True, use Jaccard similarity threshold
    
    def _is_relevant(self, qft_years: Set[int], dft_years: Set[int]) -> bool:
        """Check if document is temporally relevant."""
        if not qft_years or not dft_years:
            return False
        
        if self.use_jaccard:
            intersection = len(qft_years & dft_years)
            union = len(qft_years | dft_years)
            jaccard = intersection / union if union > 0 else 0.0
            return jaccard > self.relevance_threshold
        else:
            # Default: any overlap counts
            return len(qft_years & dft_years) > self.relevance_threshold
    
    def compute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal MAP@K.
        
        Args:
            qft: Query Focus Time (set of years or FocusTime object)
            dfts: List of Document Focus Times for retrieved docs
            k: Cutoff for top-K documents
            
        Returns:
            MAP score in [0, 1]
        """
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years:
            return 0.0
        
        if dfts is None or len(dfts) == 0:
            return 0.0
        
        # Compute relevance for each document
        relevances = []
        for dft in dfts[:k]:
            dft_years = dft.years if isinstance(dft, FocusTime) else set(dft)
            relevances.append(self._is_relevant(qft_years, dft_years))
        
        # Calculate MAP
        relevant_count = sum(relevances)
        if relevant_count == 0:
            return 0.0
        
        precision_sum = 0.0
        running_relevant = 0
        
        for i, is_rel in enumerate(relevances, start=1):
            if is_rel:
                running_relevant += 1
                precision_at_i = running_relevant / i
                precision_sum += precision_at_i
        
        map_score = precision_sum / relevant_count
        return round(map_score, 5)
    
    async def acompute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async version (same as sync for this metric)."""
        return self.compute(qft=qft, dfts=dfts, k=k, **kwargs)
