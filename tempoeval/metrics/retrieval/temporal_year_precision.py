"""Temporal Year Precision metric using Focus Time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


@dataclass
class TemporalYearPrecision(BaseRetrievalMetric):
    """
    Temporal Year Precision@K measures the proportion of years in the 
    retrieved documents' focus time that are same as query-relevant years.
    
    Formula:
        Year_Precision@K = |⋃(DFT_i for i=1..K) ∩ QFT| / |⋃(DFT_i for i=1..K)|
    
    In other words: What fraction of document years are query-relevant?
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Does NOT require LLM (uses Focus Time sets)
    
    Example:
        >>> from tempoeval.metrics import TemporalYearPrecision
        >>> metric = TemporalYearPrecision()
        >>> qft = {2010, 2015, 2020}  # Query asks about 3 years
        >>> dfts = [{2010}, {2015}]   # Docs cover 2 years
        >>> score = metric.compute(qft=qft, dfts=dfts, k=2)
        >>> print(score)  # 2/2 = 1.0 (all doc years are query-relevant)
    """
    
    name: str = field(init=False, default="temporal_year_precision")
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Temporal Year Precision@K.
        
        Args:
            qft: Query Focus Time (set of years or FocusTime object)
            dfts: List of Document Focus Times for retrieved docs
            k: Cutoff for top-K documents
            
        Returns:
            Year precision score in [0, 1]
        """
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years:
            return 0.0
        
        if dfts is None or len(dfts) == 0:
            return 0.0
        
        # Get union of DFTs for top-K documents
        union_dfts = set()
        for dft in dfts[:k]:
            if isinstance(dft, FocusTime):
                union_dfts.update(dft.years)
            else:
                union_dfts.update(dft)
        
        if not union_dfts:
            return 0.0
        
        # Precision: doc years that are query-relevant / all doc years
        relevant_years = union_dfts & qft_years
        precision = len(relevant_years) / len(union_dfts)
        
        return round(precision, 5)
    
    async def acompute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async version (same as sync for this metric)."""
        return self.compute(qft=qft, dfts=dfts, k=k, **kwargs)

