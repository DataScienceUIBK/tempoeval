"""Temporal Year Recall metric using Focus Time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

from tempoeval.core.base import BaseRetrievalMetric
from tempoeval.core.focus_time import FocusTime


@dataclass
class TemporalYearRecall(BaseRetrievalMetric):
    """
    Temporal Year Recall@K measures what fraction of query-relevant years
    are covered by the retrieved documents.
    
    Note: For Focus Time-based metrics, Year Precision and Year Recall 
    have the same formula when the denominator is |QFT|. The distinction
    becomes meaningful when considering weighted cases or different denominators.
    
    Formula:
        Year_Recall@K = |⋃(DFT_i for i=1..K) ∩ QFT| / |QFT|
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Does NOT require LLM (uses Focus Time sets)
    
    Example:
        >>> from tempoeval.metrics import TemporalYearRecall
        >>> metric = TemporalYearRecall()
        >>> qft = {2010, 2015, 2020}  # 3 years query asks about
        >>> dfts = [{2010, 2011}, {2015}]  # 2 docs
        >>> score = metric.compute(qft=qft, dfts=dfts, k=2)
        >>> print(score)  # 2/3 = 0.667 (2010 and 2015 covered)
    """
    
    name: str = field(init=False, default="temporal_year_recall")
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
        Compute Temporal Year Recall@K.
        
        Args:
            qft: Query Focus Time (set of years or FocusTime object)
            dfts: List of Document Focus Times for retrieved docs
            k: Cutoff for top-K documents
            
        Returns:
            Year recall score in [0, 1]
        """
        # Normalize inputs
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
        
        # Calculate recall: years covered / total required years
        covered_years = union_dfts & qft_years
        recall = len(covered_years) / len(qft_years)
        
        return round(recall, 5)
    
    async def acompute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        dfts: Optional[List[Union[Set[int], FocusTime]]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Async version (same as sync for this metric)."""
        return self.compute(qft=qft, dfts=dfts, k=k, **kwargs)
