"""Answer Temporal Recall metric using Focus Time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Union

from tempoeval.core.base import BaseGenerationMetric
from tempoeval.core.focus_time import FocusTime


@dataclass
class AnswerTemporalRecall(BaseGenerationMetric):
    """
    Answer Temporal Recall measures what fraction of query-relevant years
    are addressed in the generated answer.
    
    Formula:
        Answer_Recall = |AFT ∩ QFT| / |QFT|
    
    This complements Answer Temporal Alignment (which measures Precision)
    by measuring Recall - ensuring the answer covers all required time periods.
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Does NOT require LLM (uses Focus Time sets)
    
    Example:
        >>> from tempoeval.metrics import AnswerTemporalRecall
        >>> metric = AnswerTemporalRecall()
        >>> qft = {2010, 2015, 2020}  # Query asks about 3 years
        >>> aft = {2010, 2015, 2018}  # Answer mentions these years
        >>> score = metric.compute(qft=qft, aft=aft)
        >>> print(score)  # 2/3 = 0.667 (2010, 2015 covered; 2020 missing)
    """
    
    name: str = field(init=False, default="answer_temporal_recall")
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        aft: Optional[Union[Set[int], FocusTime]] = None,
        **kwargs
    ) -> float:
        """
        Compute Answer Temporal Recall.
        
        Args:
            qft: Query Focus Time (set of years or FocusTime object)
            aft: Answer Focus Time (set of years or FocusTime object)
            
        Returns:
            Recall score in [0, 1]
        """
        if qft is None:
            return 0.0
        
        qft_years = qft.years if isinstance(qft, FocusTime) else set(qft)
        
        if not qft_years:
            return 0.0
        
        if aft is None:
            return 0.0
        
        aft_years = aft.years if isinstance(aft, FocusTime) else set(aft)
        
        if not aft_years:
            return 0.0
        
        # Calculate recall: years covered / total required years
        covered_years = aft_years & qft_years
        recall = len(covered_years) / len(qft_years)
        
        return round(recall, 5)
    
    async def acompute(
        self,
        qft: Optional[Union[Set[int], FocusTime]] = None,
        aft: Optional[Union[Set[int], FocusTime]] = None,
        **kwargs
    ) -> float:
        """Async version (same as sync for this metric)."""
        return self.compute(qft=qft, aft=aft, **kwargs)
