"""Temporal Persistence metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from tempoeval.core.base import BaseRetrievalMetric


@dataclass
class TemporalPersistence(BaseRetrievalMetric):
    """
    Temporal Persistence measures the stability of retrieval performance 
    across different time periods (e.g., training era vs. future era).
    
    This addresses "Temporal Misalignment" where models degradation over time.
    Inspired by LongEval (CLEF 2024).
    
    It requires pre-computed metric scores for different time splits.
    
    Properties:
        - Range: [-inf, 1.0] (usually [0, 1])
        - Higher is better (1.0 = no degradation or improvement)
        - 1.0 means performance on target split >= baseline split
        
    Formula:
        Persistence = Score(Target_Split) / Score(Baseline_Split)
        (Capped at 1.0 usually, or raw ratio)
        
    Example:
        >>> metric = TemporalPersistence()
        >>> score = metric.compute(
        ...     baseline_score=0.8,  # Score on 2010-2015 data
        ...     target_score=0.6     # Score on 2016-2020 data
        ... )
        >>> # Result: 0.75 (25% drop)
    """
    
    name: str = field(init=False, default="temporal_persistence")
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        baseline_score: float = None,
        target_score: float = None,
        **kwargs
    ) -> float:
        """
        Compute Temporal Persistence.
        
        Args:
            baseline_score: Performance on the baseline period (e.g., training era)
            target_score: Performance on the target period (e.g., future era)
            
        Returns:
            Ratio target/baseline. Returns 1.0 if baseline is 0 (edge case).
        """
        if baseline_score is None or target_score is None:
            return 0.0
            
        if baseline_score == 0:
            if target_score == 0:
                return 1.0
            else:
                return 1.0 # Infinite improvement? Capped at 1.0 for stability.
        
        ratio = target_score / baseline_score
        return round(ratio, 4)
