"""Temporal Consistency metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import BaseReasoningMetric


@dataclass
class TemporalConsistency(BaseReasoningMetric):
    """
    Temporal Consistency measures if the system gives consistent answers
    about temporal information across multiple generations.
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Requires multiple answer samples
        - Can use LLM for comparison
    
    Example:
        >>> metric = TemporalConsistency()
        >>> score = metric.compute(
        ...     answers=[
        ...         "Python was created in 1991",
        ...         "Python was created in 1989",
        ...         "Python was created in 1991"
        ...     ]
        ... )
        >>> print(score)  # 0.67 (67% consistent)
    """
    
    name: str = field(init=False, default="temporal_consistency")
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        answers: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Compute Temporal Consistency from multiple answers.
        
        Args:
            answers: List of answer texts (sampled from same query)
            
        Returns:
            Consistency score in [0, 1]
        """
        if not answers or len(answers) < 2:
            return 0.0
        
        # Extract years from each answer
        import re
        year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')
        
        year_sets = []
        for answer in answers:
            years = set(year_pattern.findall(answer))
            year_sets.append(years)
        
        if not any(year_sets):
            return 0.0  # No temporal info to compare
        
        # Calculate pairwise consistency
        n = len(answers)
        consistent_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if year_sets[i] and year_sets[j]:
                    # Check if same years mentioned
                    if year_sets[i] == year_sets[j]:
                        consistent_pairs += 1
                    total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return round(consistent_pairs / total_pairs, 5)
    
    async def acompute(
        self,
        answers: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """Async compute (same as sync for this metric)."""
        return self.compute(answers=answers, **kwargs)
