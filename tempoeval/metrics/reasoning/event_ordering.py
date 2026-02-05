"""Event Ordering metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import BaseReasoningMetric


EVENT_ORDERING_PROMPT = """## Task
Extract the temporal sequence of events from this answer and verify ordering.

## Answer
{answer}

## Gold Event Order (if provided)
{gold_order}

## Instructions
1. Extract all events mentioned in chronological order as stated in the answer
2. If gold order is provided, compare with it
3. If no gold order, assess internal consistency of the ordering

## Output (JSON)
{{
    "extracted_events": [
        {{"event": "event description", "stated_order": 1, "temporal_marker": "first/then/after/etc"}}
    ],
    "extracted_order": ["Event A", "Event B", "Event C"],
    "gold_order": ["Event A", "Event B", "Event C"],
    "order_matches": true or false,
    "ordering_score": 0.0,
    "ordering_errors": [
        {{"expected": "A before B", "actual": "B before A"}}
    ]
}}
"""


@dataclass
class EventOrdering(BaseReasoningMetric):
    """
    Event Ordering Accuracy measures whether the temporal sequence
    of events in the answer matches the correct order.
    
    This is crucial for queries like:
    - "What happened before/after X?"
    - "Describe the timeline of..."
    - "What was the sequence of events?"
    
    Uses Kendall Tau distance to measure ordering accuracy.
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - 1.0 = perfect ordering
        - 0.0 = completely reversed
        - Requires LLM
    
    Example:
        >>> metric = EventOrdering()
        >>> score = await metric.acompute(
        ...     answer="First came A, then B, finally C.",
        ...     gold_order=["A", "B", "C"]
        ... )
        >>> print(score)  # 1.0 (perfect ordering)
    """
    
    name: str = field(init=False, default="event_ordering")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        answer: Optional[str] = None,
        gold_order: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Compute Event Ordering Accuracy (synchronous).
        
        Args:
            answer: Generated answer text
            gold_order: Gold-standard event sequence
            
        Returns:
            Ordering accuracy in [0, 1]
        """
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("EventOrdering requires an LLM provider")
        
        try:
            result = self.llm.generate_json(
                EVENT_ORDERING_PROMPT.format(
                    answer=answer,
                    gold_order=", ".join(gold_order) if gold_order else "Not provided"
                )
            )
            return result.get("ordering_score", 0.0)
        except Exception:
            return 0.0
    
    async def acompute(
        self,
        answer: Optional[str] = None,
        gold_order: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """Async compute."""
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("EventOrdering requires an LLM provider")
        
        try:
            result = await self.llm.agenerate_json(
                EVENT_ORDERING_PROMPT.format(
                    answer=answer,
                    gold_order=", ".join(gold_order) if gold_order else "Not provided"
                )
            )
            return result.get("ordering_score", 0.0)
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_kendall_tau(predicted: List[str], gold: List[str]) -> float:
        """
        Compute normalized Kendall Tau distance.
        
        Args:
            predicted: Predicted event order
            gold: Gold-standard event order
            
        Returns:
            Score in [0, 1] where 1 is perfect match
        """
        if not predicted or not gold:
            return 0.0
        
        # Create position maps
        gold_positions = {event: i for i, event in enumerate(gold)}
        
        # Count inversions
        inversions = 0
        n = len(predicted)
        
        for i in range(n):
            for j in range(i + 1, n):
                event_i, event_j = predicted[i], predicted[j]
                if event_i in gold_positions and event_j in gold_positions:
                    if gold_positions[event_i] > gold_positions[event_j]:
                        inversions += 1
        
        # Normalize
        max_inversions = n * (n - 1) / 2
        return 1 - (inversions / max_inversions) if max_inversions > 0 else 1.0
