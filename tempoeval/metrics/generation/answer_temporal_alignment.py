"""Answer Temporal Alignment metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import BaseGenerationMetric


TEMPORAL_ALIGNMENT_PROMPT = """## Task
Evaluate how well the answer's temporal focus aligns with the query's temporal intent.

## Query
{query}

## Answer
{answer}

## Expected Time Focus
Target anchors: {target_anchors}
Temporal focus type: {temporal_focus}

## Evaluation Criteria
1. Does the answer address the specific time period(s) asked about?
2. Are the time references in the answer related to the query's temporal scope?
3. Does the answer avoid discussing irrelevant time periods?

## Output (JSON)
{{
    "query_time_focus": ["list of time periods the query asks about"],
    "answer_time_focus": ["list of time periods the answer discusses"],
    "overlap_analysis": {{
        "matching_periods": ["periods that align"],
        "missing_periods": ["query periods not addressed"],
        "extra_periods": ["answer periods not in query scope"]
    }},
    "alignment_score": 0.0,
    "reason": "explanation of alignment assessment"
}}
"""


@dataclass
class AnswerTemporalAlignment(BaseGenerationMetric):
    """
    Answer Temporal Alignment measures how well the answer's temporal
    focus matches the query's temporal intent.
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Checks if answer addresses the right time periods
        - Requires LLM
    
    Example:
        >>> metric = AnswerTemporalAlignment()
        >>> score = await metric.acompute(
        ...     query="What happened in 2020?",
        ...     answer="The company was founded in 2015 and grew steadily.",
        ...     temporal_focus="specific_time",
        ...     target_anchors=["2020"]
        ... )
        >>> print(score)  # 0.2 (answer doesn't focus on 2020)
    """
    
    name: str = field(init=False, default="answer_temporal_alignment")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        temporal_focus: str = "specific_time",
        target_anchors: Optional[List[str]] = None,
        key_time_anchors: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Compute Answer Temporal Alignment (synchronous).
        
        Args:
            query: The query text
            answer: Generated answer text
            temporal_focus: Type of temporal query
            target_anchors: Expected time anchors
            key_time_anchors: Alternative name for target_anchors
            
        Returns:
            Alignment score in [0, 1]
        """
        if target_anchors is None:
            target_anchors = key_time_anchors or []
        
        if not query or not answer:
            return float('nan')
        
        if self.llm is None:
            raise ValueError("AnswerTemporalAlignment requires an LLM provider")
        
        try:
            result = self.llm.generate_json(
                TEMPORAL_ALIGNMENT_PROMPT.format(
                    query=query,
                    answer=answer,
                    target_anchors=", ".join(target_anchors) if target_anchors else "Not specified",
                    temporal_focus=temporal_focus
                )
            )
            return result.get("alignment_score", 0.0)
        except Exception:
            return float('nan')
    
    async def acompute(
        self,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        temporal_focus: str = "specific_time",
        target_anchors: Optional[List[str]] = None,
        key_time_anchors: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """Async compute."""
        if target_anchors is None:
            target_anchors = key_time_anchors or []
        
        if not query or not answer:
            return float('nan')
        
        if self.llm is None:
            raise ValueError("AnswerTemporalAlignment requires an LLM provider")
        
        try:
            result = await self.llm.agenerate_json(
                TEMPORAL_ALIGNMENT_PROMPT.format(
                    query=query,
                    answer=answer,
                    target_anchors=", ".join(target_anchors) if target_anchors else "Not specified",
                    temporal_focus=temporal_focus
                )
            )
            return result.get("alignment_score", 0.0)
        except Exception:
            return float('nan')
