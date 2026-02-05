"""Temporal Coherence metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tempoeval.core.base import BaseGenerationMetric


TEMPORAL_COHERENCE_PROMPT = """## Task
Critically evaluate the temporal coherence of this answer.

## Answer
{answer}

## Step 1: Reconstruct Timeline
First, extract every event and its associated time. List them chronologically.
If the timeline is impossible to construct (jumps around, contradictory), note it.

## Step 2: Check for Issues
Look for:
1. **Contradictions**: Is the same event given two different dates?
2. **Causality Violations**: Does an effect happen before its cause?
3. **Anachronisms**: Are technologies/concepts mentioned before they existed?

## Scoring Guide
- 1.0: Perfect timeline, clear causal flow.
- 0.7: Minor ambiguities but generally consistent.
- 0.4: Confusing timeline or minor contradictions.
- 0.0: Impossible sequence (e.g., born after death) or major contradictions.

## Output (JSON)
{{
    "timeline_reconstruction": [
        {{"time": "1990", "event": "Event A"}},
        {{"time": "1995", "event": "Event B"}}
    ],
    "coherence_issues": [
        {{
            "issue": "Event B (1995) causes Event A (1990)",
            "severity": "critical"
        }}
    ],
    "coherence_score": 0.0
}}
"""


@dataclass
class TemporalCoherence(BaseGenerationMetric):
    """
    Temporal Coherence measures the internal temporal consistency
    of the generated answer.
    
    Checks for:
    - Contradictory dates (event dated as both 2019 and 2020)
    - Impossible sequences (end before start)
    - Timeline inconsistencies
    - Duration mismatches
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Does not require context (intrinsic property of answer)
        - Requires LLM
    
    Example:
        >>> metric = TemporalCoherence()
        >>> score = await metric.acompute(
        ...     answer="The project started in 2020 and was completed in 2018."
        ... )
        >>> print(score)  # 0.0 (end date before start date)
    """
    
    name: str = field(init=False, default="temporal_coherence")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        answer: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Compute Temporal Coherence (synchronous).
        
        Args:
            answer: Generated answer text
            
        Returns:
            Coherence score in [0, 1]
        """
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalCoherence requires an LLM provider")
        
        try:
            result = self.llm.generate_json(
                TEMPORAL_COHERENCE_PROMPT.format(answer=answer)
            )
            return result.get("coherence_score", 0.0)
        except Exception:
            return 0.0
    
    async def acompute(
        self,
        answer: Optional[str] = None,
        **kwargs
    ) -> float:
        """Async compute."""
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalCoherence requires an LLM provider")
        
        try:
            result = await self.llm.agenerate_json(
                TEMPORAL_COHERENCE_PROMPT.format(answer=answer)
            )
            return result.get("coherence_score", 0.0)
        except Exception:
            return 0.0
