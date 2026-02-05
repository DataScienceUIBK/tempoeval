"""Temporal Hallucination metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import BaseGenerationMetric


TEMPORAL_HALLUCINATION_PROMPT = """## Task
Identify temporal hallucinations in the generated answer.

A temporal hallucination is a temporal claim that is:
1. NOT supported by the provided context, AND
2. Likely factually incorrect or fabricated

## Context
{context}

## Generated Answer
{answer}

## Gold Answer (if available)
{gold_answer}

## Instructions
For each temporal claim in the answer:
1. Check if it appears in the context
2. If not in context, assess if it's plausibly correct or fabricated
3. Cross-reference with gold answer if available

## Output (JSON)
{{
    "temporal_claims": [
        {{
            "claim": "the temporal claim",
            "in_context": true or false,
            "likely_fabricated": true or false,
            "contradicts_gold": true or false,
            "is_hallucination": true or false,
            "reason": "explanation"
        }}
    ],
    "total_claims": 0,
    "hallucinated_claims": ["list of hallucinated claims"],
    "hallucination_rate": 0.0
}}
"""


@dataclass
class TemporalHallucination(BaseGenerationMetric):
    """
    Temporal Hallucination Rate measures the proportion of temporal
    claims in the answer that are fabricated/unsupported.
    
    Unlike Temporal Faithfulness which checks against context,
    this also considers factual correctness against gold answers
    when available.
    
    Hallucination types detected:
    - Fabricated dates (dates not in context or reality)
    - Wrong durations (incorrect time spans)
    - Incorrect sequences (wrong temporal ordering)
    - Non-existent events (events that didn't happen)
    
    Properties:
        - Range: [0, 1]
        - LOWER is better (fewer hallucinations)
        - 0.0 = no hallucinations
        - 1.0 = all temporal claims are hallucinated
        - Requires LLM
    
    Example:
        >>> metric = TemporalHallucination()
        >>> score = await metric.acompute(
        ...     contexts=contexts,
        ...     answer=answer,
        ...     gold_answer=gold_answer
        ... )
        >>> print(score)  # 0.2 means 20% of claims are hallucinated
    """
    
    name: str = field(init=False, default="temporal_hallucination")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    lower_is_better: bool = field(init=False, default=True)
    
    def compute(
        self,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        gold_answer: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Compute Temporal Hallucination Rate (synchronous).
        
        Args:
            answer: Generated answer text
            contexts: List of context/retrieved document texts
            gold_answer: Gold-standard answer (optional)
            
        Returns:
            Hallucination rate in [0, 1] (lower is better)
        """
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalHallucination requires an LLM provider")
        
        context_text = ""
        if contexts:
            context_text = "\n\n---\n\n".join([
                f"[Document {i+1}]:\n{c}" 
                for i, c in enumerate(contexts)
            ])
        
        try:
            result = self.llm.generate_json(
                TEMPORAL_HALLUCINATION_PROMPT.format(
                    context=context_text[:8000] if context_text else "No context provided",
                    answer=answer,
                    gold_answer=gold_answer or "Not provided"
                )
            )
            return result.get("hallucination_rate", 0.0)
        except Exception:
            return 0.0
    
    async def acompute(
        self,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        gold_answer: Optional[str] = None,
        **kwargs
    ) -> float:
        """Async compute."""
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalHallucination requires an LLM provider")
        
        context_text = ""
        if contexts:
            context_text = "\n\n---\n\n".join([
                f"[Document {i+1}]:\n{c}" 
                for i, c in enumerate(contexts)
            ])
        
        try:
            result = await self.llm.agenerate_json(
                TEMPORAL_HALLUCINATION_PROMPT.format(
                    context=context_text[:8000] if context_text else "No context provided",
                    answer=answer,
                    gold_answer=gold_answer or "Not provided"
                )
            )
            return result.get("hallucination_rate", 0.0)
        except Exception:
            return 0.0
