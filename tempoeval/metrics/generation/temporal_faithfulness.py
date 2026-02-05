"""Temporal Faithfulness metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import BaseGenerationMetric


TEMPORAL_FAITHFULNESS_PROMPT = """## Task
Identify all temporal claims in the ANSWER and verify each against the CONTEXT.

## Context (Retrieved Documents)
{context}

## Generated Answer
{answer}

## Instructions

### Step 1: Extract Temporal Claims
List every temporal claim from the answer:
- Explicit dates: "in 2017", "January 2020"
- Durations: "for 3 years", "lasted 6 months"
- Sequences: "before X", "after Y", "first...then"
- Relative times: "recently", "in the past decade"

### Step 2: Verify Each Claim
For each temporal claim, search the context for supporting evidence:
- SUPPORTED: Context explicitly confirms the claim
- PARTIALLY_SUPPORTED: Context suggests but doesn't confirm exactly
- NOT_SUPPORTED: No evidence in context
- CONTRADICTED: Context states something different

### Step 3: Calculate Score
Score = (SUPPORTED + 0.5 × PARTIALLY_SUPPORTED) / Total_Claims

## Output (JSON only)
{{
    "temporal_claims": [
        {{
            "claim": "the temporal claim text",
            "claim_type": "specific_date|duration|sequence|relative",
            "evidence_found": "quote or summary of evidence from context",
            "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|CONTRADICTED"
        }}
    ],
    "summary": {{
        "total_claims": 0,
        "supported": 0,
        "partially_supported": 0,
        "not_supported": 0,
        "contradicted": 0
    }},
    "faithfulness_score": 0.0
}}
"""


@dataclass
class TemporalFaithfulness(BaseGenerationMetric):
    """
    Temporal Faithfulness measures whether temporal claims in the
    generated answer are supported by the retrieved context.
    
    This is the temporal-specific version of RAGAS Faithfulness.
    
    A temporal claim is any statement involving:
    - Specific dates or years
    - Durations ("for 5 years")
    - Relative time ("before X", "after Y")
    - Time ranges ("from 2010 to 2020")
    
    Properties:
        - Range: [0, 1]
        - Higher is better (more claims supported)
        - 1.0 = all temporal claims are faithful to context
        - 0.0 = no temporal claims are supported
        - Requires LLM
    
    Example:
        >>> metric = TemporalFaithfulness()
        >>> score = await metric.acompute(
        ...     contexts=["Bitcoin pruning was introduced in v0.11 (2015)..."],
        ...     answer="Bitcoin introduced pruning in 2017 with version 0.14."
        ... )
        >>> print(score)  # 0.0 (date 2017 contradicts context)
    """
    
    name: str = field(init=False, default="temporal_faithfulness")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_docs: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Compute Temporal Faithfulness (synchronous).
        
        Args:
            answer: Generated answer text
            contexts: List of context/retrieved document texts
            retrieved_docs: Alternative name for contexts
            
        Returns:
            Faithfulness score in [0, 1]
        """
        # Handle alternative argument names
        if contexts is None:
            contexts = retrieved_docs
        
        if not answer or not contexts:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalFaithfulness requires an LLM provider")
        
        # Format context
        context_text = "\n\n---\n\n".join([
            f"[Document {i+1}]:\n{c}" 
            for i, c in enumerate(contexts)
        ])
        
        try:
            result = self.llm.generate_json(
                TEMPORAL_FAITHFULNESS_PROMPT.format(
                    context=context_text[:8000],  # Truncate
                    answer=answer
                )
            )
            return result.get("faithfulness_score", 0.0)
        except Exception:
            return 0.0
    
    async def acompute(
        self,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        retrieved_docs: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """Async compute."""
        if contexts is None:
            contexts = retrieved_docs
        
        if not answer or not contexts:
            return 0.0
        
        if self.llm is None:
            raise ValueError("TemporalFaithfulness requires an LLM provider")
        
        context_text = "\n\n---\n\n".join([
            f"[Document {i+1}]:\n{c}" 
            for i, c in enumerate(contexts)
        ])
        
        try:
            result = await self.llm.agenerate_json(
                TEMPORAL_FAITHFULNESS_PROMPT.format(
                    context=context_text[:8000],
                    answer=answer
                )
            )
            return result.get("faithfulness_score", 0.0)
        except Exception:
            return 0.0
