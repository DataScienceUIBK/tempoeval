"""Cross-Period Reasoning metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tempoeval.core.base import BaseReasoningMetric


CROSS_PERIOD_REASONING_PROMPT = """## Task
Evaluate how well this answer synthesizes information across time periods.

## Query
{query}

## Answer
{answer}

## Time Periods to Cover
- Baseline: {baseline_anchor}
- Comparison: {comparison_anchor}

## Evaluation Criteria

1. **Period Coverage** (0-1): Does the answer discuss both periods?
2. **Explicit Comparison** (0-1): Is change/difference explicitly stated?
3. **Quantification** (0-1): Are changes quantified (numbers, percentages)?
4. **Causal Reasoning** (0-1): Are reasons for changes explained?
5. **Temporal Clarity** (0-1): Are time references unambiguous?

## Output (JSON)
{{
    "baseline_addressed": true or false,
    "comparison_addressed": true or false,
    "explicit_comparison_made": true or false,
    "quantified_changes": ["list of quantified changes"],
    "causal_explanations": ["list of causal explanations"],
    "criteria_scores": {{
        "period_coverage": 0.0,
        "explicit_comparison": 0.0,
        "quantification": 0.0,
        "causal_reasoning": 0.0,
        "temporal_clarity": 0.0
    }},
    "cross_period_score": 0.0
}}
"""


@dataclass
class CrossPeriodReasoning(BaseReasoningMetric):
    """
    Cross-Period Reasoning measures the quality of temporal synthesis
    when answering questions about changes/trends over time.
    
    Evaluates:
    - Does the answer address both time periods?
    - Is the comparison/contrast explicit?
    - Are changes quantified where appropriate?
    - Is causation/correlation properly attributed?
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Only applicable to trend/change queries
        - Requires LLM
    
    Example:
        >>> metric = CrossPeriodReasoning()
        >>> score = await metric.acompute(
        ...     query="How has renewable energy adoption changed from 2010 to 2020?",
        ...     answer="In 2010, renewables were 10% of energy. By 2020, this grew to 25%.",
        ...     baseline_anchor="2010",
        ...     comparison_anchor="2020"
        ... )
        >>> print(score)  # 0.9 (good cross-period synthesis)
    """
    
    name: str = field(init=False, default="cross_period_reasoning")
    requires_llm: bool = field(init=False, default=True)
    requires_gold: bool = field(init=False, default=False)
    
    def compute(
        self,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        baseline_anchor: str = "",
        comparison_anchor: str = "",
        **kwargs
    ) -> float:
        """
        Compute Cross-Period Reasoning (synchronous).
        
        Args:
            query: The query text
            answer: Generated answer text
            baseline_anchor: The baseline time period
            comparison_anchor: The comparison time period
            
        Returns:
            Cross-period reasoning score in [0, 1]
        """
        if not query or not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("CrossPeriodReasoning requires an LLM provider")
        
        try:
            result = self.llm.generate_json(
                CROSS_PERIOD_REASONING_PROMPT.format(
                    query=query,
                    answer=answer,
                    baseline_anchor=baseline_anchor or "earlier period",
                    comparison_anchor=comparison_anchor or "later period"
                )
            )
            return result.get("cross_period_score", 0.0)
        except Exception:
            return 0.0
    
    async def acompute(
        self,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        baseline_anchor: str = "",
        comparison_anchor: str = "",
        **kwargs
    ) -> float:
        """Async compute."""
        if not query or not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("CrossPeriodReasoning requires an LLM provider")
        
        try:
            result = await self.llm.agenerate_json(
                CROSS_PERIOD_REASONING_PROMPT.format(
                    query=query,
                    answer=answer,
                    baseline_anchor=baseline_anchor or "earlier period",
                    comparison_anchor=comparison_anchor or "later period"
                )
            )
            return result.get("cross_period_score", 0.0)
        except Exception:
            return 0.0
