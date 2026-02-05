"""Duration Accuracy metric with dual-mode (LLM or Rule-based)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from tempoeval.core.base import BaseReasoningMetric


DURATION_ACCURACY_PROMPT = """## Task
Evaluate the correctness of duration/time span claims in the answer.

## Answer
{answer}

## Gold Durations (if provided)
{gold_durations}

## Instructions
1. Extract all duration claims from the answer (e.g., "lasted 5 years", "for 3 months")
2. Compare with gold durations if provided
3. Check if durations are consistent with any dates mentioned

## Output (JSON)
{{
    "duration_claims": [
        {{
            "claim": "duration claim text",
            "extracted_duration": "5 years",
            "associated_event": "event this duration refers to",
            "gold_duration": "actual duration if known",
            "is_correct": true or false,
            "error_magnitude": 0
        }}
    ],
    "total_claims": 0,
    "correct_claims": 0,
    "accuracy_score": 0.0
}}
"""


@dataclass
class DurationAccuracy(BaseReasoningMetric):
    """
    Duration Accuracy measures the correctness of duration/time span
    claims in the answer.
    
    Supports two modes:
    - LLM-based (default): Uses LLM for complex duration verification
    - Rule-based: Uses regex to extract and verify simple durations (fast, free)
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - 1.0 = all durations correct
    
    Example:
        >>> metric = DurationAccuracy(use_llm=True)  # LLM-based (default)
        >>> metric.llm = llm_provider
        >>> score = metric.compute(...)
        
        >>> metric = DurationAccuracy(use_llm=False)  # Rule-based
        >>> score = metric.compute(answer="...", gold_durations={...})
    """
    
    name: str = field(init=False, default="duration_accuracy")
    requires_gold: bool = field(init=False, default=False)
    use_llm: bool = True  # Default to LLM, but can be disabled
    
    @property
    def requires_llm(self) -> bool:
        """Dynamic LLM requirement based on mode."""
        return self.use_llm
    
    def compute(
        self,
        answer: Optional[str] = None,
        gold_durations: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> float:
        """
        Compute Duration Accuracy.
        
        Args:
            answer: Generated answer text
            gold_durations: Dict of event -> duration (e.g., {"WWI": "4 years"})
            
        Returns:
            Accuracy score in [0, 1]
        """
        if not answer:
            return 0.0
        
        if self.use_llm:
            return self._compute_llm(answer, gold_durations)
        else:
            return self._compute_regex(answer, gold_durations)
    
    def _compute_llm(self, answer: str, gold_durations: Optional[Dict[str, str]]) -> float:
        """LLM-based duration verification."""
        if self.llm is None:
            raise ValueError("DurationAccuracy (LLM mode) requires an LLM provider")
        
        gold_str = ""
        if gold_durations:
            gold_str = "\n".join([f"- {k}: {v}" for k, v in gold_durations.items()])
        
        try:
            result = self.llm.generate_json(
                DURATION_ACCURACY_PROMPT.format(
                    answer=answer,
                    gold_durations=gold_str or "Not provided"
                )
            )
            return result.get("accuracy_score", 0.0)
        except Exception:
            return 0.0
    
    def _compute_regex(self, answer: str, gold_durations: Optional[Dict[str, str]]) -> float:
        """
        Rule-based duration verification using regex.
        
        Extracts durations like "X years", "Y months", "Z days" and compares to gold.
        """
        if not gold_durations:
            return 0.0
        
        # Extract durations from answer
        extracted = self._extract_durations_regex(answer)
        
        if not extracted:
            return 0.0
        
        # Compare with gold
        correct = 0
        total = len(gold_durations)
        
        # Normalize gold durations for comparison
        gold_normalized = {k.lower(): self._normalize_duration(v) for k, v in gold_durations.items()}
        
        for duration_text, duration_value in extracted:
            # Check if this matches any gold duration
            for gold_key, gold_value in gold_normalized.items():
                if gold_value and duration_value:
                    # Allow 10% tolerance
                    if abs(duration_value - gold_value) / max(gold_value, 1) < 0.1:
                        correct += 1
                        break
        
        return round(correct / total, 4) if total > 0 else 0.0
    
    def _extract_durations_regex(self, text: str) -> List[Tuple[str, Optional[float]]]:
        """Extract duration claims from text using regex."""
        durations = []
        
        # Pattern: X years/months/days/weeks
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(year|years)', 365),
            (r'(\d+(?:\.\d+)?)\s*(month|months)', 30),
            (r'(\d+(?:\.\d+)?)\s*(week|weeks)', 7),
            (r'(\d+(?:\.\d+)?)\s*(day|days)', 1),
            (r'(\d+(?:\.\d+)?)\s*(decade|decades)', 3650),
            (r'(\d+(?:\.\d+)?)\s*(century|centuries)', 36500),
        ]
        
        for pattern, multiplier in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match[0]) * multiplier
                    durations.append((f"{match[0]} {match[1]}", value))
                except ValueError:
                    continue
        
        return durations
    
    def _normalize_duration(self, duration_str: str) -> Optional[float]:
        """Normalize a duration string to days."""
        duration_str = duration_str.lower().strip()
        
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*year', 365),
            (r'(\d+(?:\.\d+)?)\s*month', 30),
            (r'(\d+(?:\.\d+)?)\s*week', 7),
            (r'(\d+(?:\.\d+)?)\s*day', 1),
            (r'(\d+(?:\.\d+)?)\s*decade', 3650),
            (r'(\d+(?:\.\d+)?)\s*century', 36500),
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, duration_str)
            if match:
                try:
                    return float(match.group(1)) * multiplier
                except ValueError:
                    continue
        
        return None
    
    async def acompute(
        self,
        answer: Optional[str] = None,
        gold_durations: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> float:
        """Async compute (delegates to sync for rule-based)."""
        if not self.use_llm:
            return self._compute_regex(answer, gold_durations)
        
        if not answer:
            return 0.0
        
        if self.llm is None:
            raise ValueError("DurationAccuracy (LLM mode) requires an LLM provider")
        
        gold_str = ""
        if gold_durations:
            gold_str = "\n".join([f"- {k}: {v}" for k, v in gold_durations.items()])
        
        try:
            result = await self.llm.agenerate_json(
                DURATION_ACCURACY_PROMPT.format(
                    answer=answer,
                    gold_durations=gold_str or "Not provided"
                )
            )
            return result.get("accuracy_score", 0.0)
        except Exception:
            return 0.0
