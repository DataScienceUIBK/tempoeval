"""Temporal Guidance Generation Module.

This module provides tools to generate temporal annotations for datasets
that don't have pre-existing guidance, using LLM-based analysis.

Classes:
    TemporalGuidanceGenerator: Main generator for query/passage annotations
    QueryGuidance: Dataclass for query-level temporal guidance
    PassageAnnotation: Dataclass for passage-level temporal annotations

Example:
    >>> from tempoeval.llm import AzureOpenAIProvider
    >>> from tempoeval.guidance import TemporalGuidanceGenerator
    >>> 
    >>> llm = AzureOpenAIProvider()
    >>> generator = TemporalGuidanceGenerator(llm)
    >>> 
    >>> # Generate query guidance
    >>> guidance = generator.generate_query_guidance("When did WWII end?")
    >>> print(guidance.temporal_intent)  # "when"
    >>> print(guidance.key_time_anchors)  # ["1945"]
"""

from tempoeval.guidance.generator import (
    TemporalGuidanceGenerator,
    QueryGuidance,
    PassageAnnotation,
)
from tempoeval.guidance.prompts import (
    QUERY_GUIDANCE_PROMPT,
    PASSAGE_ANNOTATION_PROMPT,
)

__all__ = [
    "TemporalGuidanceGenerator",
    "QueryGuidance",
    "PassageAnnotation",
    "QUERY_GUIDANCE_PROMPT",
    "PASSAGE_ANNOTATION_PROMPT",
]
