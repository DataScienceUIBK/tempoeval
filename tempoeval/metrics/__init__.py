"""All metrics for TempoEval."""

# Retrieval metrics (Layer 1)
from tempoeval.metrics.retrieval import (
    TemporalRecall,
    TemporalPrecision, 
    TemporalNDCG,
    TemporalMRR,
    TemporalCoverage,
    AnchorCoverage,
    TemporalDiversity,
    TemporalRelevance,
    NDCGFullCoverage,
    TemporalPersistence,
    TemporalYearPrecision,
    TemporalYearRecall,
    TemporalMAP,
    ALL_RETRIEVAL_METRICS,
)

# Generation metrics (Layer 2)
from tempoeval.metrics.generation import (
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    AnswerTemporalAlignment,
    ALL_GENERATION_METRICS,
)

# Reasoning metrics (Layer 3)
from tempoeval.metrics.reasoning import (
    EventOrdering,
    DurationAccuracy,
    CrossPeriodReasoning,
    TemporalConsistency,
    ALL_REASONING_METRICS,
)

# Composite metrics
from tempoeval.metrics.composite import TempoScore

# Convenience collections
ALL_METRICS = [
    # Retrieval
    TemporalRecall,
    TemporalPrecision,
    TemporalNDCG,
    TemporalMRR,
    TemporalCoverage,
    AnchorCoverage,
    TemporalDiversity,
    TemporalRelevance,
    NDCGFullCoverage,
    TemporalPersistence,
    TemporalYearPrecision,
    TemporalYearRecall,
    TemporalMAP,
    # Generation
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    AnswerTemporalAlignment,
    # Reasoning
    EventOrdering,
    DurationAccuracy,
    CrossPeriodReasoning,
    TemporalConsistency,
    # Composite
    TempoScore,
]

# Metrics that don't require LLM
NO_LLM_METRICS = [
    TemporalRecall,
    TemporalNDCG,
    TemporalMRR,
    AnchorCoverage,
    TemporalDiversity,
    TemporalConsistency,
    TempoScore,
    TemporalPersistence,
    TemporalYearPrecision,
    TemporalYearRecall,
    TemporalMAP,
]

# Metrics that require gold labels
GOLD_REQUIRED_METRICS = [
    TemporalRecall,
    TemporalNDCG,
    TemporalMRR,
    AnchorCoverage,
]

__all__ = [
    # Retrieval
    "TemporalRecall",
    "TemporalPrecision",
    "TemporalNDCG",
    "TemporalMRR",
    "TemporalCoverage",
    "AnchorCoverage",
    "TemporalDiversity",
    "TemporalRelevance",
    "NDCGFullCoverage",
    "TemporalPersistence",
    "TemporalYearPrecision",
    "TemporalYearRecall",
    "TemporalMAP",
    "ALL_RETRIEVAL_METRICS",
    # Generation
    "TemporalFaithfulness",
    "TemporalHallucination",
    "TemporalCoherence",
    "AnswerTemporalAlignment",
    "ALL_GENERATION_METRICS",
    # Reasoning
    "EventOrdering",
    "DurationAccuracy",
    "CrossPeriodReasoning",
    "TemporalConsistency",
    "ALL_REASONING_METRICS",
    # Composite
    "TempoScore",
    # Collections
    "ALL_METRICS",
    "NO_LLM_METRICS",
    "GOLD_REQUIRED_METRICS",
]
