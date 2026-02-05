"""Core infrastructure for TempoEval."""

from tempoeval.core.base import (
    BaseMetric,
    BaseRetrievalMetric,
    BaseGenerationMetric,
    BaseReasoningMetric,
    BaseCompositeMetric,
)
from tempoeval.core.evaluator import evaluate, evaluate_async, Evaluator
from tempoeval.core.result import EvaluationResult, QueryResult
from tempoeval.core.config import TempoEvalConfig
from tempoeval.core.focus_time import (
    FocusTime,
    FocusTimeExtractor,
    QueryFocusTime,
    DocumentFocusTime,
    AnswerFocusTime,
    WeightedFocusTime,
    ExtractionMethod,
    TemporalTagger,
    extract_qft,
    extract_dft,
    extract_aft,
    compute_temporal_relevance,
)

__all__ = [
    "BaseMetric",
    "BaseRetrievalMetric",
    "BaseGenerationMetric",
    "BaseReasoningMetric",
    "BaseCompositeMetric",
    "evaluate",
    "evaluate_async",
    "Evaluator",
    "EvaluationResult",
    "QueryResult",
    "TempoEvalConfig",
    # Focus Time
    "FocusTime",
    "FocusTimeExtractor",
    "QueryFocusTime",
    "DocumentFocusTime",
    "AnswerFocusTime",
    "WeightedFocusTime",
    "ExtractionMethod",
    "TemporalTagger",
    "extract_qft",
    "extract_dft",
    "extract_aft",
    "compute_temporal_relevance",
]

