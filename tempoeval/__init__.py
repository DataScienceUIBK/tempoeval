"""
TempoEval: Comprehensive Evaluation Framework for Temporal IR, QA, and RAG Systems.

A library for evaluating temporal aspects of information retrieval, question answering,
and retrieval-augmented generation systems with 16+ temporal-specific metrics.
"""

from tempoeval.version import __version__
from tempoeval.core.evaluator import evaluate, evaluate_async
from tempoeval.core.result import EvaluationResult
from tempoeval.core.config import TempoEvalConfig

__all__ = [
    "__version__",
    "evaluate",
    "evaluate_async",
    "EvaluationResult",
    "TempoEvalConfig",
]
