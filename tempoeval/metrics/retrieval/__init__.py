"""Retrieval metrics for TempoEval (Layer 1)."""

from tempoeval.metrics.retrieval.temporal_recall import TemporalRecall
from tempoeval.metrics.retrieval.temporal_precision import TemporalPrecision
from tempoeval.metrics.retrieval.temporal_ndcg import TemporalNDCG
from tempoeval.metrics.retrieval.temporal_mrr import TemporalMRR
from tempoeval.metrics.retrieval.temporal_coverage import TemporalCoverage
from tempoeval.metrics.retrieval.anchor_coverage import AnchorCoverage
from tempoeval.metrics.retrieval.temporal_diversity import TemporalDiversity
from tempoeval.metrics.retrieval.temporal_relevance import TemporalRelevance
from tempoeval.metrics.retrieval.ndcg_full_coverage import NDCGFullCoverage
from tempoeval.metrics.retrieval.temporal_persistence import TemporalPersistence
# Focus Time metrics
from tempoeval.metrics.retrieval.temporal_year_precision import TemporalYearPrecision
from tempoeval.metrics.retrieval.temporal_year_recall import TemporalYearRecall
from tempoeval.metrics.retrieval.temporal_map import TemporalMAP

ALL_RETRIEVAL_METRICS = [
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
    # Focus Time metrics
    TemporalYearPrecision,
    TemporalYearRecall,
    TemporalMAP,
]

__all__ = [
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
]
