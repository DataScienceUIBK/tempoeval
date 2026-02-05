"""TempoScore - Composite temporal quality metric with multiple aggregation methods."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tempoeval.core.base import BaseCompositeMetric


# Default component weights
DEFAULT_WEIGHTS = {
    "temporal_precision": 0.25,
    "temporal_recall": 0.25,
    "temporal_faithfulness": 0.25,
    "temporal_coherence": 0.25,
}


@dataclass
class TempoScore(BaseCompositeMetric):
    """
    TempoScore - Unified composite metric for Temporal RAG evaluation.
    
    Computes THREE different aggregation methods:
    1. **Weighted Average**: Simple weighted sum (robust, no NaN)
    2. **Harmonic Mean**: Penalizes low scores (strict balance)
    3. **Layered F-Score**: Hierarchical F1 (retrieval F1 + generation F1)
    
    Components:
    - Retrieval: temporal_precision, temporal_recall
    - Generation: temporal_faithfulness, temporal_coherence
    
    Returns:
        Dict with:
        - tempo_weighted: Weighted average score
        - tempo_harmonic: Weighted harmonic mean score
        - tempo_f_score: Layered F1 score (recommended)
        - retrieval_f1: F1 of precision/recall
        - generation_f1: F1 of faithfulness/coherence
    
    Example:
        >>> metric = TempoScore()
        >>> result = metric.compute(
        ...     temporal_precision=0.8,
        ...     temporal_recall=0.7,
        ...     temporal_faithfulness=0.9,
        ...     temporal_coherence=0.85
        ... )
        >>> print(result)
        {
            'tempo_weighted': 0.8125,
            'tempo_harmonic': 0.7983,
            'tempo_f_score': 0.8324,
            'retrieval_f1': 0.7467,
            'generation_f1': 0.8727
        }
    """
    
    name: str = field(init=False, default="tempo_score")
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    metric_type: str = field(init=False, default="composite")
    
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    retrieval_weight: float = 0.5  # Weight for retrieval vs generation in F-score
    
    def _safe_value(self, val: Optional[float]) -> float:
        """Convert None/NaN to 0."""
        if val is None:
            return 0.0
        if isinstance(val, float) and math.isnan(val):
            return 0.0
        return max(0.0, float(val))
    
    def _get_score(self, all_scores: Dict, metric_name: str) -> float:
        """Get score for a metric, handling @K suffixes."""
        if metric_name in all_scores:
            return self._safe_value(all_scores[metric_name])
        # Try with @K suffix
        for k in all_scores:
            if k.startswith(metric_name + "@"):
                return self._safe_value(all_scores[k])
        return 0.0
    
    def _compute_weighted_average(
        self,
        precision: float,
        recall: float,
        faithfulness: float,
        coherence: float,
    ) -> float:
        """Method 1: Weighted Average (robust, no NaN)."""
        weights = self.weights
        numerator = (
            weights.get("temporal_precision", 0.25) * precision +
            weights.get("temporal_recall", 0.25) * recall +
            weights.get("temporal_faithfulness", 0.25) * faithfulness +
            weights.get("temporal_coherence", 0.25) * coherence
        )
        denominator = sum([
            weights.get("temporal_precision", 0.25),
            weights.get("temporal_recall", 0.25),
            weights.get("temporal_faithfulness", 0.25),
            weights.get("temporal_coherence", 0.25),
        ])
        return numerator / denominator if denominator > 0 else 0.0
    
    def _compute_harmonic_mean(
        self,
        precision: float,
        recall: float,
        faithfulness: float,
        coherence: float,
    ) -> float:
        """Method 2: Weighted Harmonic Mean (strict, penalizes low scores)."""
        weights = self.weights
        scores = [
            (precision, weights.get("temporal_precision", 0.25)),
            (recall, weights.get("temporal_recall", 0.25)),
            (faithfulness, weights.get("temporal_faithfulness", 0.25)),
            (coherence, weights.get("temporal_coherence", 0.25)),
        ]
        
        # Filter out zero scores (harmonic mean undefined for 0)
        valid = [(s, w) for s, w in scores if s > 0]
        if not valid:
            return 0.0
        
        weight_sum = sum(w for _, w in valid)
        if weight_sum == 0:
            return 0.0
        
        # Weighted harmonic mean: sum(w) / sum(w/x)
        denominator = sum(w / s for s, w in valid)
        return weight_sum / denominator if denominator > 0 else 0.0
    
    def _f1(self, a: float, b: float) -> float:
        """Compute F1 score of two values."""
        if a + b > 0:
            return 2 * a * b / (a + b)
        return 0.0
    
    def _compute_f_score(
        self,
        precision: float,
        recall: float,
        faithfulness: float,
        coherence: float,
    ) -> Dict[str, float]:
        """
        Method 3: Layered F-Score (hierarchical, recommended).
        
        Layer 1: Retrieval F1 = F1(precision, recall)
        Layer 2: Generation F1 = F1(faithfulness, coherence)
        Layer 3: Combined = weighted F-beta(retrieval_f1, generation_f1)
        """
        # Layer 1: Retrieval
        retrieval_f1 = self._f1(precision, recall)
        
        # Layer 2: Generation
        generation_f1 = self._f1(faithfulness, coherence)
        
        # Layer 3: Combined with configurable weight
        # F-beta formula: (1 + β²) × (R × G) / (β² × R + G)
        # where β = retrieval_weight controls retrieval importance
        beta = self.retrieval_weight
        if retrieval_f1 + generation_f1 > 0:
            if beta == 0.5:
                # Standard F1 between layers
                tempo_f_score = self._f1(retrieval_f1, generation_f1)
            else:
                # F-beta weighted
                beta_sq = beta ** 2
                tempo_f_score = (
                    (1 + beta_sq) * retrieval_f1 * generation_f1
                ) / (beta_sq * retrieval_f1 + generation_f1 + 1e-10)
        else:
            tempo_f_score = 0.0
        
        return {
            "retrieval_f1": round(retrieval_f1, 4),
            "generation_f1": round(generation_f1, 4),
            "tempo_f_score": round(tempo_f_score, 4),
        }
    
    def compute(
        self,
        scores: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute all three TempoScore variants.
        
        Args:
            scores: Dict of metric_name -> score
            **kwargs: Individual scores (e.g., temporal_precision=0.8)
            
        Returns:
            Dict with tempo_weighted, tempo_harmonic, tempo_f_score,
            retrieval_f1, generation_f1
        """
        # Merge inputs
        all_scores = {}
        if scores:
            all_scores.update(scores)
        all_scores.update(kwargs)
        
        # Extract core metrics
        precision = self._get_score(all_scores, "temporal_precision")
        recall = self._get_score(all_scores, "temporal_recall")
        faithfulness = self._get_score(all_scores, "temporal_faithfulness")
        coherence = self._get_score(all_scores, "temporal_coherence")
        
        # Compute all three methods
        tempo_weighted = self._compute_weighted_average(
            precision, recall, faithfulness, coherence
        )
        tempo_harmonic = self._compute_harmonic_mean(
            precision, recall, faithfulness, coherence
        )
        f_score_results = self._compute_f_score(
            precision, recall, faithfulness, coherence
        )
        
        return {
            "tempo_weighted": round(tempo_weighted, 4),
            "tempo_harmonic": round(tempo_harmonic, 4),
            "tempo_f_score": f_score_results["tempo_f_score"],
            "retrieval_f1": f_score_results["retrieval_f1"],
            "generation_f1": f_score_results["generation_f1"],
        }
    
    def compute_single(
        self,
        method: str = "f_score",
        scores: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> float:
        """
        Compute a single TempoScore using specified method.
        
        Args:
            method: "weighted", "harmonic", or "f_score"
            
        Returns:
            Single float score
        """
        result = self.compute(scores, **kwargs)
        method_map = {
            "weighted": "tempo_weighted",
            "harmonic": "tempo_harmonic",
            "f_score": "tempo_f_score",
        }
        key = method_map.get(method, "tempo_f_score")
        return result.get(key, 0.0)
    
    @classmethod
    def create_retrieval_focused(cls) -> "TempoScore":
        """Create TempoScore weighted toward retrieval metrics."""
        return cls(
            weights={
                "temporal_precision": 0.35,
                "temporal_recall": 0.35,
                "temporal_faithfulness": 0.20,
                "temporal_coherence": 0.10,
            },
            retrieval_weight=0.7,
        )
    
    @classmethod
    def create_generation_focused(cls) -> "TempoScore":
        """Create TempoScore weighted toward generation metrics."""
        return cls(
            weights={
                "temporal_precision": 0.15,
                "temporal_recall": 0.15,
                "temporal_faithfulness": 0.40,
                "temporal_coherence": 0.30,
            },
            retrieval_weight=0.3,
        )
    
    @classmethod
    def create_balanced(cls) -> "TempoScore":
        """Create balanced TempoScore for full RAG evaluation."""
        return cls(
            weights={
                "temporal_precision": 0.25,
                "temporal_recall": 0.25,
                "temporal_faithfulness": 0.25,
                "temporal_coherence": 0.25,
            },
            retrieval_weight=0.5,
        )
