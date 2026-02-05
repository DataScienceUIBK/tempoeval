"""Tests for TempoEval composite metrics (TempoScore)."""

import pytest
import math
from tempoeval.metrics import TempoScore
from tempoeval.core.base import BaseCompositeMetric


class TestTempoScoreComputation:
    """Tests for TempoScore composite metric."""

    def test_compute_all_methods(self):
        """Test computing all three TempoScore methods."""
        metric = TempoScore()
        result = metric.compute(
            temporal_precision=0.8,
            temporal_recall=0.6,
            temporal_faithfulness=0.7,
            temporal_coherence=0.9,
        )
        
        assert "tempo_weighted" in result
        assert "tempo_harmonic" in result
        assert "tempo_f_score" in result
        assert "retrieval_f1" in result
        assert "generation_f1" in result

    def test_weighted_average(self):
        """Test weighted average calculation."""
        metric = TempoScore()
        result = metric.compute(
            temporal_precision=1.0,
            temporal_recall=1.0,
            temporal_faithfulness=1.0,
            temporal_coherence=1.0,
        )
        # All perfect scores -> weighted = 1.0
        assert result["tempo_weighted"] == 1.0

    def test_harmonic_mean(self):
        """Test harmonic mean calculation."""
        metric = TempoScore()
        result = metric.compute(
            temporal_precision=1.0,
            temporal_recall=1.0,
            temporal_faithfulness=1.0,
            temporal_coherence=1.0,
        )
        # All perfect scores -> harmonic = 1.0
        assert result["tempo_harmonic"] == 1.0

    def test_harmonic_mean_with_zero(self):
        """Test that harmonic mean filters zero scores."""
        metric = TempoScore()
        result = metric.compute(
            temporal_precision=0.0,  # Zero - will be filtered
            temporal_recall=1.0,
            temporal_faithfulness=1.0,
            temporal_coherence=1.0,
        )
        # Zero is filtered out, harmonic computed on remaining 3
        assert result["tempo_harmonic"] == 1.0  # Harmonic of [1, 1, 1] = 1

    def test_f_score_computation(self):
        """Test F-score computation."""
        metric = TempoScore()
        result = metric.compute(
            temporal_precision=0.8,
            temporal_recall=0.8,
            temporal_faithfulness=0.8,
            temporal_coherence=0.8,
        )
        # All same values -> F1 should be same
        assert result["retrieval_f1"] == 0.8
        assert result["generation_f1"] == 0.8

    def test_from_scores_dict(self):
        """Test computing from a scores dictionary."""
        metric = TempoScore()
        scores = {
            "temporal_precision@10": 0.75,
            "temporal_recall@10": 0.65,
            "temporal_faithfulness": 0.80,
            "temporal_coherence": 0.85,
        }
        result = metric.compute(scores=scores)
        
        assert "tempo_weighted" in result
        assert result["tempo_weighted"] > 0

    def test_compute_single_weighted(self):
        """Test compute_single with weighted method."""
        metric = TempoScore()
        score = metric.compute_single(
            method="weighted",
            temporal_precision=0.8,
            temporal_recall=0.6,
            temporal_faithfulness=0.7,
            temporal_coherence=0.5,
        )
        # (0.8 + 0.6 + 0.7 + 0.5) / 4 = 0.65 (if equal weights)
        assert 0.6 <= score <= 0.7

    def test_compute_single_f_score(self):
        """Test compute_single with f_score method."""
        metric = TempoScore()
        score = metric.compute_single(
            method="f_score",
            temporal_precision=0.8,
            temporal_recall=0.8,
            temporal_faithfulness=0.8,
            temporal_coherence=0.8,
        )
        assert score == 0.8  # All same -> F1 = same value


class TestTempoScorePresets:
    """Tests for TempoScore preset configurations."""

    def test_retrieval_focused(self):
        """Test retrieval-focused preset."""
        metric = TempoScore.create_retrieval_focused()
        assert metric.weights["temporal_precision"] > 0.25
        assert metric.weights["temporal_recall"] > 0.25

    def test_generation_focused(self):
        """Test generation-focused preset."""
        metric = TempoScore.create_generation_focused()
        assert metric.weights["temporal_faithfulness"] > 0.25
        assert metric.weights["temporal_coherence"] > 0.25

    def test_balanced(self):
        """Test balanced preset."""
        metric = TempoScore.create_balanced()
        # All weights should be equal
        weights = list(metric.weights.values())
        assert all(w == weights[0] for w in weights)


class TestHarmonicMean:
    """Tests for harmonic mean static method."""

    def test_harmonic_mean_basic(self):
        """Test basic harmonic mean."""
        scores = [0.5, 0.5]
        result = BaseCompositeMetric.harmonic_mean(scores)
        assert result == 0.5

    def test_harmonic_mean_empty(self):
        """Test harmonic mean with empty list."""
        result = BaseCompositeMetric.harmonic_mean([])
        assert math.isnan(result)

    def test_harmonic_mean_with_weights(self):
        """Test weighted harmonic mean."""
        scores = [0.5, 0.5]
        weights = [1.0, 1.0]
        result = BaseCompositeMetric.harmonic_mean(scores, weights)
        assert result == 0.5

    def test_harmonic_mean_filters_nan(self):
        """Test that harmonic mean filters NaN values."""
        scores = [0.5, float('nan'), 0.5]
        result = BaseCompositeMetric.harmonic_mean(scores)
        assert result == 0.5


class TestTempoScoreAttributes:
    """Tests for TempoScore attributes."""

    def test_name(self):
        """Test TempoScore name attribute."""
        metric = TempoScore()
        assert metric.name == "tempo_score"

    def test_metric_type(self):
        """Test TempoScore metric type."""
        metric = TempoScore()
        assert metric.metric_type == "composite"

    def test_default_weights(self):
        """Test TempoScore default weights."""
        metric = TempoScore()
        assert "temporal_precision" in metric.weights
        assert "temporal_recall" in metric.weights
        assert "temporal_faithfulness" in metric.weights
        assert "temporal_coherence" in metric.weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
