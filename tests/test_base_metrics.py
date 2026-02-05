"""Tests for TempoEval base classes and metric infrastructure."""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.core.base import (
    BaseMetric,
    BaseRetrievalMetric,
    BaseGenerationMetric,
    BaseReasoningMetric,
    BaseCompositeMetric,
)


# Test implementations
@dataclass
class DummyRetrievalMetric(BaseRetrievalMetric):
    """Dummy retrieval metric for testing."""
    name: str = field(init=False, default="dummy_retrieval")
    
    def compute(self, retrieved_ids=None, gold_ids=None, k=10, **kwargs):
        if not retrieved_ids or not gold_ids:
            return 0.0
        overlap = len(set(retrieved_ids[:k]) & set(gold_ids))
        return overlap / len(gold_ids) if gold_ids else 0.0


@dataclass
class DummyGenerationMetric(BaseGenerationMetric):
    """Dummy generation metric for testing."""
    name: str = field(init=False, default="dummy_generation")
    
    def compute(self, answer=None, contexts=None, **kwargs):
        if not answer:
            return 0.0
        return 0.8  # Fixed score for testing


@dataclass
class DummyReasoningMetric(BaseReasoningMetric):
    """Dummy reasoning metric for testing."""
    name: str = field(init=False, default="dummy_reasoning")
    
    def compute(self, answer=None, **kwargs):
        if not answer:
            return 0.0
        return 0.7  # Fixed score for testing


class TestBaseRetrievalMetric:
    """Tests for BaseRetrievalMetric."""

    def test_create_metric(self):
        """Test creating a retrieval metric."""
        metric = DummyRetrievalMetric()
        assert metric.name == "dummy_retrieval"
        assert metric.metric_type == "retrieval"

    def test_compute(self):
        """Test computing retrieval metric."""
        metric = DummyRetrievalMetric()
        score = metric.compute(
            retrieved_ids=["d1", "d2", "d3"],
            gold_ids=["d1", "d2"],
            k=3
        )
        assert score == 1.0

    def test_compute_partial(self):
        """Test computing with partial overlap."""
        metric = DummyRetrievalMetric()
        score = metric.compute(
            retrieved_ids=["d1", "d3"],
            gold_ids=["d1", "d2"],
            k=2
        )
        assert score == 0.5

    def test_compute_empty(self):
        """Test computing with empty inputs."""
        metric = DummyRetrievalMetric()
        score = metric.compute(retrieved_ids=[], gold_ids=["d1"])
        assert score == 0.0


class TestBaseGenerationMetric:
    """Tests for BaseGenerationMetric."""

    def test_create_metric(self):
        """Test creating a generation metric."""
        metric = DummyGenerationMetric()
        assert metric.name == "dummy_generation"
        assert metric.metric_type == "generation"

    def test_compute(self):
        """Test computing generation metric."""
        metric = DummyGenerationMetric()
        score = metric.compute(
            answer="The event happened in 2020.",
            contexts=["Context about 2020 events."]
        )
        assert score == 0.8

    def test_compute_empty_answer(self):
        """Test computing with empty answer."""
        metric = DummyGenerationMetric()
        score = metric.compute(answer="", contexts=["Context"])
        assert score == 0.0


class TestBaseReasoningMetric:
    """Tests for BaseReasoningMetric."""

    def test_create_metric(self):
        """Test creating a reasoning metric."""
        metric = DummyReasoningMetric()
        assert metric.name == "dummy_reasoning"
        assert metric.metric_type == "reasoning"

    def test_compute(self):
        """Test computing reasoning metric."""
        metric = DummyReasoningMetric()
        score = metric.compute(answer="Events A then B then C.")
        assert score == 0.7


class TestMetricAttributes:
    """Tests for metric attributes."""

    def test_requires_llm_default(self):
        """Test that requires_llm defaults to False."""
        metric = DummyRetrievalMetric()
        assert metric.requires_llm is False

    def test_requires_gold_default(self):
        """Test that requires_gold defaults to False."""
        metric = DummyRetrievalMetric()
        assert metric.requires_gold is False

    def test_lower_is_better_default(self):
        """Test that lower_is_better defaults to False."""
        metric = DummyRetrievalMetric()
        assert metric.lower_is_better is False

    def test_repr(self):
        """Test metric string representation."""
        metric = DummyRetrievalMetric()
        repr_str = repr(metric)
        assert "DummyRetrievalMetric" in repr_str
        assert "dummy_retrieval" in repr_str


class TestAsyncCompute:
    """Tests for async compute methods."""

    @pytest.mark.asyncio
    async def test_acompute_default(self):
        """Test that acompute defaults to sync compute."""
        metric = DummyRetrievalMetric()
        score = await metric.acompute(
            retrieved_ids=["d1", "d2"],
            gold_ids=["d1", "d2"],
            k=2
        )
        assert score == 1.0


class TestCompositeMetricBase:
    """Tests for BaseCompositeMetric."""

    def test_harmonic_mean_basic(self):
        """Test basic harmonic mean calculation."""
        scores = [0.8, 0.8]
        result = BaseCompositeMetric.harmonic_mean(scores)
        assert result == 0.8

    def test_harmonic_mean_different(self):
        """Test harmonic mean with different values."""
        scores = [0.5, 1.0]
        result = BaseCompositeMetric.harmonic_mean(scores)
        # Harmonic mean of 0.5 and 1.0 = 2 / (1/0.5 + 1/1.0) = 2/3 ≈ 0.6667
        assert round(result, 4) == round(2/3, 4)

    def test_harmonic_mean_with_zero(self):
        """Test harmonic mean handles zero by filtering."""
        scores = [0.0, 0.8]
        result = BaseCompositeMetric.harmonic_mean(scores)
        # Zero is filtered out, so just 0.8
        assert result == 0.8

    def test_harmonic_mean_weighted(self):
        """Test weighted harmonic mean."""
        scores = [0.8, 0.6]
        weights = [2.0, 1.0]  # Weight first score more
        result = BaseCompositeMetric.harmonic_mean(scores, weights)
        # Should be closer to 0.8 than 0.6
        assert 0.7 < result < 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
