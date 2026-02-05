"""Tests for TempoEval generation metrics."""

import pytest
from tempoeval.metrics import (
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    AnswerTemporalAlignment,
)


class TestGenerationMetricNames:
    """Tests for generation metric names."""

    def test_faithfulness_name(self):
        """Test TemporalFaithfulness has correct name."""
        metric = TemporalFaithfulness()
        assert metric.name == "temporal_faithfulness"

    def test_hallucination_name(self):
        """Test TemporalHallucination has correct name."""
        metric = TemporalHallucination()
        assert metric.name == "temporal_hallucination"

    def test_coherence_name(self):
        """Test TemporalCoherence has correct name."""
        metric = TemporalCoherence()
        assert metric.name == "temporal_coherence"

    def test_alignment_name(self):
        """Test AnswerTemporalAlignment has correct name."""
        metric = AnswerTemporalAlignment()
        assert metric.name == "answer_temporal_alignment"


class TestGenerationMetricTypes:
    """Tests for generation metric types."""

    def test_faithfulness_type(self):
        """Test TemporalFaithfulness is a generation metric."""
        metric = TemporalFaithfulness()
        assert metric.metric_type == "generation"

    def test_hallucination_type(self):
        """Test TemporalHallucination is a generation metric."""
        metric = TemporalHallucination()
        assert metric.metric_type == "generation"

    def test_coherence_type(self):
        """Test TemporalCoherence is a generation metric."""
        metric = TemporalCoherence()
        assert metric.metric_type == "generation"


class TestGenerationMetricRequiresLLM:
    """Tests for LLM requirements."""

    def test_faithfulness_requires_llm(self):
        """Test TemporalFaithfulness requires LLM."""
        metric = TemporalFaithfulness()
        assert metric.requires_llm is True

    def test_hallucination_requires_llm(self):
        """Test TemporalHallucination requires LLM."""
        metric = TemporalHallucination()
        assert metric.requires_llm is True

    def test_coherence_requires_llm(self):
        """Test TemporalCoherence requires LLM."""
        metric = TemporalCoherence()
        assert metric.requires_llm is True


class TestFaithfulnessWithoutLLM:
    """Tests for TemporalFaithfulness error handling."""

    def test_compute_without_llm_raises(self):
        """Test that compute without LLM raises error."""
        metric = TemporalFaithfulness()
        with pytest.raises(ValueError):
            metric.compute(
                answer="The event happened in 2020.",
                contexts=["Context about 2020."]
            )


class TestHallucinationLowerIsBetter:
    """Tests for TemporalHallucination scoring."""

    def test_lower_is_better(self):
        """Test that hallucination uses lower is better."""
        metric = TemporalHallucination()
        assert metric.lower_is_better is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
