"""Tests for TempoEval evaluator integration."""

import pytest
from tempoeval.core.evaluator import Evaluator
from tempoeval.core import TempoEvalConfig, EvaluationResult, QueryResult
from tempoeval.metrics import TemporalPrecision, TemporalRecall, TemporalDiversity
from tempoeval.core import FocusTime


class TestEvaluatorCreation:
    """Tests for Evaluator instantiation."""

    def test_create_evaluator(self):
        """Test creating an evaluator."""
        metrics = [TemporalPrecision(use_focus_time=True)]
        evaluator = Evaluator(metrics=metrics)
        assert len(evaluator.metrics) == 1

    def test_create_evaluator_with_config(self):
        """Test creating an evaluator with config."""
        config = TempoEvalConfig(k_values=[5, 10])
        metrics = [TemporalPrecision(use_focus_time=True)]
        evaluator = Evaluator(metrics=metrics, config=config)
        assert evaluator.config.k_values == [5, 10]

    def test_create_evaluator_multiple_metrics(self):
        """Test creating evaluator with multiple metrics."""
        metrics = [
            TemporalPrecision(use_focus_time=True),
            TemporalRecall(use_focus_time=True),
            TemporalDiversity(use_focus_time=True),
        ]
        evaluator = Evaluator(metrics=metrics)
        assert len(evaluator.metrics) == 3


class TestEvaluatorQueryEvaluation:
    """Tests for single query evaluation."""

    def test_evaluate_query_basic(self):
        """Test evaluating a single query."""
        metrics = [TemporalPrecision(use_focus_time=True)]
        evaluator = Evaluator(metrics=metrics)
        
        qft = FocusTime.from_years([2020])
        dfts = [FocusTime.from_years([2020]), FocusTime.from_years([2020])]
        
        result = evaluator.evaluate_query(
            query_id="q1",
            query="Test query about 2020",
            qft=qft,
            dfts=dfts,
            k=2
        )
        
        assert isinstance(result, QueryResult)
        assert result.query_id == "q1"
        # Scores include @K suffix
        assert any("temporal_precision" in key for key in result.scores)


class TestEvaluatorAttributes:
    """Tests for evaluator attributes."""

    def test_evaluator_has_metrics(self):
        """Test evaluator stores metrics."""
        metrics = [TemporalPrecision(use_focus_time=True)]
        evaluator = Evaluator(metrics=metrics)
        assert hasattr(evaluator, 'metrics')
        assert len(evaluator.metrics) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
