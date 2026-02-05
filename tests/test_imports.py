"""Tests for TempoEval imports and package structure."""

import pytest


class TestPackageImports:
    """Tests for importing TempoEval modules."""

    def test_import_main_package(self):
        """Test importing main tempoeval package."""
        import tempoeval
        assert hasattr(tempoeval, "__version__")
        assert hasattr(tempoeval, "evaluate")
        assert hasattr(tempoeval, "TempoEvalConfig")

    def test_import_core(self):
        """Test importing core module."""
        from tempoeval.core import (
            FocusTime,
            FocusTimeExtractor,
            extract_qft,
            extract_dft,
            TempoEvalConfig,
            EvaluationResult,
        )
        assert FocusTime is not None
        assert FocusTimeExtractor is not None
        assert TempoEvalConfig is not None

    def test_import_metrics(self):
        """Test importing metrics module."""
        from tempoeval.metrics import (
            TemporalPrecision,
            TemporalRecall,
            TemporalNDCG,
            TemporalCoverage,
            TemporalFaithfulness,
            EventOrdering,
            TempoScore,
            ALL_METRICS,
        )
        assert TemporalPrecision is not None
        assert TempoScore is not None
        assert len(ALL_METRICS) > 0

    def test_import_llm(self):
        """Test importing LLM module."""
        from tempoeval.llm import (
            BaseLLMProvider,
            get_provider,
        )
        assert BaseLLMProvider is not None
        assert callable(get_provider)

    def test_import_efficiency(self):
        """Test importing efficiency module."""
        from tempoeval.efficiency.tracker import EfficiencyTracker
        assert EfficiencyTracker is not None

    def test_import_utils(self):
        """Test importing utils module."""
        from tempoeval.utils import extract_temporal_expressions
        assert callable(extract_temporal_expressions)


class TestMetricCollections:
    """Tests for metric collections."""

    def test_all_metrics_list(self):
        """Test ALL_METRICS contains all metric classes."""
        from tempoeval.metrics import ALL_METRICS
        # Should contain multiple metrics
        assert len(ALL_METRICS) >= 10

    def test_no_llm_metrics(self):
        """Test NO_LLM_METRICS collection."""
        from tempoeval.metrics import NO_LLM_METRICS
        # All metrics in this list should not require LLM
        for metric_class in NO_LLM_METRICS:
            metric = metric_class()
            # Either requires_llm is False or it's a property that returns False
            if hasattr(metric, 'requires_llm'):
                if callable(getattr(type(metric), 'requires_llm', None)):
                    # It's a property
                    pass
                else:
                    assert metric.requires_llm is False


class TestVersionInfo:
    """Tests for version information."""

    def test_version_format(self):
        """Test version string format."""
        from tempoeval import __version__
        assert isinstance(__version__, str)
        # Should be semver-like (e.g., "0.1.0")
        parts = __version__.split(".")
        assert len(parts) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
