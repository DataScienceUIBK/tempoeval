"""Tests for TempoEval efficiency tracker."""

import pytest
import time
from tempoeval.efficiency.tracker import (
    EfficiencyTracker,
    EfficiencyStats,
    AggregatedStats,
    MetricType,
)


class TestEfficiencyStats:
    """Tests for EfficiencyStats dataclass."""

    def test_create_stats(self):
        """Test creating efficiency stats."""
        stats = EfficiencyStats(
            operation_id="op1",
            metric_name="temporal_precision",
            start_time=time.time()
        )
        assert stats.operation_id == "op1"
        assert stats.metric_name == "temporal_precision"
        assert stats.latency_ms == 0.0

    def test_finalize(self):
        """Test finalizing stats."""
        start = time.time()
        stats = EfficiencyStats(
            operation_id="op1",
            metric_name="test",
            start_time=start,
            prompt_tokens=100,
            completion_tokens=50
        )
        time.sleep(0.01)  # Small delay
        stats.finalize()
        
        assert stats.latency_ms > 0
        assert stats.total_tokens == 150


class TestAggregatedStats:
    """Tests for AggregatedStats dataclass."""

    def test_create_aggregated(self):
        """Test creating aggregated stats."""
        agg = AggregatedStats()
        assert agg.count == 0
        assert agg.total_latency_ms == 0.0
        assert agg.total_cost_usd == 0.0


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types(self):
        """Test metric type enum values."""
        assert MetricType.RETRIEVAL.value == "retrieval"
        assert MetricType.GENERATION.value == "generation"
        assert MetricType.REASONING.value == "reasoning"
        assert MetricType.OTHER.value == "other"


class TestEfficiencyTracker:
    """Tests for EfficiencyTracker class."""

    def test_create_tracker(self):
        """Test creating efficiency tracker."""
        tracker = EfficiencyTracker()
        assert tracker.model_name == "gpt-4o"
        assert len(tracker.completed_stats) == 0

    def test_create_tracker_with_model(self):
        """Test creating tracker with specific model."""
        tracker = EfficiencyTracker(model_name="gpt-3.5-turbo")
        assert tracker.model_name == "gpt-3.5-turbo"
        # Should have lower costs
        assert tracker.cost_per_1k_input < 0.002

    def test_start_tracking(self):
        """Test starting to track an operation."""
        tracker = EfficiencyTracker()
        tracker.start_tracking("op1", "temporal_precision")
        
        assert "op1" in tracker.active_tracks
        assert tracker.active_tracks["op1"].metric_name == "temporal_precision"

    def test_stop_tracking(self):
        """Test stopping tracking."""
        tracker = EfficiencyTracker()
        tracker.start_tracking("op1", "test_metric")
        time.sleep(0.01)
        stats = tracker.stop_tracking("op1", prompt_chars=1000, completion_chars=200)
        
        assert stats is not None
        assert stats.latency_ms > 0
        assert "op1" not in tracker.active_tracks
        assert len(tracker.completed_stats) == 1

    def test_stop_tracking_with_tokens(self):
        """Test stopping tracking with explicit token counts."""
        tracker = EfficiencyTracker()
        tracker.start_tracking("op1", "test")
        stats = tracker.stop_tracking("op1", prompt_tokens=500, completion_tokens=100)
        
        assert stats.prompt_tokens == 500
        assert stats.completion_tokens == 100
        assert stats.total_tokens == 600

    def test_stop_nonexistent(self):
        """Test stopping a non-existent operation."""
        tracker = EfficiencyTracker()
        result = tracker.stop_tracking("nonexistent")
        assert result is None

    def test_aggregated_stats(self):
        """Test getting aggregated statistics."""
        tracker = EfficiencyTracker()
        
        # Track two operations for same metric
        tracker.start_tracking("op1", "precision")
        tracker.stop_tracking("op1", prompt_tokens=100, completion_tokens=50)
        
        tracker.start_tracking("op2", "precision")
        tracker.stop_tracking("op2", prompt_tokens=200, completion_tokens=100)
        
        # Track one for different metric
        tracker.start_tracking("op3", "recall")
        tracker.stop_tracking("op3", prompt_tokens=150, completion_tokens=75)
        
        agg = tracker.get_aggregated_stats()
        
        assert "precision" in agg
        assert "recall" in agg
        assert agg["precision"].count == 2
        assert agg["recall"].count == 1
        assert agg["precision"].total_tokens == 450  # 150 + 300

    def test_summary(self):
        """Test generating summary."""
        tracker = EfficiencyTracker()
        
        tracker.start_tracking("op1", "test")
        tracker.stop_tracking("op1", prompt_tokens=1000, completion_tokens=500)
        
        summary = tracker.summary()
        
        assert "total_operations" in summary
        assert summary["total_operations"] == 1
        assert "total_cost_usd" in summary
        assert summary["total_cost_usd"] > 0
        assert "metrics_breakdown" in summary

    def test_cost_calculation(self):
        """Test cost calculation."""
        tracker = EfficiencyTracker(model_name="gpt-4o")
        
        tracker.start_tracking("op1", "test")
        stats = tracker.stop_tracking("op1", prompt_tokens=1000, completion_tokens=1000)
        
        # GPT-4o: $0.0025/1K input, $0.01/1K output
        expected_cost = (1000/1000) * 0.0025 + (1000/1000) * 0.01
        assert abs(stats.estimated_cost_usd - expected_cost) < 0.001


class TestTrackerIntegration:
    """Integration tests for efficiency tracker."""

    def test_full_evaluation_simulation(self):
        """Test simulating a full evaluation run."""
        tracker = EfficiencyTracker()
        
        # Simulate evaluating 3 queries with 2 metrics each
        for q in range(3):
            for metric in ["precision", "recall"]:
                op_id = f"q{q}_{metric}"
                tracker.start_tracking(op_id, metric)
                time.sleep(0.001)  # Simulate work
                tracker.stop_tracking(op_id, prompt_tokens=100, completion_tokens=50)
        
        summary = tracker.summary()
        
        assert summary["total_operations"] == 6
        
        agg = tracker.get_aggregated_stats()
        assert agg["precision"].count == 3
        assert agg["recall"].count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
