"""Tests for TempoEval configuration and result containers."""

import pytest
import tempfile
import os
import json
from pathlib import Path

from tempoeval.core import TempoEvalConfig
from tempoeval.core.result import EvaluationResult, QueryResult


class TestTempoEvalConfig:
    """Tests for TempoEvalConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TempoEvalConfig()
        assert config.llm == "openai"
        assert config.k_values == [5, 10, 20]
        assert config.max_workers == 10
        assert config.temperature == 0.0
        assert config.max_retries == 6

    def test_custom_config(self):
        """Test custom configuration."""
        config = TempoEvalConfig(
            llm="azure",
            k_values=[10, 20],
            max_workers=5,
            temperature=0.7,
        )
        assert config.llm == "azure"
        assert config.k_values == [10, 20]
        assert config.max_workers == 5
        assert config.temperature == 0.7

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TempoEvalConfig(llm="azure")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["llm"] == "azure"
        assert "k_values" in d
        assert "max_workers" in d

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {"llm": "anthropic", "max_workers": 20}
        config = TempoEvalConfig.from_dict(d)
        assert config.llm == "anthropic"
        assert config.max_workers == 20


class TestQueryResult:
    """Tests for QueryResult container."""

    def test_create_query_result(self):
        """Test creating a query result."""
        result = QueryResult(
            query_id="q1",
            query="What happened in 2020?",
            scores={"temporal_precision@10": 0.8, "temporal_recall@10": 0.6}
        )
        assert result.query_id == "q1"
        assert result.query == "What happened in 2020?"
        assert result.scores["temporal_precision@10"] == 0.8

    def test_get_score(self):
        """Test getting a specific score."""
        result = QueryResult(
            query_id="q1",
            query="Test",
            scores={"metric1": 0.5, "metric2": 0.7}
        )
        assert result.get_score("metric1") == 0.5
        # Missing metric should return nan
        import math
        assert math.isnan(result.get_score("nonexistent"))

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = QueryResult(
            query_id="q1",
            query="Test query",
            scores={"metric1": 0.5},
            metadata={"domain": "test"}
        )
        d = result.to_dict()
        assert d["query_id"] == "q1"
        assert d["query"] == "Test query"
        assert d["scores"]["metric1"] == 0.5
        assert d["metadata"]["domain"] == "test"


class TestEvaluationResult:
    """Tests for EvaluationResult container."""

    def test_create_empty(self):
        """Test creating empty evaluation result."""
        result = EvaluationResult()
        assert len(result) == 0
        assert result.query_results == []

    def test_add_query_result(self):
        """Test adding query results."""
        result = EvaluationResult()
        qr = QueryResult(query_id="q1", query="Test", scores={"m1": 0.5})
        result.add_query_result(qr)
        assert len(result) == 1

    def test_mean_scores(self):
        """Test computing mean scores."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T1", scores={"m1": 0.6}))
        result.add_query_result(QueryResult(query_id="q2", query="T2", scores={"m1": 0.8}))
        
        means = result.mean()
        assert "m1" in means
        assert means["m1"] == 0.7

    def test_std_scores(self):
        """Test computing standard deviation."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T1", scores={"m1": 0.6}))
        result.add_query_result(QueryResult(query_id="q2", query="T2", scores={"m1": 0.8}))
        
        stds = result.std()
        assert "m1" in stds
        assert stds["m1"] > 0

    def test_aggregate(self):
        """Test aggregating scores."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T1", scores={"m1": 0.5}))
        result.add_query_result(QueryResult(query_id="q2", query="T2", scores={"m1": 0.7}))
        
        result.aggregate()
        assert result.aggregated_scores["m1"] == 0.6

    def test_get_score(self):
        """Test getting aggregated score."""
        result = EvaluationResult()
        result.aggregated_scores = {"m1": 0.75}
        assert result.get_score("m1") == 0.75

    def test_get_scores(self):
        """Test getting all aggregated scores."""
        result = EvaluationResult()
        result.aggregated_scores = {"m1": 0.75, "m2": 0.85}
        scores = result.get_scores()
        assert scores["m1"] == 0.75
        assert scores["m2"] == 0.85

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T", scores={"m1": 0.5}))
        result.aggregated_scores = {"m1": 0.5}
        
        d = result.to_dict()
        assert "query_results" in d
        assert "aggregated_scores" in d
        assert len(d["query_results"]) == 1

    def test_to_json(self):
        """Test saving to JSON file."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T", scores={"m1": 0.5}))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            result.to_json(filepath)
            
            assert os.path.exists(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert "query_results" in data

    def test_to_csv(self):
        """Test saving to CSV file."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T1", scores={"m1": 0.5}))
        result.add_query_result(QueryResult(query_id="q2", query="T2", scores={"m1": 0.7}))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.csv")
            result.to_csv(filepath)
            
            assert os.path.exists(filepath)
            with open(filepath) as f:
                content = f.read()
            assert "query_id" in content
            assert "q1" in content

    def test_summary(self):
        """Test generating summary string."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T1", scores={"m1": 0.5}))
        result.add_query_result(QueryResult(query_id="q2", query="T2", scores={"m1": 0.7}))
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert "TempoEval" in summary
        assert "m1" in summary

    def test_get_query_results(self):
        """Test getting results for specific query."""
        result = EvaluationResult()
        qr = QueryResult(query_id="q1", query="T1", scores={"m1": 0.5})
        result.add_query_result(qr)
        
        found = result.get_query_results("q1")
        assert found is not None
        assert found.query_id == "q1"
        
        not_found = result.get_query_results("q999")
        assert not_found is None

    def test_iter_queries(self):
        """Test iterating over query results."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T1", scores={"m1": 0.5}))
        result.add_query_result(QueryResult(query_id="q2", query="T2", scores={"m1": 0.7}))
        
        query_ids = []
        for qid, scores in result.iter_queries():
            query_ids.append(qid)
        
        assert "q1" in query_ids
        assert "q2" in query_ids

    def test_repr(self):
        """Test string representation."""
        result = EvaluationResult()
        result.add_query_result(QueryResult(query_id="q1", query="T", scores={"m1": 0.5}))
        result.aggregate()
        
        repr_str = repr(result)
        assert "EvaluationResult" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
