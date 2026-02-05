"""Tests for TempoEval retrieval metrics."""

import pytest
from tempoeval.core import FocusTime
from tempoeval.metrics import (
    TemporalPrecision,
    TemporalRecall,
    TemporalDiversity,
    TemporalYearPrecision,
    TemporalYearRecall,
)


class TestTemporalPrecisionFocusTime:
    """Tests for TemporalPrecision with Focus Time mode."""

    def test_basic_precision(self):
        """Test basic precision calculation."""
        metric = TemporalPrecision(use_focus_time=True)
        qft = FocusTime.from_years([2020, 2021])
        dfts = [
            FocusTime.from_years([2020]),      # Relevant
            FocusTime.from_years([2019]),      # Not relevant
            FocusTime.from_years([2021]),      # Relevant
        ]
        score = metric.compute(qft=qft, dfts=dfts, k=3)
        # 2 relevant / 3 total = 0.6667
        assert round(score, 4) == round(2/3, 4)

    def test_precision_all_relevant(self):
        """Test precision when all documents are relevant."""
        metric = TemporalPrecision(use_focus_time=True)
        qft = FocusTime.from_years([2020])
        dfts = [FocusTime.from_years([2020]) for _ in range(5)]
        score = metric.compute(qft=qft, dfts=dfts, k=5)
        assert score == 1.0

    def test_precision_none_relevant(self):
        """Test precision when no documents are relevant."""
        metric = TemporalPrecision(use_focus_time=True)
        qft = FocusTime.from_years([2020])
        dfts = [FocusTime.from_years([2019]) for _ in range(5)]
        score = metric.compute(qft=qft, dfts=dfts, k=5)
        assert score == 0.0

    def test_precision_k_cutoff(self):
        """Test precision with K cutoff."""
        metric = TemporalPrecision(use_focus_time=True)
        qft = FocusTime.from_years([2020])
        dfts = [
            FocusTime.from_years([2020]),      # Relevant (pos 1)
            FocusTime.from_years([2020]),      # Relevant (pos 2)
            FocusTime.from_years([2019]),      # Not relevant (pos 3)
            FocusTime.from_years([2020]),      # Relevant (pos 4) - excluded
            FocusTime.from_years([2020]),      # Relevant (pos 5) - excluded
        ]
        score = metric.compute(qft=qft, dfts=dfts, k=3)
        assert round(score, 4) == round(2/3, 4)

    def test_precision_empty_qft(self):
        """Test precision with empty QFT."""
        metric = TemporalPrecision(use_focus_time=True)
        qft = FocusTime()
        dfts = [FocusTime.from_years([2020])]
        score = metric.compute(qft=qft, dfts=dfts, k=1)
        assert score == 0.0


class TestTemporalRecallFocusTime:
    """Tests for TemporalRecall with Focus Time mode."""

    def test_basic_recall(self):
        """Test basic recall calculation."""
        metric = TemporalRecall(use_focus_time=True)
        qft = FocusTime.from_years([2020, 2021])
        dfts = [
            FocusTime.from_years([2020]),      # Relevant
            FocusTime.from_years([2021]),      # Relevant
            FocusTime.from_years([2019]),      # Not relevant
        ]
        # 2 relevant in top-3 / 2 total relevant
        score = metric.compute(qft=qft, dfts=dfts, total_relevant=2, k=3)
        assert score == 1.0

    def test_recall_partial(self):
        """Test recall when not all relevant docs are retrieved."""
        metric = TemporalRecall(use_focus_time=True)
        qft = FocusTime.from_years([2020])
        dfts = [
            FocusTime.from_years([2020]),      # Relevant
            FocusTime.from_years([2019]),      # Not relevant
        ]
        # 1 relevant in top-2 / 3 total relevant in collection
        score = metric.compute(qft=qft, dfts=dfts, total_relevant=3, k=2)
        assert round(score, 4) == round(1/3, 4)


class TestTemporalRecallGold:
    """Tests for TemporalRecall with gold labels."""

    def test_gold_recall_full(self):
        """Test gold-based recall with full retrieval."""
        metric = TemporalRecall()  # Default gold mode
        score = metric.compute(
            retrieved_ids=["doc1", "doc2", "doc3"],
            gold_ids=["doc1", "doc2"],
            k=3
        )
        assert score == 1.0

    def test_gold_recall_partial(self):
        """Test gold-based recall with partial retrieval."""
        metric = TemporalRecall()
        score = metric.compute(
            retrieved_ids=["doc1", "doc3", "doc4"],
            gold_ids=["doc1", "doc2"],
            k=3
        )
        assert score == 0.5

    def test_gold_recall_no_overlap(self):
        """Test gold-based recall with no overlap."""
        metric = TemporalRecall()
        score = metric.compute(
            retrieved_ids=["doc3", "doc4"],
            gold_ids=["doc1", "doc2"],
            k=2
        )
        assert score == 0.0


class TestTemporalDiversity:
    """Tests for TemporalDiversity metric."""

    def test_diversity_with_focus_time(self):
        """Test temporal diversity with focus time mode."""
        metric = TemporalDiversity(use_focus_time=True)
        dfts = [
            FocusTime.from_years([2018]),
            FocusTime.from_years([2019]),
            FocusTime.from_years([2020]),
        ]
        # 3 unique years / 3 total scope
        score = metric.compute(dfts=dfts, total_scope=10, k=3)
        assert score == 0.3  # 3/10

    def test_diversity_same_years(self):
        """Test diversity when all docs have same year."""
        metric = TemporalDiversity(use_focus_time=True)
        dfts = [FocusTime.from_years([2020]) for _ in range(5)]
        # 1 unique year / 10 total scope
        score = metric.compute(dfts=dfts, total_scope=10, k=5)
        assert score == 0.1  # 1/10

    def test_empty_diversity(self):
        """Test diversity with no documents."""
        metric = TemporalDiversity(use_focus_time=True)
        score = metric.compute(dfts=[], k=5)
        assert score == 0.0


class TestTemporalYearPrecision:
    """Tests for TemporalYearPrecision metric."""

    def test_year_precision_exact_match(self):
        """Test year precision with exact match."""
        metric = TemporalYearPrecision()
        qft = FocusTime.from_years([2020])
        dfts = [FocusTime.from_years([2020])]
        score = metric.compute(qft=qft, dfts=dfts, k=1)
        assert score == 1.0


class TestTemporalYearRecall:
    """Tests for TemporalYearRecall metric."""

    def test_year_recall_full(self):
        """Test year recall with full coverage."""
        metric = TemporalYearRecall()
        qft = FocusTime.from_years([2020, 2021])
        dfts = [
            FocusTime.from_years([2020, 2021]),
        ]
        score = metric.compute(qft=qft, dfts=dfts, k=1)
        assert score == 1.0

    def test_year_recall_partial(self):
        """Test year recall with partial coverage."""
        metric = TemporalYearRecall()
        qft = FocusTime.from_years([2020, 2021, 2022])
        dfts = [
            FocusTime.from_years([2020]),
            FocusTime.from_years([2021]),
        ]
        # Only 2/3 years covered
        score = metric.compute(qft=qft, dfts=dfts, k=2)
        assert round(score, 4) == round(2/3, 4)


class TestMetricNames:
    """Tests for metric name attributes."""

    def test_precision_name(self):
        """Test TemporalPrecision has correct name."""
        metric = TemporalPrecision()
        assert metric.name == "temporal_precision"

    def test_recall_name(self):
        """Test TemporalRecall has correct name."""
        metric = TemporalRecall()
        assert metric.name == "temporal_recall"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
