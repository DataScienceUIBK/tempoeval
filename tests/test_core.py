"""Tests for TempoEval core module - Focus Time extraction and utilities."""

import pytest
from tempoeval.core import (
    FocusTime,
    FocusTimeExtractor,
    ExtractionMethod,
    extract_qft,
    extract_dft,
    extract_aft,
    compute_temporal_relevance,
)


class TestFocusTime:
    """Tests for FocusTime dataclass."""

    def test_create_from_years(self):
        """Test creating FocusTime from a list of years."""
        ft = FocusTime.from_years([2020, 2021, 2022])
        assert ft.years == {2020, 2021, 2022}

    def test_create_from_set(self):
        """Test creating FocusTime from a set of years."""
        ft = FocusTime.from_years({2020, 2021})
        assert ft.years == {2020, 2021}

    def test_create_from_range(self):
        """Test creating FocusTime from a year range."""
        ft = FocusTime.from_range(2018, 2022)
        assert ft.years == {2018, 2019, 2020, 2021, 2022}

    def test_intersection(self):
        """Test FocusTime intersection (AND) - returns set."""
        ft1 = FocusTime.from_years([2019, 2020, 2021])
        ft2 = FocusTime.from_years([2020, 2021, 2022])
        result = ft1 & ft2  # Returns a set
        assert result == {2020, 2021}

    def test_union(self):
        """Test FocusTime union (OR) - returns set."""
        ft1 = FocusTime.from_years([2019, 2020])
        ft2 = FocusTime.from_years([2021, 2022])
        result = ft1 | ft2  # Returns a set
        assert result == {2019, 2020, 2021, 2022}

    def test_len(self):
        """Test FocusTime length."""
        ft = FocusTime.from_years([2020, 2021, 2022])
        assert len(ft) == 3

    def test_jaccard_similarity(self):
        """Test Jaccard similarity between Focus Times."""
        ft1 = FocusTime.from_years([2020, 2021])
        ft2 = FocusTime.from_years([2020, 2021, 2022])
        # Intersection = {2020, 2021} = 2
        # Union = {2020, 2021, 2022} = 3
        # Jaccard = 2/3 = 0.6667
        jaccard = ft1.jaccard(ft2)
        assert round(jaccard, 4) == round(2/3, 4)

    def test_jaccard_no_overlap(self):
        """Test Jaccard similarity with no overlap."""
        ft1 = FocusTime.from_years([2020])
        ft2 = FocusTime.from_years([2022])
        assert ft1.jaccard(ft2) == 0.0

    def test_overlap_count(self):
        """Test overlap count between Focus Times."""
        ft1 = FocusTime.from_years([2019, 2020, 2021])
        ft2 = FocusTime.from_years([2020, 2021, 2022])
        assert ft1.overlap(ft2) == 2

    def test_covers(self):
        """Test if one FocusTime covers another (any overlap by default)."""
        ft1 = FocusTime.from_years([2019, 2020, 2021, 2022])
        ft2 = FocusTime.from_years([2020, 2021])
        assert ft1.covers(ft2)  # ft1 has overlap with ft2


class TestFocusTimeExtractor:
    """Tests for FocusTimeExtractor class."""

    def test_extract_explicit_years(self):
        """Test extracting explicit years from text."""
        extractor = FocusTimeExtractor()
        text = "The 2008 financial crisis affected markets until 2010."
        years = extractor.extract_explicit_years(text)
        assert 2008 in years
        assert 2010 in years

    def test_extract_explicit_years_range(self):
        """Test extracting years from text."""
        extractor = FocusTimeExtractor()
        text = "From 1990 to 1995, the country experienced growth."
        years = extractor.extract_explicit_years(text)
        assert 1990 in years
        assert 1995 in years

    def test_extract_decades(self):
        """Test extracting decades."""
        extractor = FocusTimeExtractor()
        text = "During the 1990s, technology advanced rapidly."
        years = extractor.extract_explicit_years(text)
        # Should contain years 1990-1999
        assert 1990 in years
        assert 1999 in years


class TestExtractFunctions:
    """Tests for extract_qft, extract_dft, extract_aft convenience functions."""

    def test_extract_qft(self):
        """Test Query Focus Time extraction."""
        query = "What happened during the 2008 financial crisis?"
        ft = extract_qft(query)
        assert isinstance(ft, FocusTime)
        assert 2008 in ft.years

    def test_extract_dft(self):
        """Test Document Focus Time extraction."""
        document = "Lehman Brothers collapsed in September 2008, triggering a global recession."
        ft = extract_dft(document)
        assert isinstance(ft, FocusTime)
        assert 2008 in ft.years

    def test_extract_aft(self):
        """Test Answer Focus Time extraction."""
        answer = "The crisis began in 2008 and recovery started in 2009."
        ft = extract_aft(answer)
        assert isinstance(ft, FocusTime)
        assert 2008 in ft.years
        assert 2009 in ft.years

    def test_extract_qft_no_years(self):
        """Test QFT extraction with no explicit years."""
        query = "What is the capital of France?"
        ft = extract_qft(query)
        assert isinstance(ft, FocusTime)
        # May have years from implicit references or none


class TestTemporalRelevance:
    """Tests for compute_temporal_relevance function."""

    def test_temporal_relevance_full_overlap(self):
        """Test relevance with full overlap."""
        qft = FocusTime.from_years([2020])
        dft = FocusTime.from_years([2020])
        is_rel, score = compute_temporal_relevance(qft, dft)
        assert is_rel is True
        assert score == 1.0

    def test_temporal_relevance_partial_overlap_jaccard(self):
        """Test relevance with partial overlap using Jaccard mode."""
        qft = FocusTime.from_years([2020, 2021])
        dft = FocusTime.from_years([2020, 2022])
        is_rel, score = compute_temporal_relevance(qft, dft, mode="jaccard")
        # Jaccard = 1/3 ≈ 0.333
        assert is_rel is True
        assert 0.3 < score < 0.4

    def test_temporal_relevance_no_overlap(self):
        """Test relevance with no overlap."""
        qft = FocusTime.from_years([2020])
        dft = FocusTime.from_years([2022])
        is_rel, score = compute_temporal_relevance(qft, dft)
        assert is_rel is False
        assert score == 0.0

    def test_temporal_relevance_empty(self):
        """Test relevance with empty Focus Times."""
        qft = FocusTime()
        dft = FocusTime.from_years([2020])
        is_rel, score = compute_temporal_relevance(qft, dft)
        assert is_rel is False
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
