"""Temporal expression parser utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TemporalExpression:
    """A parsed temporal expression."""
    text: str
    type: str  # year, date, duration, relative, range
    normalized: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


# Regex patterns for temporal expressions
YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')
DATE_PATTERN = re.compile(
    r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+ \d{1,2},? \d{4}|\d{1,2} \w+ \d{4})\b',
    re.IGNORECASE
)
DURATION_PATTERN = re.compile(
    r'\b(\d+\s*(?:years?|months?|weeks?|days?|hours?|minutes?|decades?|centuries?))\b',
    re.IGNORECASE
)
RELATIVE_PATTERN = re.compile(
    r'\b(recently|today|yesterday|last\s+\w+|next\s+\w+|current|nowadays|now|present)\b',
    re.IGNORECASE
)
RANGE_PATTERN = re.compile(
    r'\b(from\s+\d{4}\s+to\s+\d{4}|\d{4}\s*[-–—]\s*\d{4})\b',
    re.IGNORECASE
)


def extract_temporal_expressions(text: str) -> List[TemporalExpression]:
    """
    Extract temporal expressions from text.
    
    Args:
        text: Input text to parse
        
    Returns:
        List of TemporalExpression objects
    """
    expressions = []
    
    # Extract years
    for match in YEAR_PATTERN.finditer(text):
        expressions.append(TemporalExpression(
            text=match.group(),
            type="year",
            normalized=match.group(),
            start_pos=match.start(),
            end_pos=match.end(),
        ))
    
    # Extract dates
    for match in DATE_PATTERN.finditer(text):
        expressions.append(TemporalExpression(
            text=match.group(),
            type="date",
            start_pos=match.start(),
            end_pos=match.end(),
        ))
    
    # Extract durations
    for match in DURATION_PATTERN.finditer(text):
        expressions.append(TemporalExpression(
            text=match.group(),
            type="duration",
            start_pos=match.start(),
            end_pos=match.end(),
        ))
    
    # Extract relative expressions
    for match in RELATIVE_PATTERN.finditer(text):
        expressions.append(TemporalExpression(
            text=match.group(),
            type="relative",
            start_pos=match.start(),
            end_pos=match.end(),
        ))
    
    # Extract ranges
    for match in RANGE_PATTERN.finditer(text):
        expressions.append(TemporalExpression(
            text=match.group(),
            type="range",
            start_pos=match.start(),
            end_pos=match.end(),
        ))
    
    # Sort by position and deduplicate overlaps
    expressions.sort(key=lambda x: x.start_pos)
    
    return expressions


def extract_years(text: str) -> List[str]:
    """Extract all years from text."""
    return YEAR_PATTERN.findall(text)


def has_temporal_content(text: str) -> bool:
    """Check if text contains any temporal expressions."""
    return bool(extract_temporal_expressions(text))
