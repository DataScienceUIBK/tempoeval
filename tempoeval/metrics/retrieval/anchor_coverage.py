"""Anchor Coverage metric with dual-mode (Rule-based or LLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from tempoeval.core.base import BaseRetrievalMetric


ANCHOR_EXTRACTION_PROMPT = """## Task
Extract the CENTRAL temporal focus of the documents.

## Documents
{documents}

## Instructions
Identify ONLY the time periods that are fundamentally important to the document's main topic.
Do NOT include:
- Publication dates, copyright years, or citation dates
- Incidental mentions (e.g., "born in 1950" if the article is about their work in 1990)
- Relative dates without context (e.g., "last year")

Focus on:
- The main era/period being discussed
- Key event dates
- Explicit time scopes (e.g., "study from 2010 to 2015")

## Output (JSON)
{{
    "extracted_anchors": ["1999", "the 1990s", "Victorian era"],
    "anchor_count": 3
}}
"""


@dataclass
class AnchorCoverage(BaseRetrievalMetric):
    """
    Anchor Coverage measures the F1 score between time anchors found
    in retrieved documents and the time anchors required by the query.
    
    Supports two modes:
    - Rule-based (default): Uses regex to extract years (fast, free)
    - LLM-based: Uses LLM to extract semantic time references (more accurate)
    
    Properties:
        - Range: [0, 1]
        - Higher is better
        - Measures precision AND recall of time anchors
    
    Example:
        >>> metric = AnchorCoverage(use_llm=False)  # Rule-based
        >>> score = metric.compute(...)
        
        >>> metric = AnchorCoverage(use_llm=True)  # LLM-based
        >>> metric.llm = llm_provider
        >>> score = metric.compute(...)
    """
    
    name: str = field(init=False, default="anchor_coverage")
    requires_gold: bool = field(init=False, default=True)
    use_llm: bool = False  # User can set this to True for LLM mode
    
    @property
    def requires_llm(self) -> bool:
        """Dynamic LLM requirement based on mode."""
        return self.use_llm
    
    def compute(
        self,
        retrieved_anchors: Optional[List[str]] = None,
        required_anchors: Optional[List[str]] = None,
        retrieved_docs: Optional[List[str]] = None,
        key_time_anchors: Optional[List[str]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """
        Compute Anchor Coverage F1.
        
        Args:
            retrieved_anchors: Time anchors from retrieved docs (or extracted)
            required_anchors: Required time anchors (gold)
            retrieved_docs: Alternative - docs to extract anchors from
            key_time_anchors: Alternative name for required_anchors
            k: Cutoff for top-K
            
        Returns:
            F1 score in [0, 1]
        """
        # Handle alternative argument names
        if required_anchors is None:
            required_anchors = key_time_anchors
        
        if required_anchors is None or len(required_anchors) == 0:
            return 0.0
        
        # Extract anchors from docs if not provided directly
        if retrieved_anchors is None and retrieved_docs:
            if self.use_llm:
                retrieved_anchors = self._extract_anchors_llm(retrieved_docs[:k])
            else:
                retrieved_anchors = self._extract_anchors_regex(retrieved_docs[:k])
        
        if retrieved_anchors is None or len(retrieved_anchors) == 0:
            return 0.0
        
        # Convert to sets (normalize to lowercase for comparison)
        retrieved_set = set(str(a).lower() for a in retrieved_anchors)
        required_set = set(str(a).lower() for a in required_anchors)
        
        # Calculate intersection
        intersection = len(retrieved_set & required_set)
        
        # Calculate precision and recall
        precision = intersection / len(retrieved_set) if retrieved_set else 0.0
        recall = intersection / len(required_set) if required_set else 0.0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 5)
    
    def _extract_anchors_regex(self, docs: List[str]) -> List[str]:
        """
        Extract time anchors using regex patterns (fast, rule-based).
        """
        import re
        
        anchors: Set[str] = set()
        
        # Pattern for years (1900-2099)
        year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')
        # Pattern for decades (the 1990s, 80s)
        decade_pattern = re.compile(r'\b(the\s+)?(19|20)?\d0s\b', re.IGNORECASE)
        
        for doc in docs:
            years = year_pattern.findall(doc)
            anchors.update(years)
            decades = decade_pattern.findall(doc)
            anchors.update([''.join(d) for d in decades])
        
        return list(anchors)
    
    def _extract_anchors_llm(self, docs: List[str]) -> List[str]:
        """
        Extract time anchors using LLM (more accurate, handles semantic references).
        """
        if self.llm is None:
            raise ValueError("LLM mode requires an LLM provider (set metric.llm)")
        
        docs_text = "\n\n---\n\n".join([f"[Doc {i+1}]: {d[:2000]}" for i, d in enumerate(docs)])
        
        try:
            result = self.llm.generate_json(
                ANCHOR_EXTRACTION_PROMPT.format(documents=docs_text)
            )
            return result.get("extracted_anchors", [])
        except Exception:
            # Fallback to regex if LLM fails
            return self._extract_anchors_regex(docs)
