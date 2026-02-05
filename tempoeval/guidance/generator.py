"""Temporal Guidance Generator.

This module generates temporal annotations for datasets that don't have
pre-existing guidance. It uses LLM-based analysis to:

1. Analyze queries to extract temporal intent, signals, and retrieval plans
2. Annotate passages with temporal information (signals, events, time scope)

This is useful when evaluating on custom datasets without TEMPO-style annotations.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tempoeval.guidance.prompts import (
    SYSTEM_PROMPT,
    QUERY_GUIDANCE_PROMPT,
    PASSAGE_ANNOTATION_PROMPT,
)


def _dedupe(seq: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in seq or []:
        x2 = (x or "").strip()
        if x2 and x2 not in seen:
            seen.add(x2)
            out.append(x2)
    return out


def _parse_json_loose(s: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code blocks."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to find JSON object in text
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(s[first:last+1])
    raise ValueError("Could not parse JSON from model output")


@dataclass
class QueryGuidance:
    """Generated guidance for a query."""
    is_temporal_query: bool = True
    temporal_intent: str = "none"
    query_temporal_signals: List[str] = field(default_factory=list)
    query_temporal_events: List[str] = field(default_factory=list)
    query_summary: str = ""
    temporal_reasoning_class_primary: str = "time_period_contextualization"
    temporal_reasoning_class_secondary: List[str] = field(default_factory=list)
    retrieval_reasoning: str = ""
    retrieval_plan: List[Dict[str, Any]] = field(default_factory=list)
    key_time_anchors: List[str] = field(default_factory=list)
    expected_granularity: str = "unknown"
    quality_checks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_temporal_query": self.is_temporal_query,
            "temporal_intent": self.temporal_intent,
            "query_temporal_signals": self.query_temporal_signals,
            "query_temporal_events": self.query_temporal_events,
            "query_summary": self.query_summary,
            "temporal_reasoning_class_primary": self.temporal_reasoning_class_primary,
            "temporal_reasoning_class_secondary": self.temporal_reasoning_class_secondary,
            "retrieval_reasoning": self.retrieval_reasoning,
            "retrieval_plan": self.retrieval_plan,
            "key_time_anchors": self.key_time_anchors,
            "expected_granularity": self.expected_granularity,
            "quality_checks": self.quality_checks,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QueryGuidance":
        """Create from dictionary."""
        return cls(
            is_temporal_query=bool(d.get("is_temporal_query", True)),
            temporal_intent=d.get("temporal_intent", "none"),
            query_temporal_signals=d.get("query_temporal_signals", []),
            query_temporal_events=d.get("query_temporal_events", []),
            query_summary=d.get("query_summary", ""),
            temporal_reasoning_class_primary=d.get("temporal_reasoning_class_primary", "time_period_contextualization"),
            temporal_reasoning_class_secondary=d.get("temporal_reasoning_class_secondary", []),
            retrieval_reasoning=d.get("retrieval_reasoning", ""),
            retrieval_plan=d.get("retrieval_plan", []),
            key_time_anchors=d.get("key_time_anchors", []),
            expected_granularity=d.get("expected_granularity", "unknown"),
            quality_checks=d.get("quality_checks", []),
        )


@dataclass  
class PassageAnnotation:
    """Generated temporal annotation for a passage."""
    doc_id: str = ""
    confidence: float = 0.0
    passage_temporal_signals: List[str] = field(default_factory=list)
    passage_temporal_events: List[str] = field(default_factory=list)
    time_mentions: List[str] = field(default_factory=list)
    time_scope_guess: Dict[str, str] = field(default_factory=lambda: {
        "start_iso": "",
        "end_iso": "",
        "granularity": "unknown"
    })
    tense_guess: str = "unknown"
    read_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "doc_id": self.doc_id,
            "confidence": self.confidence,
            "passage_temporal_signals": self.passage_temporal_signals,
            "passage_temporal_events": self.passage_temporal_events,
            "time_mentions": self.time_mentions,
            "time_scope_guess": self.time_scope_guess,
            "tense_guess": self.tense_guess,
        }
        if self.read_error:
            d["read_error"] = self.read_error
        return d


class TemporalGuidanceGenerator:
    """
    Generates temporal guidance for queries and passages using LLM.
    
    This is useful for datasets that don't have pre-existing temporal
    annotations like TEMPO. It can generate:
    
    1. Query-level guidance (temporal intent, signals, retrieval plan)
    2. Passage-level annotations (temporal signals, events, time scope)
    
    Example:
        >>> from tempoeval.llm import AzureOpenAIProvider
        >>> from tempoeval.guidance import TemporalGuidanceGenerator
        >>> 
        >>> llm = AzureOpenAIProvider()
        >>> generator = TemporalGuidanceGenerator(llm)
        >>> 
        >>> # Generate query guidance
        >>> guidance = generator.generate_query_guidance(
        ...     "When did World War I begin?"
        ... )
        >>> print(guidance.temporal_intent)  # "when"
        >>> print(guidance.key_time_anchors)  # ["1914"]
        >>> 
        >>> # Annotate passages
        >>> annotations = generator.annotate_passages(
        ...     query="When did World War I begin?",
        ...     passages=["WWI started in 1914...", "The war ended in 1918..."],
        ...     doc_ids=["doc1", "doc2"]
        ... )
    """
    
    def __init__(
        self,
        llm,
        max_retries: int = 2,
        concurrency: int = 10,
        max_passage_chars: int = 6000,
    ):
        """
        Initialize the guidance generator.
        
        Args:
            llm: LLM provider (must have generate() method)
            max_retries: Number of retries for API calls
            concurrency: Max parallel passage annotation requests
            max_passage_chars: Truncate passages to this length
        """
        self.llm = llm
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.max_passage_chars = max_passage_chars
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retries."""
        last_err = None
        for _ in range(max(1, self.max_retries + 1)):
            try:
                # Use generate_json if available, otherwise generate
                if hasattr(self.llm, 'generate_json'):
                    result = self.llm.generate_json(prompt)
                    return json.dumps(result)
                else:
                    return self.llm.generate(prompt)
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        raise last_err
    
    def generate_query_guidance(self, query: str) -> QueryGuidance:
        """
        Generate temporal guidance for a query.
        
        Args:
            query: The query text
            
        Returns:
            QueryGuidance object with temporal analysis
        """
        prompt = QUERY_GUIDANCE_PROMPT.format(query=query)
        
        try:
            raw = self._call_llm(prompt)
            parsed = _parse_json_loose(raw)
        except Exception as e:
            # Return default guidance on error
            return QueryGuidance(
                is_temporal_query=False,
                temporal_intent="none",
                query_summary=f"parse_error: {type(e).__name__}",
            )
        
        return QueryGuidance(
            is_temporal_query=bool(parsed.get("is_temporal_query", True)),
            temporal_intent=parsed.get("temporal_intent", "none"),
            query_temporal_signals=_dedupe(parsed.get("query_temporal_signals", [])),
            query_temporal_events=_dedupe(parsed.get("query_temporal_events", [])),
            query_summary=parsed.get("query_summary", ""),
            temporal_reasoning_class_primary=parsed.get("temporal_reasoning_class_primary", "time_period_contextualization"),
            temporal_reasoning_class_secondary=parsed.get("temporal_reasoning_class_secondary", []),
            retrieval_reasoning=parsed.get("retrieval_reasoning", ""),
            retrieval_plan=parsed.get("retrieval_plan", []),
            key_time_anchors=_dedupe(parsed.get("key_time_anchors", [])),
            expected_granularity=parsed.get("expected_granularity", "unknown"),
            quality_checks=parsed.get("quality_checks", []),
        )
    
    def annotate_passage(
        self,
        query: str,
        passage: str,
        doc_id: str = "",
    ) -> PassageAnnotation:
        """
        Annotate a single passage with temporal information.
        
        Args:
            query: The query text
            passage: The passage text
            doc_id: Document ID for the passage
            
        Returns:
            PassageAnnotation with temporal analysis
        """
        # Truncate passage if needed
        if self.max_passage_chars and len(passage) > self.max_passage_chars:
            passage = passage[:self.max_passage_chars]
        
        prompt = PASSAGE_ANNOTATION_PROMPT.format(query=query, passage=passage)
        
        try:
            raw = self._call_llm(prompt)
            parsed = _parse_json_loose(raw)
        except Exception:
            return PassageAnnotation(
                doc_id=doc_id,
                confidence=0.0,
            )
        
        tsg = parsed.get("time_scope_guess", {}) or {}
        
        return PassageAnnotation(
            doc_id=doc_id,
            confidence=float(parsed.get("confidence", 0.0)),
            passage_temporal_signals=_dedupe(parsed.get("passage_temporal_signals", [])),
            passage_temporal_events=_dedupe(parsed.get("passage_temporal_events", [])),
            time_mentions=_dedupe(parsed.get("time_mentions", [])),
            time_scope_guess={
                "start_iso": (tsg.get("start_iso") or "").strip(),
                "end_iso": (tsg.get("end_iso") or "").strip(),
                "granularity": (tsg.get("granularity") or "unknown").strip(),
            },
            tense_guess=parsed.get("tense_guess", "unknown"),
        )
    
    def annotate_passages(
        self,
        query: str,
        passages: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> List[PassageAnnotation]:
        """
        Annotate multiple passages in parallel.
        
        Args:
            query: The query text
            passages: List of passage texts
            doc_ids: Optional list of document IDs
            
        Returns:
            List of PassageAnnotation objects
        """
        if not passages:
            return []
        
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(passages))]
        
        results: Dict[int, PassageAnnotation] = {}
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {}
            for i, (passage, doc_id) in enumerate(zip(passages, doc_ids)):
                future = executor.submit(
                    self.annotate_passage, query, passage, doc_id
                )
                futures[future] = i
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = PassageAnnotation(doc_id=doc_ids[idx])
        
        # Return in original order
        return [results[i] for i in range(len(passages))]
    
    def generate_full_annotation(
        self,
        query_id: str,
        query: str,
        gold_passages: Optional[List[Tuple[str, str]]] = None,
        negative_passages: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete annotation for a query with passages.
        
        Args:
            query_id: Query identifier
            query: Query text
            gold_passages: List of (doc_id, passage_text) for gold passages
            negative_passages: List of (doc_id, passage_text) for negative passages
            
        Returns:
            Complete annotation dict matching TEMPO format
        """
        # Generate query guidance
        guidance = self.generate_query_guidance(query)
        
        # Annotate gold passages
        gold_annotations = []
        if gold_passages:
            gold_texts = [p[1] for p in gold_passages]
            gold_ids = [p[0] for p in gold_passages]
            gold_annotations = [
                a.to_dict() 
                for a in self.annotate_passages(query, gold_texts, gold_ids)
            ]
        
        # Annotate negative passages
        neg_annotations = []
        if negative_passages:
            neg_texts = [p[1] for p in negative_passages]
            neg_ids = [p[0] for p in negative_passages]
            neg_annotations = [
                a.to_dict()
                for a in self.annotate_passages(query, neg_texts, neg_ids)
            ]
        
        return {
            "id": query_id,
            "query": query,
            "query_guidance": guidance.to_dict(),
            "gold_passage_annotations": gold_annotations,
            "negative_passage_annotations": neg_annotations,
        }
