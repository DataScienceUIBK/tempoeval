"""Base dataset classes for TempoEval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class TemporalQuery:
    """
    Schema for a single temporal query with associated data.
    
    Attributes:
        id: Unique query identifier
        query: Query text
        gold_ids: Gold-standard document IDs
        temporal_focus: Type of temporal query
        key_time_anchors: Important time anchors
        is_cross_period: Whether this is a cross-period query
        metadata: Additional metadata
    """
    
    id: str
    query: str
    gold_ids: List[str] = field(default_factory=list)
    temporal_focus: str = "specific_time"
    key_time_anchors: List[str] = field(default_factory=list)
    baseline_anchor: str = ""
    comparison_anchor: str = ""
    is_cross_period: bool = False
    gold_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for evaluation."""
        return {
            "id": self.id,
            "query": self.query,
            "gold_ids": self.gold_ids,
            "temporal_focus": self.temporal_focus,
            "key_time_anchors": self.key_time_anchors,
            "baseline_anchor": self.baseline_anchor,
            "comparison_anchor": self.comparison_anchor,
            "is_cross_period": self.is_cross_period,
            "gold_answer": self.gold_answer,
            **self.metadata,
        }


class TemporalDataset:
    """
    Container for temporal evaluation datasets.
    
    Provides iteration, filtering, and export capabilities.
    """
    
    def __init__(
        self,
        queries: List[TemporalQuery],
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.queries = queries
        self.name = name
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate as dicts (for evaluation)."""
        for q in self.queries:
            yield q.to_dict()
    
    def __getitem__(self, idx: int) -> TemporalQuery:
        return self.queries[idx]
    
    def filter(self, **kwargs) -> "TemporalDataset":
        """
        Filter queries by attribute values.
        
        Example:
            >>> dataset.filter(temporal_focus="change_over_time")
        """
        filtered = []
        for q in self.queries:
            match = True
            for key, value in kwargs.items():
                if getattr(q, key, None) != value:
                    match = False
                    break
            if match:
                filtered.append(q)
        
        return TemporalDataset(
            queries=filtered,
            name=f"{self.name}_filtered",
            metadata=self.metadata,
        )
    
    def cross_period_only(self) -> "TemporalDataset":
        """Get only cross-period queries."""
        return self.filter(is_cross_period=True)
    
    def sample(self, n: int, seed: int = 42) -> "TemporalDataset":
        """Random sample of queries."""
        import random
        random.seed(seed)
        sampled = random.sample(self.queries, min(n, len(self.queries)))
        return TemporalDataset(
            queries=sampled,
            name=f"{self.name}_sample_{n}",
            metadata=self.metadata,
        )
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts."""
        return [q.to_dict() for q in self.queries]
    
    def to_pandas(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.to_list())
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]], name: str = "") -> "TemporalDataset":
        """Create from list of dicts."""
        queries = []
        for item in data:
            q = TemporalQuery(
                id=str(item.get("id", len(queries))),
                query=item.get("query", ""),
                gold_ids=item.get("gold_ids", []),
                temporal_focus=item.get("temporal_focus", "specific_time"),
                key_time_anchors=item.get("key_time_anchors", []),
                baseline_anchor=item.get("baseline_anchor", ""),
                comparison_anchor=item.get("comparison_anchor", ""),
                is_cross_period=item.get("is_cross_period", False),
                gold_answer=item.get("gold_answer"),
                metadata={k: v for k, v in item.items() if k not in [
                    "id", "query", "gold_ids", "temporal_focus", "key_time_anchors",
                    "baseline_anchor", "comparison_anchor", "is_cross_period", "gold_answer"
                ]},
            )
            queries.append(q)
        
        return cls(queries=queries, name=name)
    
    @classmethod
    def from_pandas(cls, df: "pd.DataFrame", name: str = "") -> "TemporalDataset":
        """Create from pandas DataFrame."""
        return cls.from_list(df.to_dict("records"), name=name)
    
    @classmethod
    def from_json(cls, path: str, name: str = "") -> "TemporalDataset":
        """Load from JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_list(data, name=name or path)
    
    def __repr__(self) -> str:
        return f"TemporalDataset(name='{self.name}', queries={len(self)})"
