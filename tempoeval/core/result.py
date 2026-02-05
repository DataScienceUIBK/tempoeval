"""Result containers for TempoEval evaluations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    
    query_id: str
    query: str
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_score(self, metric_name: str) -> float:
        """Get score for a specific metric."""
        return self.scores.get(metric_name, float('nan'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "scores": self.scores,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """
    Container for evaluation results across multiple queries.
    
    Provides aggregation, export, and analysis methods.
    """
    
    query_results: List[QueryResult] = field(default_factory=list)
    aggregated_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.query_results)
    
    def get_score(self, metric_name: str) -> float:
        """Get aggregated score for a metric."""
        return self.aggregated_scores.get(metric_name, float('nan'))
    
    def get_scores(self) -> Dict[str, float]:
        """Get all aggregated scores."""
        return self.aggregated_scores.copy()
    
    def mean(self, metric_name: Optional[str] = None) -> Dict[str, float]:
        """
        Compute mean scores across all queries.
        
        Args:
            metric_name: Specific metric to compute mean for (None = all)
            
        Returns:
            Dictionary of mean scores
        """
        if not self.query_results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for qr in self.query_results:
            all_metrics.update(qr.scores.keys())
        
        if metric_name:
            all_metrics = {metric_name} if metric_name in all_metrics else set()
        
        means = {}
        for m in all_metrics:
            values = [qr.scores.get(m) for qr in self.query_results 
                      if m in qr.scores and not math.isnan(qr.scores.get(m, float('nan')))]
            if values:
                means[m] = round(sum(values) / len(values), 5)
        
        return means
    
    def std(self, metric_name: Optional[str] = None) -> Dict[str, float]:
        """
        Compute standard deviation of scores across all queries.
        
        Args:
            metric_name: Specific metric to compute std for (None = all)
            
        Returns:
            Dictionary of standard deviations
        """
        if not self.query_results:
            return {}
        
        means = self.mean(metric_name)
        stds = {}
        
        for m, mean_val in means.items():
            values = [qr.scores.get(m) for qr in self.query_results 
                      if m in qr.scores and not math.isnan(qr.scores.get(m, float('nan')))]
            if len(values) > 1:
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                stds[m] = round(math.sqrt(variance), 5)
            else:
                stds[m] = 0.0
        
        return stds
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of results.
        
        Returns:
            Formatted summary string
        """
        lines = ["TempoEval Results", "=" * 40]
        
        means = self.mean()
        stds = self.std()
        
        # Sort metrics by name
        for metric in sorted(means.keys()):
            mean_val = means[metric]
            std_val = stds.get(metric, 0.0)
            lines.append(f"{metric:30s}: {mean_val:.3f} ± {std_val:.3f}")
        
        lines.append("")
        lines.append(f"Evaluated {len(self.query_results)} queries")
        
        if "domains" in self.metadata:
            lines.append(f"Domains: {', '.join(self.metadata['domains'])}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_results": [qr.to_dict() for qr in self.query_results],
            "aggregated_scores": self.aggregated_scores,
            "metadata": self.metadata,
        }
    
    def to_json(self, path: str, indent: int = 2) -> None:
        """
        Save results to JSON file.
        
        Args:
            path: Output file path
            indent: JSON indentation
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    def to_csv(self, path: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            path: Output file path
        """
        try:
            import pandas as pd
            df = self.to_pandas()
            df.to_csv(path, index=False)
        except ImportError:
            # Fallback without pandas
            import csv
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            if not self.query_results:
                return
            
            # Get all metric names
            all_metrics = set()
            for qr in self.query_results:
                all_metrics.update(qr.scores.keys())
            
            fieldnames = ["query_id", "query"] + sorted(all_metrics)
            
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for qr in self.query_results:
                    row = {"query_id": qr.query_id, "query": qr.query}
                    row.update(qr.scores)
                    writer.writerow(row)
    
    def to_pandas(self) -> "pd.DataFrame":
        """
        Convert to pandas DataFrame.
        
        Returns:
            DataFrame with query-level results
        """
        import pandas as pd
        
        rows = []
        for qr in self.query_results:
            row = {"query_id": qr.query_id, "query": qr.query}
            row.update(qr.scores)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_query_results(self, query_id: str) -> Optional[QueryResult]:
        """Get results for a specific query."""
        for qr in self.query_results:
            if qr.query_id == query_id:
                return qr
        return None
    
    def iter_queries(self) -> Iterator[Tuple[str, Dict[str, float]]]:
        """Iterate over query results."""
        for qr in self.query_results:
            yield qr.query_id, qr.scores
    
    def add_query_result(self, result: QueryResult) -> None:
        """Add a query result."""
        self.query_results.append(result)
    
    def aggregate(self) -> None:
        """Compute and store aggregated scores."""
        self.aggregated_scores = self.mean()
    
    def __repr__(self) -> str:
        return f"EvaluationResult(queries={len(self)}, metrics={list(self.aggregated_scores.keys())})"
