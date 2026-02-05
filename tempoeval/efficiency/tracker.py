"""Efficiency tracking module."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    REASONING = "reasoning"
    OTHER = "other"

@dataclass
class EfficiencyStats:
    """Statistics for a single operation."""
    operation_id: str
    metric_name: str
    start_time: float
    end_time: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    def finalize(self):
        """Finalize stats after operation completes."""
        if self.end_time == 0.0:
            self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000.0
        self.total_tokens = self.prompt_tokens + self.completion_tokens

@dataclass
class AggregatedStats:
    """Aggregated statistics."""
    count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

class EfficiencyTracker:
    """
    Tracks efficiency metrics (latency, tokens, cost) for evaluation runs.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.active_tracks: Dict[str, EfficiencyStats] = {}
        self.completed_stats: List[EfficiencyStats] = []
        self.cost_per_1k_input = 0.0025  # Default to GPT-4o approx
        self.cost_per_1k_output = 0.01
        
        # Adjust costs based on model name (simplified)
        if "gpt-3.5" in model_name:
            self.cost_per_1k_input = 0.0005
            self.cost_per_1k_output = 0.0015
        elif "claude-3-opus" in model_name:
            self.cost_per_1k_input = 0.015
            self.cost_per_1k_output = 0.075
            
    def start_tracking(self, operation_id: str, metric_name: str) -> None:
        """Start tracking an operation."""
        self.active_tracks[operation_id] = EfficiencyStats(
            operation_id=operation_id,
            metric_name=metric_name,
            start_time=time.time()
        )
        
    def stop_tracking(
        self, 
        operation_id: str, 
        prompt_chars: int = 0, 
        completion_chars: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> Optional[EfficiencyStats]:
        """Stop tracking an operation and record stats."""
        if operation_id not in self.active_tracks:
            return None
            
        stats = self.active_tracks.pop(operation_id)
        stats.end_time = time.time()
        
        # Estimate tokens if not provided (approx 4 chars per token)
        if prompt_tokens == 0 and prompt_chars > 0:
            prompt_tokens = prompt_chars // 4
        if completion_tokens == 0 and completion_chars > 0:
            completion_tokens = completion_chars // 4
            
        stats.prompt_tokens = prompt_tokens
        stats.completion_tokens = completion_tokens
        stats.finalize()
        
        # Calculate cost
        input_cost = (stats.prompt_tokens / 1000) * self.cost_per_1k_input
        output_cost = (stats.completion_tokens / 1000) * self.cost_per_1k_output
        stats.estimated_cost_usd = input_cost + output_cost
        
        self.completed_stats.append(stats)
        return stats

    def get_aggregated_stats(self) -> Dict[str, AggregatedStats]:
        """Get statistics aggregated by metric name."""
        agg: Dict[str, AggregatedStats] = {}
        
        for s in self.completed_stats:
            if s.metric_name not in agg:
                agg[s.metric_name] = AggregatedStats(min_latency_ms=float('inf'))
            
            a = agg[s.metric_name]
            a.count += 1
            a.total_latency_ms += s.latency_ms
            a.total_tokens += s.total_tokens
            a.total_cost_usd += s.estimated_cost_usd
            
            if s.latency_ms < a.min_latency_ms:
                a.min_latency_ms = s.latency_ms
            if s.latency_ms > a.max_latency_ms:
                a.max_latency_ms = s.latency_ms
                
        # Final calculations
        for a in agg.values():
            if a.count > 0:
                a.avg_latency_ms = a.total_latency_ms / a.count
                if a.min_latency_ms == float('inf'):
                    a.min_latency_ms = 0.0
                    
        return agg

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary."""
        total_cost = sum(s.estimated_cost_usd for s in self.completed_stats)
        total_latency = sum(s.latency_ms for s in self.completed_stats)
        total_ops = len(self.completed_stats)
        
        return {
            "total_operations": total_ops,
            "total_cost_usd": round(total_cost, 4),
            "total_latency_seconds": round(total_latency / 1000, 2),
            "avg_latency_ms": round(total_latency / total_ops, 2) if total_ops > 0 else 0,
            "metrics_breakdown": {k: vars(v) for k, v in self.get_aggregated_stats().items()}
        }
