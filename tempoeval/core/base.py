"""Base classes for all TempoEval metrics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tempoeval.llm.base import BaseLLMProvider


@dataclass
class BaseMetric(ABC):
    """
    Abstract base class for all TempoEval metrics.
    
    Attributes:
        name: Unique identifier for the metric.
        requires_llm: Whether this metric needs an LLM provider.
        requires_gold: Whether this metric needs gold labels.
        lower_is_better: If True, lower scores are better (e.g., hallucination rate).
    """
    
    name: str = field(init=False)
    requires_llm: bool = field(init=False, default=False)
    requires_gold: bool = field(init=False, default=False)
    lower_is_better: bool = field(init=False, default=False)
    
    # LLM provider (set by evaluator if needed)
    llm: Optional["BaseLLMProvider"] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Set class-level attributes after init."""
        if not hasattr(self, 'name') or self.name is None:
            self.name = self.__class__.__name__.lower()
    
    @abstractmethod
    def compute(self, **kwargs) -> float:
        """
        Synchronous metric computation.
        
        Returns:
            float: Metric score, typically in [0, 1] range.
        """
        pass
    
    async def acompute(self, **kwargs) -> float:
        """
        Asynchronous metric computation. Defaults to sync version.
        Override for LLM-based metrics to enable async.
        
        Returns:
            float: Metric score.
        """
        return self.compute(**kwargs)
    
    def validate_inputs(self, **kwargs) -> None:
        """Validate required inputs are present."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass
class BaseRetrievalMetric(BaseMetric):
    """
    Base class for Layer 1 (Retrieval) metrics.
    
    Retrieval metrics evaluate the quality of document retrieval,
    specifically for temporal relevance.
    
    Common inputs:
        - retrieved_ids: List of retrieved document IDs
        - retrieved_docs: List of retrieved document texts
        - gold_ids: Gold-standard document IDs
        - query: The query text
        - temporal_focus: Type of temporal query (specific_time, duration, etc.)
        - k: Cutoff for @K metrics
    """
    
    metric_type: str = field(init=False, default="retrieval")
    
    def compute(
        self,
        retrieved_ids: Optional[List[str]] = None,
        gold_ids: Optional[List[str]] = None,
        k: int = 10,
        **kwargs
    ) -> float:
        """Default compute for retrieval metrics."""
        raise NotImplementedError("Subclasses must implement compute()")


@dataclass
class BaseGenerationMetric(BaseMetric):
    """
    Base class for Layer 2 (Generation) metrics.
    
    Generation metrics evaluate the quality of generated answers,
    specifically for temporal accuracy and faithfulness.
    
    Common inputs:
        - answer: Generated answer text
        - contexts: List of context/retrieved document texts
        - gold_answer: Gold-standard answer
        - query: The query text
    """
    
    metric_type: str = field(init=False, default="generation")
    
    def compute(
        self,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """Default compute for generation metrics."""
        raise NotImplementedError("Subclasses must implement compute()")


@dataclass
class BaseReasoningMetric(BaseMetric):
    """
    Base class for Layer 3 (Reasoning) metrics.
    
    Reasoning metrics evaluate temporal reasoning capabilities,
    such as event ordering, duration accuracy, and cross-period synthesis.
    
    Common inputs:
        - answer: Generated answer text
        - gold_events: Gold-standard event sequence
        - gold_durations: Gold-standard durations
        - query: The query text
    """
    
    metric_type: str = field(init=False, default="reasoning")
    
    def compute(
        self,
        answer: Optional[str] = None,
        **kwargs
    ) -> float:
        """Default compute for reasoning metrics."""
        raise NotImplementedError("Subclasses must implement compute()")


@dataclass
class BaseCompositeMetric(BaseMetric):
    """
    Base class for composite metrics that aggregate multiple metrics.
    
    Example: TempoScore combines recall, precision, faithfulness, coverage.
    """
    
    metric_type: str = field(init=False, default="composite")
    weights: Dict[str, float] = field(default_factory=dict)
    component_metrics: Dict[str, BaseMetric] = field(default_factory=dict, repr=False)
    
    def compute(self, **kwargs) -> Dict[str, float]:
        """Compute all component metrics and aggregate."""
        raise NotImplementedError("Subclasses must implement compute()")
    
    @staticmethod
    def harmonic_mean(scores: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Compute weighted harmonic mean of scores.
        
        Harmonic mean is sensitive to low values - if any component is 0,
        the result is 0. This encourages balanced performance.
        
        Args:
            scores: List of metric scores
            weights: Optional weights (defaults to equal)
            
        Returns:
            Weighted harmonic mean
        """
        if not scores:
            return float('nan')
        
        # Filter out NaN values
        valid = [(s, w) for s, w in zip(scores, weights or [1.0] * len(scores)) 
                 if not math.isnan(s) and s > 0]
        
        if not valid:
            return float('nan')
        
        total_weight = sum(w for _, w in valid)
        weighted_sum = sum(w / s for s, w in valid)
        
        if weighted_sum == 0:
            return 0.0
        
        return round(total_weight / weighted_sum, 5)
