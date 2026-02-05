"""Main evaluation orchestrator for TempoEval."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

from tempoeval.core.base import BaseMetric
from tempoeval.core.config import TempoEvalConfig
from tempoeval.core.result import EvaluationResult, QueryResult
from tempoeval.efficiency.tracker import EfficiencyTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Main evaluation orchestrator.
    
    Handles metric computation for single queries or datasets,
    with support for parallel execution, async evaluation,
    and efficiency tracking.
    """
    
    def __init__(
        self,
        metrics: List[BaseMetric],
        config: Optional[TempoEvalConfig] = None,
        llm: Optional[Any] = None,
        efficiency_tracker: Optional[EfficiencyTracker] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute
            config: Evaluation configuration
            llm: LLM provider instance or name
            efficiency_tracker: Optional efficiency tracker
        """
        self.metrics = metrics
        self.config = config or TempoEvalConfig()
        self.llm = llm
        self.efficiency_tracker = efficiency_tracker
        
        # Inject LLM into metrics that need it
        self._setup_llm()
    
    def _setup_llm(self) -> None:
        """Set up LLM provider for metrics that require it."""
        if self.llm is None:
            return
        
        # If llm is a string, create provider
        if isinstance(self.llm, str):
            self.llm = self._create_llm_provider(self.llm)
        
        # Inject into metrics
        for metric in self.metrics:
            if metric.requires_llm:
                metric.llm = self.llm
    
    def _create_llm_provider(self, provider_name: str) -> Any:
        """Create LLM provider from name."""
        from tempoeval.llm import get_provider
        return get_provider(provider_name)
    
    def evaluate_query(
        self,
        query_id: str,
        query: str,
        k: int = 10,
        **kwargs
    ) -> QueryResult:
        """
        Evaluate a single query.
        
        Args:
            query_id: Unique query identifier
            query: Query text
            k: Cutoff for @K metrics
            **kwargs: Additional arguments passed to metrics
            
        Returns:
            QueryResult with scores for all metrics
        """
        scores = {}
        
        for metric in self.metrics:
            # Determine metric name (handle @K)
            if hasattr(metric, 'metric_type') and metric.metric_type == "retrieval":
                metric_name_k = f"{metric.name}@{k}"
            else:
                metric_name_k = metric.name
                
            # Start tracking if enabled
            op_id = f"{query_id}_{metric.name}"
            if self.efficiency_tracker:
                self.efficiency_tracker.start_tracking(op_id, metric.name)
            
            try:
                score = metric.compute(query=query, k=k, **kwargs)
                scores[metric_name_k] = score
                
            except Exception as e:
                logger.warning(f"Error computing {metric.name} for query {query_id}: {e}")
                scores[metric.name] = float('nan')
            
            finally:
                if self.efficiency_tracker:
                    # Estimate chars for cost (prompt approx = query len + some overhead)
                    # This is rough; for better accuracy we'd need metrics to return token usage
                    prompt_chars = len(query) + 500  # Rough overhead
                    completion_chars = 50  # Rough response assumption
                    self.efficiency_tracker.stop_tracking(
                        op_id, 
                        prompt_chars=prompt_chars if metric.requires_llm else 0,
                        completion_chars=completion_chars if metric.requires_llm else 0
                    )
        
        return QueryResult(query_id=query_id, query=query, scores=scores)
    
    async def aevaluate_query(
        self,
        query_id: str,
        query: str,
        k: int = 10,
        **kwargs
    ) -> QueryResult:
        """
        Asynchronously evaluate a single query.
        
        Args:
            query_id: Unique query identifier
            query: Query text
            k: Cutoff for @K metrics
            **kwargs: Additional arguments passed to metrics
            
        Returns:
            QueryResult with scores for all metrics
        """
        scores = {}
        
        # Run all metrics concurrently
        async def compute_metric(metric: BaseMetric):
            # Determine metric name
            if hasattr(metric, 'metric_type') and metric.metric_type == "retrieval":
                metric_name_k = f"{metric.name}@{k}"
            else:
                metric_name_k = metric.name
            
            op_id = f"{query_id}_{metric.name}"
            
            if self.efficiency_tracker:
               self.efficiency_tracker.start_tracking(op_id, metric.name)
               
            try:
                score = await metric.acompute(query=query, k=k, **kwargs)
                return metric_name_k, score
            except Exception as e:
                logger.warning(f"Error computing {metric.name} for query {query_id}: {e}")
                return metric.name, float('nan')
            finally:
                if self.efficiency_tracker:
                    prompt_chars = len(query) + 500
                    completion_chars = 50
                    self.efficiency_tracker.stop_tracking(
                        op_id, 
                        prompt_chars=prompt_chars if metric.requires_llm else 0,
                        completion_chars=completion_chars if metric.requires_llm else 0
                    )
        
        results = await asyncio.gather(*[compute_metric(m) for m in self.metrics])
        
        for metric_name, score in results:
            scores[metric_name] = score
        
        return QueryResult(query_id=query_id, query=query, scores=scores)
    
    def evaluate(
        self,
        data: Union[Dict, List[Dict]],
        k_values: Optional[List[int]] = None,
    ) -> EvaluationResult:
        """
        Evaluate on data (single query or batch).
        
        Args:
            data: Single query dict or list of query dicts
            k_values: List of K values for @K metrics
            
        Returns:
            EvaluationResult with all scores
        """
        k_values = k_values or self.config.k_values
        
        # Normalize to list
        if isinstance(data, dict):
            data = [data]
        
        result = EvaluationResult()
        
        # Use thread pool for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for item in data:
                query_id = item.get("id", item.get("query_id", str(len(futures))))
                query = item.get("query", "")
                
                for k in k_values:
                    futures.append(
                        executor.submit(
                            self.evaluate_query,
                            query_id=f"{query_id}@{k}",
                            query=query,
                            k=k,
                            **item
                        )
                    )
            
            # Collect results with progress bar
            iterator = as_completed(futures)
            if self.config.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Evaluating")
            
            for future in iterator:
                try:
                    qr = future.result()
                    result.add_query_result(qr)
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
        
        result.aggregate()
        
        # Attach efficiency stats if tracking
        if self.efficiency_tracker:
            result.metadata["efficiency"] = self.efficiency_tracker.summary()
            
        return result
    
    async def aevaluate(
        self,
        data: Union[Dict, List[Dict]],
        k_values: Optional[List[int]] = None,
    ) -> EvaluationResult:
        """
        Asynchronously evaluate on data.
        
        Args:
            data: Single query dict or list of query dicts
            k_values: List of K values for @K metrics
            
        Returns:
            EvaluationResult with all scores
        """
        k_values = k_values or self.config.k_values
        
        if isinstance(data, dict):
            data = [data]
        
        result = EvaluationResult()
        
        # Create all tasks
        tasks = []
        for item in data:
            query_id = item.get("id", item.get("query_id", str(len(tasks))))
            query = item.get("query", "")
            
            for k in k_values:
                tasks.append(
                    self.aevaluate_query(
                        query_id=f"{query_id}@{k}",
                        query=query,
                        k=k,
                        **item
                    )
                )
        
        # Run in batches
        batch_size = self.config.batch_size
        all_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for r in batch_results:
                if isinstance(r, Exception):
                    logger.error(f"Error in async evaluation: {r}")
                else:
                    all_results.append(r)
        
        for qr in all_results:
            result.add_query_result(qr)
        
        result.aggregate()
        
        if self.efficiency_tracker:
            result.metadata["efficiency"] = self.efficiency_tracker.summary()
            
        return result


def evaluate(
    data: Optional[Union[Dict, List[Dict]]] = None,
    dataset: Optional[Any] = None,
    metrics: Optional[List[BaseMetric]] = None,
    retrieval_results: Optional[Union[str, Dict]] = None,
    k_values: Optional[List[int]] = None,
    llm: Optional[Union[str, Any]] = None,
    efficiency_tracker: Optional[EfficiencyTracker] = None,
    max_workers: int = 10,
    show_progress: bool = True,
    **kwargs
) -> EvaluationResult:
    """
    Main evaluation function (synchronous).
    
    Args:
        data: Single query dict or list of query dicts
        dataset: TemporalDataset instance
        metrics: List of metrics to compute
        retrieval_results: Path to scores.json or dict of retrieval scores
        k_values: List of K values for @K metrics
        llm: LLM provider name or instance
        efficiency_tracker: Optional efficiency tracker
        max_workers: Number of parallel workers
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments
        
    Returns:
        EvaluationResult with all scores
    """
    if metrics is None:
        raise ValueError("Must provide metrics to compute")
    
    config = TempoEvalConfig(
        llm=llm or "openai",
        k_values=k_values or [5, 10, 20],
        max_workers=max_workers,
        show_progress=show_progress,
    )
    
    evaluator = Evaluator(
        metrics=metrics, 
        config=config, 
        llm=llm,
        efficiency_tracker=efficiency_tracker
    )
    
    # Handle dataset
    if dataset is not None:
        data = list(dataset)
    
    if data is None:
        raise ValueError("Must provide data or dataset")
    
    return evaluator.evaluate(data=data, k_values=k_values)


async def evaluate_async(
    data: Optional[Union[Dict, List[Dict]]] = None,
    dataset: Optional[Any] = None,
    metrics: Optional[List[BaseMetric]] = None,
    k_values: Optional[List[int]] = None,
    llm: Optional[Union[str, Any]] = None,
    efficiency_tracker: Optional[EfficiencyTracker] = None,
    batch_size: int = 20,
    show_progress: bool = True,
    **kwargs
) -> EvaluationResult:
    """
    Main evaluation function (asynchronous).
    
    Args:
        data: Single query dict or list of query dicts
        dataset: TemporalDataset instance
        metrics: List of metrics to compute
        k_values: List of K values for @K metrics
        llm: LLM provider name or instance
        efficiency_tracker: Optional efficiency tracker
        batch_size: Batch size for concurrent processing
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments
        
    Returns:
        EvaluationResult with all scores
    """
    if metrics is None:
        raise ValueError("Must provide metrics to compute")
    
    config = TempoEvalConfig(
        llm=llm or "openai",
        k_values=k_values or [5, 10, 20],
        batch_size=batch_size,
        show_progress=show_progress,
    )
    
    evaluator = Evaluator(
        metrics=metrics, 
        config=config, 
        llm=llm,
        efficiency_tracker=efficiency_tracker
    )
    
    if dataset is not None:
        data = list(dataset)
    
    if data is None:
        raise ValueError("Must provide data or dataset")
    
    return await evaluator.aevaluate(data=data, k_values=k_values)
