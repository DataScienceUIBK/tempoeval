"""Benchmark runner."""

import logging
from typing import Any, Dict, List, Optional
from tempoeval.core.evaluator import evaluate
from tempoeval.core.result import EvaluationResult

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Runs standardized benchmarks."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        
    def run_suite(
        self,
        suite_name: str,
        models: List[str],
        dataset: Any,
        metrics: List[Any],
        k_values: List[int] = [10]
    ) -> Dict[str, EvaluationResult]:
        """
        Run a benchmark suite across multiple models.
        
        Args:
            suite_name: Name of the benchmark suite
            models: List of model names/providers
            dataset: Dataset to evaluate on
            metrics: Metrics to compute
            k_values: K values
            
        Returns:
            Dict of model_name -> EvaluationResult
        """
        results = {}
        
        for model in models:
            logger.info(f"Running suite '{suite_name}' on model: {model}")
            try:
                # Initialize specific LLM provider or retriever if needed
                # For now assume 'model' string is sufficient for Evaluator
                
                res = evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=model,
                    k_values=k_values,
                    show_progress=True
                )
                results[model] = res
                
            except Exception as e:
                logger.error(f"Failed to run benchmark for {model}: {e}")
                
        return results
