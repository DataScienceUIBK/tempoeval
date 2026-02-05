"""Benchmark reporter to generate leaderboards."""

import json
from typing import Dict
from tempoeval.core.result import EvaluationResult

class BenchmarkReporter:
    """Generates reports and leaderboards."""
    
    def generate_markdown_table(self, results: Dict[str, EvaluationResult]) -> str:
        """
        Generate a Markdown leaderboard table.
        
        Args:
            results: Dict of model_name -> EvaluationResult
            
        Returns:
            Markdown table string
        """
        if not results:
            return "No results to report."
            
        # Get all metric keys from first result
        first_res = next(iter(results.values()))
        metrics = sorted(first_res.summary().keys())
        
        # Header
        header = "| Model | " + " | ".join(metrics) + " | Efficiency (Lat/Cost) |"
        separator = "|---|" + "|".join(["---"] * len(metrics)) + "|---|"
        
        rows = []
        for model, res in results.items():
            scores = res.summary()
            row_vals = [model]
            
            for m in metrics:
                val = scores.get(m, float('nan'))
                row_vals.append(f"{val:.3f}")
            
            # Efficiency info if available
            eff_str = "-"
            if "efficiency" in res.metadata:
                eff = res.metadata["efficiency"]
                lat = eff.get("avg_latency_ms", 0)
                cost = eff.get("total_cost_usd", 0)
                eff_str = f"{lat:.0f}ms / ${cost:.4f}"
            row_vals.append(eff_str)
            
            rows.append("| " + " | ".join(row_vals) + " |")
            
        return "\n".join([header, separator] + rows)

    def save_report(self, results: Dict[str, EvaluationResult], path: str):
        """Save full report to JSON."""
        out = {}
        for model, res in results.items():
            out[model] = {
                "summary": res.summary(),
                "metadata": res.metadata
            }
        
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
