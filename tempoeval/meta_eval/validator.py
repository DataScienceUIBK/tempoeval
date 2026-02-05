"""Validator engine for meta-evaluation."""

import logging
from typing import Any, Dict, List, Optional
from tempoeval.meta_eval.base import BaseValidator, TestCase

logger = logging.getLogger(__name__)

class MetricValidator(BaseValidator):
    
    def validate(self, metric: Any, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Validate a metric against test cases.
        
        Args:
           metric: The instantiated metric object
           test_cases: List of TestCase objects
           
        Returns:
           Validation report
        """
        results = {
            "metric": metric.name,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for case in test_cases:
            try:
                # Run on original
                score_orig = metric.compute(**case.original_input)
                
                # Run on perturbed
                score_pert = metric.compute(**case.perturbed_input)
                
                # Check expectation
                passed = False
                if case.expected_result == "lower":
                    passed = score_pert < score_orig
                elif case.expected_result == "higher":
                    passed = score_pert > score_orig
                elif case.expected_result == "same":
                    passed = abs(score_pert - score_orig) < 1e-6
                    
                status = "PASS" if passed else "FAIL"
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append({
                    "type": case.perturbation_type,
                    "score_orig": score_orig,
                    "score_pert": score_pert,
                    "expected": case.expected_result,
                    "status": status
                })
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                results["failed"] += 1
                results["details"].append({
                    "error": str(e),
                    "status": "ERROR"
                })
                
        return results
