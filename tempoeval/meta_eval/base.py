"""Base classes for meta-evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TestCase:
    """A single test case for meta-evaluation."""
    original_input: Dict[str, Any]
    perturbed_input: Dict[str, Any]
    perturbation_type: str
    expected_result: str  # "lower", "higher", "same"

class BasePerturbator(ABC):
    """Base class for creating synthetic failure modes."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def perturb(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply perturbation to input data."""
        pass

class BaseValidator(ABC):
    """Base class for validating metrics."""
    
    @abstractmethod
    def validate(self, metric: Any, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run validation on test cases."""
        pass
