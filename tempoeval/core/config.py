"""Configuration management for TempoEval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TempoEvalConfig:
    """
    Configuration for TempoEval evaluation.
    
    Attributes:
        llm: LLM provider name or instance ("openai", "azure", "anthropic", "gemini", "ollama")
        k_values: List of K values for @K metrics (default: [5, 10, 20])
        max_workers: Number of parallel workers for evaluation
        batch_size: Batch size for async evaluation
        cache_dir: Directory for caching LLM responses
        show_progress: Whether to show progress bars
        temperature: LLM temperature (0.0 for deterministic)
        max_retries: Maximum retries for LLM calls
    """
    
    # LLM configuration
    llm: str = "openai"
    temperature: float = 0.0
    max_retries: int = 6
    
    # Evaluation settings
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    max_workers: int = 10
    batch_size: int = 20
    
    # Caching
    cache_dir: Optional[str] = ".tempoeval_cache"
    use_cache: bool = True
    
    # Output
    show_progress: bool = True
    output_dir: Optional[str] = None
    output_format: str = "json"  # json, csv, pandas
    
    # Advanced
    timeout_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "llm": self.llm,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "k_values": self.k_values,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "cache_dir": self.cache_dir,
            "use_cache": self.use_cache,
            "show_progress": self.show_progress,
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TempoEvalConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
