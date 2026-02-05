"""Base LLM provider class."""

from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Retryable HTTP status codes
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _sleep_backoff(attempt: int, base: float = 0.8, cap: float = 10.0) -> None:
    """Exponential backoff with jitter."""
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.7 + 0.6 * random.random())
    time.sleep(delay)


def _is_retryable_exception(e: Exception) -> bool:
    """Check if exception is retryable."""
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    if status is not None:
        return int(status) in _RETRYABLE_STATUS
    msg = str(e).lower()
    return any(x in msg for x in [
        "rate limit", "timeout", "temporarily", 
        "overloaded", "try again", "connection"
    ])


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Provides common interface for generating JSON responses
    with retry logic and error handling.
    """
    
    def __init__(self, max_retries: int = 6, temperature: float = 0.0):
        self.max_retries = max_retries
        self.temperature = temperature
    
    @abstractmethod
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response from prompt (synchronous).
        
        Args:
            prompt: Input prompt
            
        Returns:
            Parsed JSON dictionary
        """
        pass
    
    async def agenerate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response from prompt (asynchronous).
        
        Default implementation wraps sync method.
        Override for true async support.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Parsed JSON dictionary
        """
        return self.generate_json(prompt)
    
    def generate(self, prompt: str) -> str:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclass must implement generate()")
    
    async def agenerate(self, prompt: str) -> str:
        """
        Generate text response from prompt (async).
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        return self.generate(prompt)
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from text, handling common issues.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed dictionary
        """
        import re
        
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Return empty dict as fallback
        logger.warning(f"Failed to parse JSON from: {text[:200]}...")
        return {}
