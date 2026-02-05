"""Anthropic Claude LLM provider."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, Optional

from tempoeval.llm.base import BaseLLMProvider, _sleep_backoff


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.
    
    Environment variables:
        ANTHROPIC_API_KEY: API key
        ANTHROPIC_MODEL: Model name (default: claude-3-5-sonnet-20240620)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 6,
        temperature: float = 0.0,
    ):
        super().__init__(max_retries=max_retries, temperature=temperature)
        
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("Please install 'anthropic' package: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        
        if not self.api_key:
            raise ValueError("Anthropic provider requires ANTHROPIC_API_KEY")
        
        self._tls = threading.local()
    
    def _client(self):
        """Get thread-local client."""
        if not hasattr(self._tls, "client"):
            self._tls.client = self._anthropic.Anthropic(api_key=self.api_key)
        return self._tls.client
    
    def _clean_json(self, text: str) -> str:
        """Extract JSON from Claude's verbose response."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response."""
        last_err = None
        full_prompt = prompt + "\n\nImportant: Respond ONLY with valid JSON."
        
        for attempt in range(self.max_retries):
            try:
                msg = self._client().messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                content = msg.content[0].text
                return json.loads(self._clean_json(content))
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    if isinstance(e, (
                        self._anthropic.RateLimitError,
                        self._anthropic.APIConnectionError,
                        self._anthropic.InternalServerError,
                    )):
                        _sleep_backoff(attempt)
                        continue
                raise
        
        raise RuntimeError(f"Anthropic call failed: {last_err}")
    
    async def agenerate_json(self, prompt: str) -> Dict[str, Any]:
        """Async JSON generation."""
        client = self._anthropic.AsyncAnthropic(api_key=self.api_key)
        last_err = None
        full_prompt = prompt + "\n\nImportant: Respond ONLY with valid JSON."
        
        for attempt in range(self.max_retries):
            try:
                msg = await client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                content = msg.content[0].text
                return json.loads(self._clean_json(content))
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    if isinstance(e, (
                        self._anthropic.RateLimitError,
                        self._anthropic.APIConnectionError,
                        self._anthropic.InternalServerError,
                    )):
                        import asyncio
                        await asyncio.sleep(0.8 * (2 ** attempt))
                        continue
                raise
        
        raise RuntimeError(f"Anthropic async call failed: {last_err}")
