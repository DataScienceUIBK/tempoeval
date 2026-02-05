"""OpenAI LLM provider."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from tempoeval.llm.base import BaseLLMProvider, _is_retryable_exception, _sleep_backoff


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider.
    
    Environment variables:
        OPENAI_API_KEY: API key
        OPENAI_MODEL: Model name (default: gpt-4o)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 6,
        temperature: float = 0.0,
    ):
        super().__init__(max_retries=max_retries, temperature=temperature)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        
        if not self.api_key:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY")
        
        self._tls = threading.local()
    
    def _client(self):
        """Get thread-local client."""
        from openai import OpenAI
        
        if not hasattr(self._tls, "client"):
            self._tls.client = OpenAI(api_key=self.api_key)
        return self._tls.client
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response."""
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                resp = self._client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    _sleep_backoff(attempt)
                    continue
                raise
        
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}")
    
    async def agenerate_json(self, prompt: str) -> Dict[str, Any]:
        """Async JSON generation."""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.api_key)
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    import asyncio
                    await asyncio.sleep(0.8 * (2 ** attempt))
                    continue
                raise
        
        raise RuntimeError(f"OpenAI async call failed: {last_err}")
