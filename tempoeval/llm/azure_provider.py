"""Azure OpenAI LLM provider."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from tempoeval.llm.base import BaseLLMProvider, _is_retryable_exception, _sleep_backoff


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI API provider.
    
    Environment variables:
        AZURE_OPENAI_ENDPOINT: Azure endpoint URL
        AZURE_OPENAI_API_KEY: API key
        AZURE_DEPLOYMENT_NAME: Deployment name
        AZURE_OPENAI_API_VERSION: API version (default: 2025-01-01-preview)
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 6,
        temperature: float = 0.0,
    ):
        super().__init__(max_retries=max_retries, temperature=temperature)
        
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        if not all([self.endpoint, self.api_key, self.deployment_name]):
            raise ValueError(
                "Azure provider requires AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_DEPLOYMENT_NAME"
            )
        
        self._tls = threading.local()
    
    def _client(self):
        """Get thread-local client."""
        from openai import AzureOpenAI
        
        if not hasattr(self._tls, "client"):
            self._tls.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._tls.client
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response."""
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                resp = self._client().chat.completions.create(
                    model=self.deployment_name,
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
        
        raise RuntimeError(f"Azure OpenAI call failed: {last_err}")
    
    async def agenerate_json(self, prompt: str) -> Dict[str, Any]:
        """Async JSON generation."""
        from openai import AsyncAzureOpenAI
        
        client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=self.deployment_name,
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
        
        raise RuntimeError(f"Azure OpenAI async call failed: {last_err}")
    
    def generate(self, prompt: str) -> str:
        """Generate text response (non-JSON)."""
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                resp = self._client().chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    _sleep_backoff(attempt)
                    continue
                raise
        
        raise RuntimeError(f"Azure OpenAI call failed: {last_err}")
    
    async def agenerate(self, prompt: str) -> str:
        """Async text generation."""
        from openai import AsyncAzureOpenAI
        
        client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    import asyncio
                    await asyncio.sleep(0.8 * (2 ** attempt))
                    continue
                raise
        
        raise RuntimeError(f"Azure OpenAI async call failed: {last_err}")
