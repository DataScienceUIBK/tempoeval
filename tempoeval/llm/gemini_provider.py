"""Google Gemini LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from tempoeval.llm.base import BaseLLMProvider, _sleep_backoff


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider.
    
    Environment variables:
        GOOGLE_API_KEY: API key
        GEMINI_MODEL: Model name (default: gemini-1.5-pro)
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
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError(
                "Please install 'google-generativeai' package: "
                "pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        
        if not self.api_key:
            raise ValueError("Gemini provider requires GOOGLE_API_KEY")
        
        # Configure and create model
        self._genai.configure(api_key=self.api_key)
        self.model = self._genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"}
        )
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response."""
        from google.api_core import exceptions
        
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                return json.loads(response.text)
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    if isinstance(e, (
                        exceptions.ResourceExhausted,
                        exceptions.ServiceUnavailable,
                        exceptions.InternalServerError,
                    )):
                        _sleep_backoff(attempt)
                        continue
                raise
        
        raise RuntimeError(f"Gemini call failed: {last_err}")
    
    async def agenerate_json(self, prompt: str) -> Dict[str, Any]:
        """Async JSON generation."""
        from google.api_core import exceptions
        
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.model.generate_content_async(prompt)
                return json.loads(response.text)
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    if isinstance(e, (
                        exceptions.ResourceExhausted,
                        exceptions.ServiceUnavailable,
                        exceptions.InternalServerError,
                    )):
                        import asyncio
                        await asyncio.sleep(0.8 * (2 ** attempt))
                        continue
                raise
        
        raise RuntimeError(f"Gemini async call failed: {last_err}")
