"""LiteLLM unified provider for TempoEval.

LiteLLM provides a unified interface to 100+ LLM providers including:
- OpenAI, Azure, Anthropic, Google, Cohere
- Replicate, Together AI, Hugging Face
- Local models via Ollama, vLLM, etc.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from tempoeval.llm.base import BaseLLMProvider, _is_retryable_exception, _sleep_backoff


class LiteLLMProvider(BaseLLMProvider):
    """
    LiteLLM unified provider supporting 100+ LLM backends.
    
    LiteLLM simplifies LLM integration by providing a single interface
    that works with OpenAI, Anthropic, Azure, Google, local models, and more.
    
    Environment variables (depends on the model):
        OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, etc.
    
    Example models:
        - "gpt-4o" (OpenAI)
        - "claude-3-5-sonnet-20240620" (Anthropic)
        - "azure/gpt-4" (Azure)
        - "gemini/gemini-1.5-pro" (Google)
        - "ollama/llama2" (Local via Ollama)
        - "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1" (Together AI)
        - "huggingface/meta-llama/Llama-2-7b-chat-hf" (Hugging Face)
    
    Installation:
        pip install litellm
    
    Example:
        >>> from tempoeval.llm import LiteLLMProvider
        >>> llm = LiteLLMProvider(model="gpt-4o")
        >>> result = llm.generate_json("Return JSON: {\"test\": true}")
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 6,
        temperature: float = 0.0,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize LiteLLM provider.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20240620")
            api_key: API key (if not set via environment)
            api_base: Custom API base URL
            max_retries: Maximum retry attempts
            temperature: Generation temperature
            timeout: Request timeout in seconds
            **kwargs: Additional parameters passed to litellm.completion()
        """
        super().__init__(max_retries=max_retries, temperature=temperature)
        
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "Please install 'litellm' package: pip install litellm"
            )
        
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.extra_kwargs = kwargs
        
        # Set API key if provided
        if api_key:
            # LiteLLM uses environment variables, but we can set them
            if "gpt" in model.lower() or "openai" in model.lower():
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif "claude" in model.lower() or "anthropic" in model.lower():
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif "gemini" in model.lower():
                os.environ.setdefault("GEMINI_API_KEY", api_key)
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response using LiteLLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Parsed JSON dictionary
        """
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                response = self._litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    **self.extra_kwargs
                )
                
                content = response.choices[0].message.content
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                # Try to extract JSON from response
                return self._parse_json(content)
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    _sleep_backoff(attempt)
                    continue
                raise
        
        raise RuntimeError(f"LiteLLM call failed after {self.max_retries} retries: {last_err}")
    
    async def agenerate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Async JSON generation using LiteLLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Parsed JSON dictionary
        """
        import asyncio
        
        last_err = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self._litellm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    **self.extra_kwargs
                )
                
                content = response.choices[0].message.content
                return json.loads(content)
                
            except json.JSONDecodeError:
                return self._parse_json(content)
                
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    await asyncio.sleep(0.8 * (2 ** attempt))
                    continue
                raise
        
        raise RuntimeError(f"LiteLLM async call failed: {last_err}")
    
    def generate(self, prompt: str) -> str:
        """Generate text response."""
        response = self._litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            timeout=self.timeout,
            api_key=self.api_key,
            api_base=self.api_base,
            **self.extra_kwargs
        )
        return response.choices[0].message.content
    
    async def agenerate(self, prompt: str) -> str:
        """Async text generation."""
        response = await self._litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            timeout=self.timeout,
            api_key=self.api_key,
            api_base=self.api_base,
            **self.extra_kwargs
        )
        return response.choices[0].message.content
    
    @classmethod
    def list_supported_models(cls) -> list:
        """List some commonly used model identifiers."""
        return [
            # OpenAI
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            # Anthropic
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            # Google
            "gemini/gemini-1.5-pro",
            "gemini/gemini-1.5-flash",
            # Azure (prefix with azure/)
            "azure/gpt-4",
            "azure/gpt-35-turbo",
            # Local (Ollama)
            "ollama/llama2",
            "ollama/mistral",
            "ollama/codellama",
            # Together AI
            "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            # Replicate
            "replicate/meta/llama-2-70b-chat",
        ]
