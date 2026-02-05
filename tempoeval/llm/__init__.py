"""LLM providers for TempoEval."""

from tempoeval.llm.base import BaseLLMProvider
from tempoeval.llm.openai_provider import OpenAIProvider
from tempoeval.llm.azure_provider import AzureOpenAIProvider
from tempoeval.llm.anthropic_provider import AnthropicProvider
from tempoeval.llm.gemini_provider import GeminiProvider
from tempoeval.llm.litellm_provider import LiteLLMProvider

# Provider registry
_PROVIDERS = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "litellm": LiteLLMProvider,
}


def get_provider(name: str, **kwargs) -> BaseLLMProvider:
    """
    Get LLM provider by name.
    
    Args:
        name: Provider name ("openai", "azure", "anthropic", "gemini", "litellm")
        **kwargs: Provider-specific configuration
        
    Returns:
        Initialized provider instance
    """
    name = name.lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS.keys())}")
    
    return _PROVIDERS[name](**kwargs)


def register_provider(name: str, provider_class: type) -> None:
    """Register a custom provider."""
    _PROVIDERS[name.lower()] = provider_class


__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "LiteLLMProvider",
    "get_provider",
    "register_provider",
]
