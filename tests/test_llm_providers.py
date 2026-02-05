"""Tests for TempoEval LLM providers."""

import pytest
from tempoeval.llm import (
    BaseLLMProvider,
    get_provider,
    register_provider,
)


class TestProviderRegistry:
    """Tests for LLM provider registry."""

    def test_get_provider_unknown(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(ValueError):
            get_provider("nonexistent_provider")

    def test_get_provider_case_insensitive(self):
        """Test provider names are case insensitive."""
        try:
            get_provider("OPENAI", api_key="dummy")
        except ValueError:
            pytest.fail("Case insensitive lookup failed")
        except Exception:
            pass  # Other errors (like API key) are OK


class TestProviderImports:
    """Tests for provider imports."""

    def test_import_openai_provider(self):
        """Test importing OpenAI provider."""
        from tempoeval.llm import OpenAIProvider
        assert OpenAIProvider is not None

    def test_import_azure_provider(self):
        """Test importing Azure provider."""
        from tempoeval.llm import AzureOpenAIProvider
        assert AzureOpenAIProvider is not None

    def test_import_anthropic_provider(self):
        """Test importing Anthropic provider."""
        from tempoeval.llm import AnthropicProvider
        assert AnthropicProvider is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
