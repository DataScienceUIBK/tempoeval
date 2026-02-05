# LLM Providers API

Unified interface for interacting with Large Language Models.

## Azure OpenAI

```python
from tempoeval.llm import AzureOpenAIProvider

llm = AzureOpenAIProvider(
    deployment_name="gpt-4",
    api_version="2024-02-15-preview",
    # api_base and api_key read from env vars by default
)
```

## OpenAI (Standard)

```python
from tempoeval.llm import OpenAIProvider
llm = OpenAIProvider(model="gpt-4-turbo")
```

## Anthropic (Claude)

```python
from tempoeval.llm import AnthropicProvider
llm = AnthropicProvider(model="claude-3-opus-20240229")
```

## Google Gemini

```python
from tempoeval.llm import GeminiProvider
llm = GeminiProvider(model="gemini-1.5-pro")
```

## LiteLLM (Universal)
Support for 100+ models via LiteLLM wrapper.

```python
from tempoeval.llm import LiteLLMProvider
llm = LiteLLMProvider(model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1")
```
